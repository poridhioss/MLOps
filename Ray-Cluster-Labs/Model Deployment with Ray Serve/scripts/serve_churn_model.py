from ray import serve
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import pickle
import urllib.request
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Ray Serve Churn Prediction API",
    description="Predict customer churn using XGBoost model deployed with Ray Serve",
    version="1.0.0"
)

# Pydantic Models
class CustomerData(BaseModel):
    """Input schema for customer data"""
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    
    # Numerical features
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Senior citizen (0 or 1)")
    tenure: int = Field(..., ge=0, description="Months with company")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges")
    
    # Categorical features
    gender: Literal["Male", "Female"]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", 
        "Mailed check", 
        "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ]
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST001",
                "SeniorCitizen": 0,
                "tenure": 12,
                "MonthlyCharges": 65.5,
                "gender": "Male",
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check"
            }
        }

class PredictionResponse(BaseModel):
    """Output schema for prediction"""
    customer_id: Optional[str] = None
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    will_churn: bool = Field(..., description="Binary prediction")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    risk_level: Literal["Low", "Medium", "High"] = Field(..., description="Risk level")

class BatchPredictionRequest(BaseModel):
    """Input schema for batch predictions"""
    customers: List[CustomerData] = Field(..., description="List of customers")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    framework: str

# Ray Serve Deployment
@serve.deployment(
    ray_actor_options={
        "num_cpus": 1,
        "memory": 512 * 1024 * 1024  # 512MB per replica
    },
    autoscaling_config={
        "min_replicas": 2,
        "max_replicas": 5,
        "target_ongoing_requests": 10,
    }
)
@serve.ingress(app)
class ChurnPredictorServe:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = None
        
        # GitHub model URL - Updated with correct raw URL
        self.github_model_url = "https://raw.githubusercontent.com/poridhioss/MLOps/main/Ray-Cluster-Labs/Model%20Deployment%20with%20Ray%20Serve/artifacts/churn_xgboost_model_mlflow.pkl"
        
        # Load model on initialization
        self._load_model()
    
    def _load_model(self):
        """Load model from GitHub"""
        try:
            logger.info(f"Loading model from GitHub: {self.github_model_url}")
            
            # Download from GitHub
            with urllib.request.urlopen(self.github_model_url) as response:
                model_bytes = response.read()
            
            # Deserialize
            model_data = pickle.loads(model_bytes)
            self.model = model_data['model']
            self.label_encoders = model_data['label_encoders']
            
            # Define feature columns (same as training)
            self.feature_columns = [
                'SeniorCitizen', 'tenure', 'MonthlyCharges',
                'gender', 'Partner', 'Dependents', 'PhoneService',
                'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod'
            ]
            
            logger.info("✓ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _preprocess(self, data: dict) -> pd.DataFrame:
        """Preprocess customer data"""
        df = pd.DataFrame([data])
        
        categorical_columns = [
            'gender', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]
        
        for col in categorical_columns:
            if col in df.columns and col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df[self.feature_columns]
    
    @app.get("/", response_model=dict)
    async def root(self):
        """Root endpoint with API information"""
        return {
            "message": "Ray Serve Churn Prediction API",
            "version": "1.0.0",
            "framework": "Ray Serve",
            "autoscaling": "2-5 replicas",
            "docs": "/docs",
            "health": "/health"
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health(self):
        """Health check endpoint"""
        return {
            "status": "healthy",
            "model_loaded": self.model is not None,
            "framework": "Ray Serve"
        }
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(self, customer: CustomerData):
        """
        Predict churn for a single customer
        
        Args:
            customer: Customer data with demographic and service information
            
        Returns:
            Prediction with churn probability and classification
        """
        try:
            # Preprocess
            X = self._preprocess(customer.dict())
            
            # Predict
            dmatrix = xgb.DMatrix(X)
            churn_prob = float(self.model.predict(dmatrix)[0])
            will_churn = churn_prob > 0.5
            confidence = abs(churn_prob - 0.5) * 2  # Convert to 0-1 scale
            
            # Determine risk level
            if churn_prob < 0.3:
                risk_level = "Low"
            elif churn_prob < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            return PredictionResponse(
                customer_id=customer.customer_id or "unknown",
                churn_probability=round(churn_prob, 4),
                will_churn=bool(will_churn),
                confidence=round(confidence, 4),
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    @app.post("/predict-batch", response_model=List[PredictionResponse])
    async def predict_batch(self, request: BatchPredictionRequest):
        """
        Predict churn for multiple customers
        
        Args:
            request: Batch of customer data
            
        Returns:
            List of predictions for each customer
        """
        try:
            logger.info(f"Processing batch prediction for {len(request.customers)} customers")
            
            results = []
            for customer in request.customers:
                result = await self.predict(customer)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    
    @app.get("/model-info")
    async def get_model_info(self):
        """Get information about the loaded model"""
        try:
            return {
                "model_loaded": self.model is not None,
                "model_source": "GitHub",
                "github_url": self.github_model_url,
                "feature_count": len(self.feature_columns) if self.feature_columns else 0,
                "features": self.feature_columns,
                "autoscaling": {
                    "min_replicas": 2,
                    "max_replicas": 5,
                    "target_requests_per_replica": 10
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

# Deployment entry point
deployment = ChurnPredictorServe.bind()

# Main function for deployment
if __name__ == "__main__":
    import ray
    
    # Initialize Ray
    ray.init(address="auto", ignore_reinit_error=True)
    
    # Start Ray Serve
    serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})
    
    # Deploy the application
    serve.run(
        deployment,
        name="churn-predictor",
        route_prefix="/"
    )
    
    logger.info("="*60)
    logger.info("Ray Serve Deployment Successful!")
    logger.info("="*60)
    logger.info("Application: churn-predictor")
    logger.info("Autoscaling: 2-5 replicas")
    logger.info("Endpoint: http://0.0.0.0:8000")
    logger.info("Docs: http://0.0.0.0:8000/docs")
    logger.info("="*60)