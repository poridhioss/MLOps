variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-southeast-1"
}

variable "k3s_token" {
  description = "K3s cluster token"
  type        = string
  sensitive   = true
}

variable "ssh_key_name" {
  description = "Name of SSH key pair"
  type        = string
  default     = "my-key"
}

provider "aws" {
  region = var.aws_region
}

resource "aws_vpc" "my_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "k3s-mlflow-vpc"
  }
}

resource "aws_subnet" "public_subnet" {
  vpc_id                  = aws_vpc.my_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "ap-southeast-1a"
  map_public_ip_on_launch = true

  tags = {
    Name = "k3s-mlflow-public-subnet"
  }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.my_vpc.id
  tags = {
    Name = "k3s-mlflow-internet-gateway"
  }
}

resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.my_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }

  tags = {
    Name = "k3s-mlflow-public-route-table"
  }
}

resource "aws_route_table_association" "public_association" {
  subnet_id      = aws_subnet.public_subnet.id
  route_table_id = aws_route_table.public_rt.id
}

# Security group for K3s cluster
resource "aws_security_group" "k3s_sg" {
  name        = "k3s-cluster-sg"
  description = "Security group for K3s cluster"
  vpc_id      = aws_vpc.my_vpc.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 6443
    to_port     = 6443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8472
    to_port     = 8472
    protocol    = "udp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  ingress {
    from_port   = 10250
    to_port     = 10250
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  ingress {
    from_port   = 30000
    to_port     = 32767
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "tcp"
    self      = true
  }

  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "udp"
    self      = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "k3s-security-group"
  }
}

# Security group for MinIO
resource "aws_security_group" "minio_sg" {
  name        = "minio-sg"
  description = "Security group for MinIO instance"
  vpc_id      = aws_vpc.my_vpc.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # MinIO API port
  ingress {
    from_port   = 9000
    to_port     = 9000
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  # MinIO Console port
  ingress {
    from_port   = 9001
    to_port     = 9001
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "minio-security-group"
  }
}

# Security group for MLflow
resource "aws_security_group" "mlflow_sg" {
  name        = "mlflow-sg"
  description = "Security group for MLflow tracking server"
  vpc_id      = aws_vpc.my_vpc.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # MLflow tracking server port
  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # PostgreSQL port (only from within VPC)
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "mlflow-security-group"
  }
}

# MinIO instance
resource "aws_instance" "minio_instance" {
  ami           = "ami-0672fd5b9210aa093"
  instance_type = "t2.micro"
  subnet_id     = aws_subnet.public_subnet.id
  key_name      = var.ssh_key_name

  vpc_security_group_ids = [aws_security_group.minio_sg.id]

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 10
    delete_on_termination = true
    encrypted             = true

    tags = {
      Name = "minio-root-volume"
    }
  }

  user_data = base64encode(<<-EOF
              #!/bin/bash
              set -e
              
              # Update system
              apt-get update
              apt-get upgrade -y
              
              # Create MinIO user and directories
              useradd -r minio-user
              mkdir -p /usr/local/share/minio
              mkdir -p /var/log/minio
              chown minio-user:minio-user /usr/local/share/minio
              chown minio-user:minio-user /var/log/minio
              
              # Download and install MinIO server
              wget https://dl.min.io/server/minio/release/linux-amd64/minio
              chmod +x minio
              mv minio /usr/local/bin/
              
              # Install MinIO client
              wget https://dl.min.io/client/mc/release/linux-amd64/mc
              chmod +x mc
              mv mc /usr/local/bin/
              
              # Start MinIO in background
              sudo -u minio-user MINIO_ROOT_USER=minioadmin MINIO_ROOT_PASSWORD=minioadmin123 \
                /usr/local/bin/minio server /usr/local/share/minio \
                --address :9000 --console-address :9001 > /var/log/minio/minio.log 2>&1 &
              
              # Wait for MinIO to start
              sleep 30
              
              # Configure MinIO client and create bucket
              /usr/local/bin/mc alias set local http://localhost:9000 minioadmin minioadmin123
              /usr/local/bin/mc mb local/ray-data || true
              
              echo "MinIO installation completed" >> /var/log/minio-install.log
              EOF
  )

  tags = {
    Name = "minio-instance"
    Role = "storage"
  }
}

# MLflow tracking server instance
resource "aws_instance" "mlflow_instance" {
  ami           = "ami-0672fd5b9210aa093"
  instance_type = "t3.small"
  subnet_id     = aws_subnet.public_subnet.id
  key_name      = var.ssh_key_name

  vpc_security_group_ids = [aws_security_group.mlflow_sg.id]

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 20
    delete_on_termination = true
    encrypted             = true

    tags = {
      Name = "mlflow-root-volume"
    }
  }

  user_data = base64encode(<<-EOF
              #!/bin/bash
              set -e
              
              # Update system
              apt-get update
              apt-get upgrade -y
              
              # Install Python and pip
              apt-get install -y python3 python3-pip python3-venv postgresql postgresql-contrib
              
              # Create MLflow user and directories
              useradd -r -m -s /bin/bash mlflow-user
              mkdir -p /var/log/mlflow
              chown mlflow-user:mlflow-user /var/log/mlflow
              
              # Setup PostgreSQL
              sudo -u postgres psql -c "CREATE DATABASE mlflow_db;"
              sudo -u postgres psql -c "CREATE USER mlflow_user WITH PASSWORD 'mlflow_password';"
              sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;"
              
              # Grant schema permissions (required for PostgreSQL 15+)
              sudo -u postgres psql -d mlflow_db -c "GRANT ALL PRIVILEGES ON SCHEMA public TO mlflow_user;"
              sudo -u postgres psql -d mlflow_db -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mlflow_user;"
              sudo -u postgres psql -d mlflow_db -c "GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mlflow_user;"
              sudo -u postgres psql -d mlflow_db -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO mlflow_user;"
              sudo -u postgres psql -d mlflow_db -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO mlflow_user;"
              
              # Configure PostgreSQL to accept local connections
              echo "host    mlflow_db    mlflow_user    127.0.0.1/32    md5" >> /etc/postgresql/*/main/pg_hba.conf
              systemctl restart postgresql
              
              # Install MLflow and dependencies as mlflow-user
              sudo -u mlflow-user bash << 'MLFLOW_SETUP'
              cd /home/mlflow-user
              python3 -m venv mlflow-env
              source mlflow-env/bin/activate
              pip install --upgrade pip
              pip install mlflow psycopg2-binary boto3
              MLFLOW_SETUP
              
              # Get MinIO endpoint from instance
              MINIO_IP="${aws_instance.minio_instance.private_ip}"
              
              # Create MLflow service with MinIO credentials
              cat > /etc/systemd/system/mlflow.service << MLFLOW_SERVICE
              [Unit]
              Description=MLflow Tracking Server
              After=network.target postgresql.service
              
              [Service]
              Type=simple
              User=mlflow-user
              WorkingDirectory=/home/mlflow-user
              Environment="PATH=/home/mlflow-user/mlflow-env/bin"
              Environment="AWS_ACCESS_KEY_ID=minioadmin"
              Environment="AWS_SECRET_ACCESS_KEY=minioadmin123"
              Environment="AWS_DEFAULT_REGION=us-east-1"
              Environment="MLFLOW_S3_ENDPOINT_URL=http://$MINIO_IP:9000"
              ExecStart=/home/mlflow-user/mlflow-env/bin/mlflow server \
                --backend-store-uri postgresql://mlflow_user:mlflow_password@localhost/mlflow_db \
                --default-artifact-root s3://ray-data/mlflow-artifacts \
                --serve-artifacts \
                --host 0.0.0.0 \
                --port 5000
              Restart=always
              StandardOutput=append:/var/log/mlflow/mlflow.log
              StandardError=append:/var/log/mlflow/mlflow.log
              
              [Install]
              WantedBy=multi-user.target
              MLFLOW_SERVICE
              
              # Enable and start MLflow service
              systemctl daemon-reload
              systemctl enable mlflow.service
              systemctl start mlflow.service
              
              echo "MLflow installation completed" >> /var/log/mlflow-install.log
              EOF
  )

  depends_on = [aws_instance.minio_instance]

  tags = {
    Name = "mlflow-tracking-server"
    Role = "mlflow"
  }
}

# K3s master instance
resource "aws_instance" "k3s_master" {
  ami           = "ami-0672fd5b9210aa093"
  instance_type = "t3.medium"
  subnet_id     = aws_subnet.public_subnet.id
  key_name      = var.ssh_key_name

  vpc_security_group_ids = [aws_security_group.k3s_sg.id]

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 20
    delete_on_termination = true
    encrypted             = true

    tags = {
      Name = "k3s-master-root-volume"
    }
  }

  user_data = base64encode(<<-EOF
              #!/bin/bash
              sleep 30
              apt-get update
              PRIVATE_IP=$(hostname -I | awk '{print $1}')
              export K3S_TOKEN="${var.k3s_token}"
              curl -sfL https://get.k3s.io | sh -s - server \
                --disable traefik \
                --node-ip=$PRIVATE_IP \
                --cluster-cidr=10.42.0.0/16 \
                --service-cidr=10.43.0.0/16 \
                --flannel-backend=vxlan
              sleep 15
              mkdir -p /home/ubuntu/.kube
              cp /etc/rancher/k3s/k3s.yaml /home/ubuntu/.kube/config
              sed -i "s/127.0.0.1/$PRIVATE_IP/g" /home/ubuntu/.kube/config
              chown ubuntu:ubuntu /home/ubuntu/.kube/config
              chmod 600 /home/ubuntu/.kube/config
              usermod -aG sudo ubuntu
              echo 'export KUBECONFIG=/home/ubuntu/.kube/config' >> /home/ubuntu/.bashrc
              echo 'export PATH=$PATH:/usr/local/bin' >> /home/ubuntu/.bashrc
              if [ ! -L /usr/local/bin/kubectl ]; then
                ln -s /usr/local/bin/k3s /usr/local/bin/kubectl
              fi
              EOF
  )

  tags = {
    Name = "k3s-master"
    Role = "master"
  }
}

# K3s worker instances
resource "aws_instance" "k3s_workers" {
  count         = 2
  ami           = "ami-0672fd5b9210aa093"
  instance_type = "t3.medium"
  subnet_id     = aws_subnet.public_subnet.id
  key_name      = var.ssh_key_name

  vpc_security_group_ids = [aws_security_group.k3s_sg.id]

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 20
    delete_on_termination = true
    encrypted             = true

    tags = {
      Name = "k3s-worker-${count.index + 1}-root-volume"
    }
  }

  user_data = base64encode(<<-EOF
              #!/bin/bash
              sleep 30
              apt-get update
              sleep 90
              PRIVATE_IP=$(hostname -I | awk '{print $1}')
              export K3S_TOKEN="${var.k3s_token}"
              export K3S_URL="https://${aws_instance.k3s_master.private_ip}:6443"
              curl -sfL https://get.k3s.io | sh -s - agent \
                --node-ip=$PRIVATE_IP
              if [ ! -L /usr/local/bin/kubectl ]; then
                ln -s /usr/local/bin/k3s /usr/local/bin/kubectl
              fi
              EOF
  )

  depends_on = [aws_instance.k3s_master]

  tags = {
    Name = "k3s-worker-${count.index + 1}"
    Role = "worker"
  }
}

# Outputs
output "minio_public_ip" {
  description = "Public IP of the MinIO instance"
  value       = aws_instance.minio_instance.public_ip
}

output "minio_private_ip" {
  description = "Private IP of the MinIO instance"
  value       = aws_instance.minio_instance.private_ip
}

output "mlflow_public_ip" {
  description = "Public IP of the MLflow tracking server"
  value       = aws_instance.mlflow_instance.public_ip
}

output "mlflow_private_ip" {
  description = "Private IP of the MLflow tracking server"
  value       = aws_instance.mlflow_instance.private_ip
}

output "master_public_ip" {
  description = "Public IP of the K3s master node"
  value       = aws_instance.k3s_master.public_ip
}

output "worker_public_ips" {
  description = "Public IPs of the K3s worker nodes"
  value       = aws_instance.k3s_workers[*].public_ip
}

output "master_private_ip" {
  description = "Private IP of the K3s master node"
  value       = aws_instance.k3s_master.private_ip
}

output "worker_private_ips" {
  description = "Private IPs of the K3s worker nodes"
  value       = aws_instance.k3s_workers[*].private_ip
}


