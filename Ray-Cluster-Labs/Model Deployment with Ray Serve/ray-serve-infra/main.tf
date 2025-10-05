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

# Data source for Ubuntu AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# VPC Configuration
resource "aws_vpc" "ray_serve_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "ray-serve-vpc"
  }
}

# Public Subnet
resource "aws_subnet" "public_subnet" {
  vpc_id                  = aws_vpc.ray_serve_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true

  tags = {
    Name = "ray-serve-public-subnet"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.ray_serve_vpc.id
  
  tags = {
    Name = "ray-serve-internet-gateway"
  }
}

# Route Table
resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.ray_serve_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }

  tags = {
    Name = "ray-serve-public-route-table"
  }
}

# Route Table Association
resource "aws_route_table_association" "public_association" {
  subnet_id      = aws_subnet.public_subnet.id
  route_table_id = aws_route_table.public_rt.id
}

# Security Group for K3s Cluster
resource "aws_security_group" "k3s_sg" {
  name        = "ray-serve-k3s-sg"
  description = "Security group for K3s cluster"
  vpc_id      = aws_vpc.ray_serve_vpc.id

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH access"
  }

  # Kubernetes API server
  ingress {
    from_port   = 6443
    to_port     = 6443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Kubernetes API"
  }

  # Flannel VXLAN
  ingress {
    from_port   = 8472
    to_port     = 8472
    protocol    = "udp"
    cidr_blocks = ["10.0.0.0/16"]
    description = "Flannel VXLAN"
  }

  # Kubelet API
  ingress {
    from_port   = 10250
    to_port     = 10250
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
    description = "Kubelet API"
  }

  # NodePort services
  ingress {
    from_port   = 30000
    to_port     = 32767
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "NodePort services"
  }

  # Allow all traffic within security group
  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "tcp"
    self      = true
    description = "Internal cluster communication (TCP)"
  }

  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "udp"
    self      = true
    description = "Internal cluster communication (UDP)"
  }

  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }

  tags = {
    Name = "ray-serve-k3s-security-group"
  }
}

# K3s Master Instance
resource "aws_instance" "k3s_master" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t3.medium"
  subnet_id     = aws_subnet.public_subnet.id
  key_name      = var.ssh_key_name

  vpc_security_group_ids = [aws_security_group.k3s_sg.id]

  associate_public_ip_address = true

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 20
    delete_on_termination = true
    encrypted             = true

    tags = {
      Name = "ray-serve-master-root-volume"
    }
  }

  user_data = base64encode(<<-EOF
              #!/bin/bash
              set -e
              
              # Wait for system to be ready
              sleep 30
              
              # Update system
              apt-get update
              
              # Get private IP
              PRIVATE_IP=$(hostname -I | awk '{print $1}')
              
              # Install K3s server
              export K3S_TOKEN="${var.k3s_token}"
              curl -sfL https://get.k3s.io | sh -s - server \
                --disable traefik \
                --node-ip=$PRIVATE_IP \
                --cluster-cidr=10.42.0.0/16 \
                --service-cidr=10.43.0.0/16 \
                --flannel-backend=vxlan
              
              # Wait for K3s to be ready
              sleep 15
              
              # Configure kubectl for ubuntu user
              mkdir -p /home/ubuntu/.kube
              cp /etc/rancher/k3s/k3s.yaml /home/ubuntu/.kube/config
              sed -i "s/127.0.0.1/$PRIVATE_IP/g" /home/ubuntu/.kube/config
              chown ubuntu:ubuntu /home/ubuntu/.kube/config
              chmod 600 /home/ubuntu/.kube/config
              
              # Add ubuntu to sudo group
              usermod -aG sudo ubuntu
              
              # Setup environment variables
              echo 'export KUBECONFIG=/home/ubuntu/.kube/config' >> /home/ubuntu/.bashrc
              echo 'export PATH=$PATH:/usr/local/bin' >> /home/ubuntu/.bashrc
              
              # Create kubectl symlink
              if [ ! -L /usr/local/bin/kubectl ]; then
                ln -s /usr/local/bin/k3s /usr/local/bin/kubectl
              fi
              
              echo "K3s master setup complete" > /var/log/k3s-master-setup.log
              EOF
  )

  tags = {
    Name = "ray-serve-k3s-master"
    Role = "master"
  }
}

# K3s Worker Instances
resource "aws_instance" "k3s_workers" {
  count         = 2
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t3.medium"
  subnet_id     = aws_subnet.public_subnet.id
  key_name      = var.ssh_key_name

  vpc_security_group_ids = [aws_security_group.k3s_sg.id]

  associate_public_ip_address = true

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 20
    delete_on_termination = true
    encrypted             = true

    tags = {
      Name = "ray-serve-worker-${count.index + 1}-root-volume"
    }
  }

  user_data = base64encode(<<-EOF
              #!/bin/bash
              set -e
              
              # Wait for system to be ready
              sleep 30
              
              # Update system
              apt-get update
              
              # Wait for master to be ready
              sleep 90
              
              # Get private IP
              PRIVATE_IP=$(hostname -I | awk '{print $1}')
              
              # Install K3s agent
              export K3S_TOKEN="${var.k3s_token}"
              export K3S_URL="https://${aws_instance.k3s_master.private_ip}:6443"
              curl -sfL https://get.k3s.io | sh -s - agent \
                --node-ip=$PRIVATE_IP
              
              # Create kubectl symlink
              if [ ! -L /usr/local/bin/kubectl ]; then
                ln -s /usr/local/bin/k3s /usr/local/bin/kubectl
              fi
              
              echo "K3s worker setup complete" > /var/log/k3s-worker-setup.log
              EOF
  )

  depends_on = [aws_instance.k3s_master]

  tags = {
    Name = "ray-serve-k3s-worker-${count.index + 1}"
    Role = "worker"
  }
}

# Outputs
output "master_public_ip" {
  description = "Public IP of the K3s master node"
  value       = aws_instance.k3s_master.public_ip
}

output "master_private_ip" {
  description = "Private IP of the K3s master node"
  value       = aws_instance.k3s_master.private_ip
}

output "worker_public_ips" {
  description = "Public IPs of the K3s worker nodes"
  value       = aws_instance.k3s_workers[*].public_ip
}

output "worker_private_ips" {
  description = "Private IPs of the K3s worker nodes"
  value       = aws_instance.k3s_workers[*].private_ip
}