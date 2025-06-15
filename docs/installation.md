# Installation Guide

This guide walks you through installing DataMCPServerAgent on your system, from basic setup to advanced configurations with cloud integrations.

## 📋 Prerequisites

### System Requirements

#### Minimum Requirements
- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.9 or higher
- **Memory**: 4GB RAM
- **Storage**: 2GB free space
- **Network**: Internet connection for package installation

#### Recommended Requirements
- **Memory**: 8GB+ RAM
- **Storage**: 10GB+ SSD storage
- **CPU**: Multi-core processor for better performance
- **Network**: Stable high-speed internet

### Software Dependencies

#### Required
- **Python 3.9+**: [Download Python](https://python.org/downloads)
- **Git**: [Download Git](https://git-scm.com/downloads)

#### Optional (for advanced features)
- **Node.js 18+**: For web UI ([Download Node.js](https://nodejs.org))
- **Docker**: For containerized deployment ([Download Docker](https://docker.com))
- **Redis**: For distributed memory ([Download Redis](https://redis.io))
- **PostgreSQL**: For persistent storage ([Download PostgreSQL](https://postgresql.org))

## 🚀 Quick Installation

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/DataMCPServerAgent.git
cd DataMCPServerAgent

# Run automated setup script
chmod +x scripts/setup.sh
./scripts/setup.sh
```

The setup script will:
- Install Python dependencies
- Set up environment configuration
- Initialize the database
- Start the development server

### Option 2: Manual Installation

#### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/DataMCPServerAgent.git
cd DataMCPServerAgent
```

#### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

#### Step 4: Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (see Configuration section below)
nano .env  # or your preferred editor
```

#### Step 5: Initialize Database

```bash
# Run database migrations
python app/cli.py migrate

# Create initial data (optional)
python app/cli.py seed
```

#### Step 6: Start the System

```bash
# Start API server
python app/main_consolidated.py api

# Verify installation
curl http://localhost:8003/health
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root with the following configuration:

```bash
# Core Application Settings
APP_NAME=DataMCPServerAgent
APP_VERSION=2.0.0
APP_ENV=development
DEBUG=true

# Server Configuration
API_HOST=localhost
API_PORT=8003
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/datamcp
# Or use SQLite for development:
# DATABASE_URL=sqlite:///./data/datamcp.db

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=3600

# Security Settings
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here
API_KEY_HEADER=X-API-Key

# Agent Configuration
DEFAULT_AGENT_TYPE=research
MAX_CONCURRENT_AGENTS=10
ENABLE_LEARNING=true
MEMORY_BACKEND=postgresql

# Cloud Provider Settings (optional)
# AWS
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1

# Azure
AZURE_SUBSCRIPTION_ID=your_subscription_id
AZURE_CLIENT_ID=your_client_id
AZURE_CLIENT_SECRET=your_client_secret

# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
GCP_PROJECT_ID=your_project_id

# Third-party APIs
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
BRIGHT_DATA_API_KEY=your_bright_data_key
```

### Database Setup

#### PostgreSQL (Recommended for Production)

```bash
# Install PostgreSQL
# Ubuntu/Debian:
sudo apt-get install postgresql postgresql-contrib

# macOS with Homebrew:
brew install postgresql

# Windows: Download from https://postgresql.org

# Create database and user
sudo -u postgres psql
CREATE DATABASE datamcp;
CREATE USER datamcp_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE datamcp TO datamcp_user;
\q

# Update .env file
DATABASE_URL=postgresql://datamcp_user:your_password@localhost:5432/datamcp
```

#### SQLite (For Development)

```bash
# SQLite is included with Python, no additional installation needed
# Update .env file
DATABASE_URL=sqlite:///./data/datamcp.db

# Create data directory
mkdir -p data
```

#### Redis (Optional, for Distributed Features)

```bash
# Install Redis
# Ubuntu/Debian:
sudo apt-get install redis-server

# macOS with Homebrew:
brew install redis

# Windows: Use WSL or Docker

# Start Redis service
# Ubuntu/Debian:
sudo systemctl start redis-server
sudo systemctl enable redis-server

# macOS:
brew services start redis

# Verify Redis is running
redis-cli ping
# Should return: PONG

# Update .env file
REDIS_URL=redis://localhost:6379/0
```

### Web UI Installation

#### Install Node.js Dependencies

```bash
cd agent-ui
npm install
# or with yarn:
# yarn install
```

#### Configure Web UI

```bash
# Create environment file
cp .env.example .env.local

# Edit configuration
nano .env.local
```

```bash
# .env.local content
NEXT_PUBLIC_API_URL=http://localhost:8003
NEXT_PUBLIC_WS_URL=ws://localhost:8003
NEXT_PUBLIC_APP_NAME=DataMCP Agent UI
```

#### Start Web UI

```bash
# Development mode
npm run dev

# Production build
npm run build
npm start
```

Access the web UI at `http://localhost:3000`

## 🐳 Docker Installation

### Quick Start with Docker Compose

```bash
# Clone repository
git clone https://github.com/your-org/DataMCPServerAgent.git
cd DataMCPServerAgent

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### Manual Docker Setup

```bash
# Build the image
docker build -t datamcp-agent .

# Run with Docker
docker run -d \
  --name datamcp-agent \
  -p 8003:8003 \
  -e DATABASE_URL=sqlite:///./data/datamcp.db \
  -v $(pwd)/data:/app/data \
  datamcp-agent

# Check container status
docker ps

# View logs
docker logs datamcp-agent
```

## ☁️ Cloud Installation

### AWS Deployment

#### Using AWS CLI

```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure

# Deploy using CloudFormation
aws cloudformation create-stack \
  --stack-name datamcp-stack \
  --template-body file://deployment/aws/cloudformation.yaml \
  --parameters ParameterKey=InstanceType,ParameterValue=t3.medium \
  --capabilities CAPABILITY_IAM
```

#### Using Terraform

```bash
# Install Terraform
# Follow instructions at https://terraform.io

# Navigate to Terraform configuration
cd deployment/terraform/aws

# Initialize Terraform
terraform init

# Plan deployment
terraform plan

# Apply configuration
terraform apply
```

### Azure Deployment

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Create resource group
az group create --name datamcp-rg --location eastus

# Deploy using ARM template
az deployment group create \
  --resource-group datamcp-rg \
  --template-file deployment/azure/template.json \
  --parameters @deployment/azure/parameters.json
```

### Google Cloud Deployment

```bash
# Install Google Cloud SDK
# Follow instructions at https://cloud.google.com/sdk

# Initialize gcloud
gcloud init

# Create project (if needed)
gcloud projects create datamcp-project

# Set project
gcloud config set project datamcp-project

# Deploy using Cloud Run
gcloud run deploy datamcp-agent \
  --image gcr.io/datamcp-project/agent:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 🧪 Verify Installation

### Basic Health Check

```bash
# Check API health
curl http://localhost:8003/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "2.0.0",
  "components": {
    "database": "healthy",
    "cache": "healthy",
    "agents": "healthy"
  }
}
```

### Test Agent Creation

```bash
# Create a test agent
curl -X POST http://localhost:8003/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "research",
    "name": "Test Agent",
    "configuration": {
      "max_iterations": 5
    }
  }'

# Expected response:
{
  "agent_id": "agent-123",
  "agent_type": "research",
  "name": "Test Agent",
  "status": "active",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Run Integration Tests

```bash
# Run basic integration tests
python -m pytest tests/integration/test_basic.py -v

# Run full test suite
python -m pytest tests/ -v --cov=app
```

## 🔧 Troubleshooting

### Common Issues

#### Port Already in Use

```bash
# Check what's using port 8003
lsof -i :8003

# Kill process if needed
kill -9 <PID>

# Or use different port
API_PORT=8004 python app/main_consolidated.py api
```

#### Database Connection Issues

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql -h localhost -U datamcp_user -d datamcp

# Check SQLite file permissions
ls -la data/datamcp.db
chmod 664 data/datamcp.db
```

#### Redis Connection Issues

```bash
# Check Redis is running
redis-cli ping

# Check Redis logs
sudo journalctl -u redis-server

# Test connection
redis-cli -h localhost -p 6379
```

#### Python Environment Issues

```bash
# Check Python version
python --version

# Check virtual environment
which python

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### Log Analysis

```bash
# Check application logs
tail -f logs/app.log

# Check specific component logs
grep "ERROR" logs/app.log
grep "agent" logs/app.log

# Enable debug logging
LOG_LEVEL=DEBUG python app/main_consolidated.py api
```

### Performance Issues

```bash
# Check system resources
htop
df -h
free -h

# Monitor database performance
# PostgreSQL:
psql -c "SELECT * FROM pg_stat_activity;"

# Check Redis memory usage
redis-cli info memory
```

## 📦 Production Installation

### Security Hardening

```bash
# Generate secure secrets
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Set secure file permissions
chmod 600 .env
chmod 700 data/

# Configure firewall (Ubuntu)
sudo ufw allow 8003/tcp
sudo ufw enable
```

### Performance Optimization

```bash
# Install production WSGI server
pip install gunicorn uvicorn[standard]

# Start with Gunicorn
gunicorn app.main_consolidated:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8003 \
  --timeout 300
```

### Monitoring Setup

```bash
# Install monitoring dependencies
pip install prometheus-client statsd

# Configure monitoring
MONITORING_ENABLED=true
METRICS_PORT=9090
```

## 🎯 Next Steps

After successful installation:

1. **Explore the API**: Visit `http://localhost:8003/docs` for interactive API documentation
2. **Try the Web UI**: Access `http://localhost:3000` for the graphical interface
3. **Read the Tutorials**: Check out [Usage Guide](usage.md) for examples
4. **Join the Community**: Connect with other users in [GitHub Discussions](https://github.com/your-org/DataMCPServerAgent/discussions)

## 📞 Getting Help

If you encounter issues during installation:

1. **Check Troubleshooting**: Review the troubleshooting section above
2. **Search Issues**: Look through [GitHub Issues](https://github.com/your-org/DataMCPServerAgent/issues)
3. **Ask for Help**: Create a new issue with detailed error information
4. **Join Discord**: Get real-time help from the community

---

**Installation successful!** 🎉 You're ready to start building with DataMCPServerAgent.