# FailureDNA: AWS Cloud Deployment Guide

This guide details two production deployment options for **FailureDNA** on Amazon Web Services (AWS):
- **Option 1: AWS App Runner** (Fully managed, zero-infrastructure container service)
- **Option 2: AWS ECS on Fargate** (Serverless container orchestration)

---

## 1. Option 1: AWS App Runner (Recommended for Quick Deployment)

AWS App Runner provides fully managed container execution with automatic TLS/HTTPS, continuous auto-scaling, and simple health checking.

### Step 1: Push Docker Image to Amazon ECR

```bash
# 1. Authenticate Docker with Amazon ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# 2. Create ECR repository (if not already existing)
aws ecr create-repository --repository-name failuredna --region us-east-1

# 3. Tag and push the local Docker image
docker tag failuredna:latest <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/failuredna:latest
docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/failuredna:latest
```

### Step 2: Deploy Service via AWS App Runner

#### Via AWS Management Console:
1. Open the [AWS App Runner Console](https://console.aws.amazon.com/apprunner).
2. Click **Create service**.
3. Under **Source**:
   - Choose **Container registry** $\rightarrow$ **Amazon ECR**.
   - Browse and select `failuredna:latest`.
   - Set Deployment trigger to **Automatic**.
4. Under **Configure service**:
   - **Service name**: `failuredna-platform`
   - **Virtual CPU**: `1 vCPU`
   - **Memory**: `2 GB`
   - **Port**: `8000`
5. Under **Health check**:
   - **Protocol**: `HTTP`
   - **Path**: `/health`
   - **Interval**: `10 seconds`
6. Click **Create & deploy**.

#### Via AWS CloudFormation (Infrastructure-as-Code):
Deploy using the provided template ([aws/cloudformation.yml](file:///c:/Users/Harish/Downloads/FailureDNA/aws/cloudformation.yml)):
```bash
aws cloudformation deploy \
  --template-file aws/cloudformation.yml \
  --stack-name failuredna-stack \
  --parameter-overrides ImageUri=<AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/failuredna:latest \
  --capabilities CAPABILITY_NAMED_IAM
```

---

## 2. Option 2: AWS ECS on Fargate (Enterprise Orchestration)

For organizations integrating with existing VPCs, Application Load Balancers, or ECS clusters:

1. **Register Task Definition**:
   ```bash
   # Replace ACCOUNT_ID and REGION in aws/task-definition.json, then register:
   aws ecs register-task-definition --cli-input-json file://aws/task-definition.json
   ```

2. **Create or Update Service**:
   ```bash
   aws ecs create-service \
     --cluster failuredna-cluster \
     --service-name failuredna-service \
     --task-definition failuredna-service \
     --desired-count 1 \
     --launch-type FARGATE \
     --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}"
   ```

3. **Configure Application Load Balancer (ALB)**:
   - Target Group Protocol: `HTTP`, Port `8000`
   - Health Check Path: `/health`

---

## 3. Automated CI/CD via GitHub Actions

The repository includes [.github/workflows/deploy-aws.yml](file:///c:/Users/Harish/Downloads/FailureDNA/.github/workflows/deploy-aws.yml).

To enable automated ECR builds on git pushes:
1. In your GitHub repository $\rightarrow$ **Settings** $\rightarrow$ **Secrets**:
   - `AWS_ACCESS_KEY_ID`: IAM user access key with ECR permissions.
   - `AWS_SECRET_ACCESS_KEY`: IAM user secret key.
   - `AWS_REGION`: e.g. `us-east-1`.
   - `ECR_REPOSITORY`: e.g. `failuredna`.
2. Every push to `main` builds the multi-stage Docker image and publishes it to your AWS ECR repository.
