# 🚀 Deployment Guide - Modern Qweenone v2.0

Complete deployment guide for production environments **without Kubernetes**.

---

## 📋 Deployment Options

### Option 1: Docker Compose (Recommended)
**Best for:** Single-server deployments, development, staging

- ✅ Simple setup
- ✅ All services included
- ✅ Easy monitoring
- ✅ Local development friendly

### Option 2: Systemd Services
**Best for:** Direct host deployment, maximum performance

- ✅ Native performance
- ✅ System integration
- ✅ Auto-restart on failure
- ✅ Resource control

### Option 3: Docker Swarm
**Best for:** Multi-server deployments, high availability

- ✅ Container orchestration
- ✅ Load balancing
- ✅ Rolling updates
- ✅ Service discovery

---

## 🐳 Docker Compose Deployment

### Prerequisites

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version
```

### Production Deployment

```bash
# 1. Clone repository
git clone <repository-url>
cd qweenone

# 2. Configure environment
cp .env.example .env
nano .env  # Edit with your configuration

# 3. Build and start services
docker-compose -f docker-compose.modern.yml up -d

# 4. Verify services are running
docker-compose -f docker-compose.modern.yml ps

# 5. Check logs
docker-compose -f docker-compose.modern.yml logs -f

# 6. Test the system
docker-compose -f docker-compose.modern.yml exec qweenone-modern \
  python src/modern_main.py --status
```

### Service Management

```bash
# Start all services
docker-compose -f docker-compose.modern.yml up -d

# Stop all services
docker-compose -f docker-compose.modern.yml down

# Restart specific service
docker-compose -f docker-compose.modern.yml restart qweenone-modern

# View logs
docker-compose -f docker-compose.modern.yml logs -f qweenone-modern

# Scale services (if configured)
docker-compose -f docker-compose.modern.yml up -d --scale qweenone-modern=3

# Update and restart
docker-compose -f docker-compose.modern.yml pull
docker-compose -f docker-compose.modern.yml up -d
```

---

## ⚙️ Systemd Service Deployment

### Install as System Service

#### 1. Create Service File

```bash
sudo nano /etc/systemd/system/qweenone.service
```

```ini
[Unit]
Description=Qweenone Modern Agentic System
After=network.target redis.service postgresql.service

[Service]
Type=simple
User=qweenone
Group=qweenone
WorkingDirectory=/opt/qweenone
Environment="PATH=/opt/qweenone/venv/bin"
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=/opt/qweenone/.env

ExecStart=/opt/qweenone/venv/bin/python src/modern_main.py --task "monitor system"
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10

StandardOutput=append:/var/log/qweenone/output.log
StandardError=append:/var/log/qweenone/error.log

[Install]
WantedBy=multi-user.target
```

#### 2. Setup Installation

```bash
# Create user
sudo useradd -r -s /bin/bash qweenone

# Create directories
sudo mkdir -p /opt/qweenone
sudo mkdir -p /var/log/qweenone

# Clone repository
cd /opt/qweenone
sudo git clone <repository-url> .

# Setup Python virtual environment
sudo python3 -m venv venv
sudo /opt/qweenone/venv/bin/pip install -r requirements_modern.txt
sudo /opt/qweenone/venv/bin/playwright install chromium

# Set permissions
sudo chown -R qweenone:qweenone /opt/qweenone
sudo chown -R qweenone:qweenone /var/log/qweenone

# Configure environment
sudo cp .env.example .env
sudo nano .env  # Edit configuration
sudo chown qweenone:qweenone .env
sudo chmod 600 .env
```

#### 3. Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable qweenone

# Start service
sudo systemctl start qweenone

# Check status
sudo systemctl status qweenone

# View logs
sudo journalctl -u qweenone -f
```

---

## 🌊 Docker Swarm Deployment

### Initialize Swarm

```bash
# On manager node
docker swarm init

# On worker nodes (use token from init output)
docker swarm join --token <token> <manager-ip>:2377
```

### Create Stack

```bash
# Deploy stack
docker stack deploy -c docker-compose.modern.yml qweenone

# List services
docker stack services qweenone

# View logs
docker service logs -f qweenone_qweenone-modern

# Scale services
docker service scale qweenone_qweenone-modern=5

# Update service
docker service update qweenone_qweenone-modern --image qweenone:latest

# Remove stack
docker stack rm qweenone
```

---

## 📊 Monitoring & Observability

### Prometheus Metrics

Access: `http://localhost:9090`

**Key Metrics:**
- Task execution time
- Task success/failure rates
- LLM API costs
- Message queue depth
- Agent activity

**Example Queries:**
```promql
# Average task execution time
rate(task_execution_seconds_sum[5m]) / rate(task_execution_seconds_count[5m])

# Task failure rate
rate(task_failures_total[5m])

# API cost per minute
rate(llm_api_cost_total[1m])
```

### Grafana Dashboards

Access: `http://localhost:3000` (admin/admin)

**Pre-configured Dashboards:**
1. System Overview
2. Task Execution Metrics
3. LLM API Usage & Costs
4. A2A Communication Stats
5. Resource Utilization

### Application Logs

```bash
# Docker Compose logs
docker-compose -f docker-compose.modern.yml logs -f qweenone-modern

# Systemd logs
sudo journalctl -u qweenone -f

# Application log files
tail -f /app/logs/qweenone.log
tail -f /var/log/qweenone/output.log
```

---

## 🔐 Security Best Practices

### 1. API Key Management

```bash
# Use Docker secrets (Swarm mode)
echo "your_openai_key" | docker secret create openai_api_key -

# Or use environment files with restricted permissions
chmod 600 .env
chown qweenone:qweenone .env
```

### 2. Database Security

```bash
# PostgreSQL: Use strong passwords
POSTGRES_PASSWORD=$(openssl rand -base64 32)

# Enable SSL
POSTGRES_SSL_MODE=require

# Restrict network access
# In docker-compose.yml, don't expose port 5432 externally
```

### 3. Message Queue Security

```bash
# RabbitMQ: Strong credentials
RABBITMQ_DEFAULT_PASS=$(openssl rand -base64 32)

# Enable SSL/TLS
RABBITMQ_SSL_CERTFILE=/certs/cert.pem
RABBITMQ_SSL_KEYFILE=/certs/key.pem

# Use vhosts for isolation
RABBITMQ_DEFAULT_VHOST=/qweenone
```

### 4. Redis Security

```bash
# Set password
redis-server --requirepass $(openssl rand -base64 32)

# Disable dangerous commands
redis-server --rename-command FLUSHDB "" --rename-command FLUSHALL ""

# Enable SSL
redis-server --tls-port 6380 --tls-cert-file cert.pem --tls-key-file key.pem
```

---

## 🌍 Cloud Deployment

### AWS Deployment

#### Using EC2 + Docker Compose

```bash
# 1. Launch EC2 instance (t3.medium or larger)
# 2. Install Docker and Docker Compose
# 3. Clone and configure

sudo yum update -y
sudo yum install -y docker git
sudo service docker start
sudo usermod -a -G docker ec2-user

# Clone and deploy
git clone <repository-url>
cd qweenone
docker-compose -f docker-compose.modern.yml up -d
```

#### Using ECS (Elastic Container Service)

```bash
# 1. Push image to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker build -f Dockerfile.modern -t qweenone:latest .
docker tag qweenone:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/qweenone:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/qweenone:latest

# 2. Create task definition (see ecs-task-definition.json)
# 3. Create ECS service
# 4. Configure load balancer
```

### Google Cloud Platform

```bash
# Using Cloud Run
gcloud builds submit --tag gcr.io/PROJECT_ID/qweenone
gcloud run deploy qweenone --image gcr.io/PROJECT_ID/qweenone --platform managed

# Using Compute Engine + Docker
gcloud compute instances create qweenone-instance \
  --image-family=cos-stable \
  --image-project=cos-cloud \
  --machine-type=e2-medium

# SSH and deploy
gcloud compute ssh qweenone-instance
# ... then follow Docker Compose deployment
```

### DigitalOcean

```bash
# Using Droplet
doctl compute droplet create qweenone \
  --image docker-20-04 \
  --size s-2vcpu-4gb \
  --region nyc1

# SSH and deploy
doctl compute ssh qweenone
# ... then follow Docker Compose deployment
```

---

## 🔧 Maintenance

### Backup Procedures

```bash
# Backup PostgreSQL (Prefect data)
docker-compose -f docker-compose.modern.yml exec postgres \
  pg_dump -U prefect prefect > backup_prefect_$(date +%Y%m%d).sql

# Backup Redis data
docker-compose -f docker-compose.modern.yml exec redis \
  redis-cli --rdb /data/dump.rdb
docker cp qweenone-redis:/data/dump.rdb backup_redis_$(date +%Y%m%d).rdb

# Backup RabbitMQ definitions
docker-compose -f docker-compose.modern.yml exec rabbitmq \
  rabbitmqadmin export backup_rabbitmq_$(date +%Y%m%d).json
```

### Update Procedures

```bash
# 1. Pull latest code
git pull origin main

# 2. Check for breaking changes
git log --oneline

# 3. Update dependencies
pip install -r requirements_modern.txt --upgrade

# 4. Run migrations (if any)
# python scripts/migrate.py

# 5. Restart services
docker-compose -f docker-compose.modern.yml restart

# 6. Verify functionality
python src/modern_main.py --status
```

### Health Checks

```bash
# Check all services
docker-compose -f docker-compose.modern.yml ps

# Test Redis
redis-cli ping

# Test RabbitMQ
curl -u qweenone:qweenone_secret http://localhost:15672/api/overview

# Test PostgreSQL
psql -h localhost -U prefect -d prefect -c "SELECT 1"

# Test Qweenone system
python src/modern_main.py --status
```

---

## 📏 Resource Requirements

### Minimum Requirements

| Component | CPU | RAM | Disk |
|-----------|-----|-----|------|
| Qweenone Modern | 1 core | 1 GB | 5 GB |
| Redis | 0.5 core | 512 MB | 1 GB |
| RabbitMQ | 0.5 core | 512 MB | 2 GB |
| PostgreSQL | 0.5 core | 512 MB | 5 GB |
| **Total** | **2.5 cores** | **2.5 GB** | **13 GB** |

### Recommended Production

| Component | CPU | RAM | Disk |
|-----------|-----|-----|------|
| Qweenone Modern | 4 cores | 8 GB | 20 GB |
| Redis | 2 cores | 4 GB | 10 GB |
| RabbitMQ | 2 cores | 4 GB | 20 GB |
| PostgreSQL | 2 cores | 4 GB | 50 GB |
| Prometheus | 1 core | 2 GB | 20 GB |
| Grafana | 1 core | 1 GB | 5 GB |
| **Total** | **12 cores** | **23 GB** | **125 GB** |

---

## 🎯 Production Checklist

### Pre-Deployment
- [ ] All dependencies installed
- [ ] Environment variables configured
- [ ] API keys secured
- [ ] Database initialized
- [ ] Services tested individually
- [ ] Integration tests passed
- [ ] Backup strategy defined

### Security
- [ ] API keys in secrets/vault
- [ ] Database passwords changed from defaults
- [ ] SSL/TLS enabled for external connections
- [ ] Firewall rules configured
- [ ] Only necessary ports exposed
- [ ] User permissions restricted

### Monitoring
- [ ] Prometheus configured
- [ ] Grafana dashboards imported
- [ ] Alerting rules defined
- [ ] Log aggregation setup
- [ ] Health check endpoints working

### High Availability
- [ ] Multiple service replicas (if needed)
- [ ] Database replication configured
- [ ] Redis persistence enabled
- [ ] RabbitMQ clustering (for HA)
- [ ] Load balancer configured
- [ ] Backup automation setup

### Documentation
- [ ] Deployment process documented
- [ ] Configuration documented
- [ ] Runbooks created
- [ ] Team trained
- [ ] Troubleshooting guide available

---

## 🔥 Quick Production Setup

### Single Command Deployment

```bash
# Clone and deploy in one go
git clone <repository-url> qweenone && \
cd qweenone && \
cp .env.example .env && \
nano .env && \
docker-compose -f docker-compose.modern.yml up -d && \
echo "✅ Qweenone deployed! Access Prefect UI at http://localhost:4200"
```

---

## 📊 Monitoring Setup

### Prometheus Configuration

Already included in `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'qweenone'
    static_configs:
      - targets: ['qweenone-modern:8000']
  
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
  
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
```

### Grafana Dashboard Import

1. Access Grafana: `http://localhost:3000`
2. Login: `admin/admin`
3. Add Prometheus data source
4. Import dashboard JSON (see `grafana/dashboards/`)

---

## 🆘 Troubleshooting

### Container Won't Start

```bash
# Check container logs
docker-compose -f docker-compose.modern.yml logs qweenone-modern

# Check resource usage
docker stats

# Rebuild image
docker-compose -f docker-compose.modern.yml build --no-cache qweenone-modern
docker-compose -f docker-compose.modern.yml up -d
```

### Service Connection Issues

```bash
# Test service connectivity
docker-compose -f docker-compose.modern.yml exec qweenone-modern ping redis
docker-compose -f docker-compose.modern.yml exec qweenone-modern ping postgres

# Check network
docker network inspect qweenone-network

# Restart all services
docker-compose -f docker-compose.modern.yml restart
```

### Performance Issues

```bash
# Check resource usage
docker stats

# Increase resource limits in docker-compose.yml
services:
  qweenone-modern:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G

# Scale up
docker-compose -f docker-compose.modern.yml up -d --scale qweenone-modern=3
```

---

## 🔄 Backup & Recovery

### Automated Backup Script

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/qweenone/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup PostgreSQL
docker-compose -f docker-compose.modern.yml exec -T postgres \
  pg_dump -U prefect prefect > $BACKUP_DIR/prefect.sql

# Backup Redis
docker-compose -f docker-compose.modern.yml exec -T redis \
  redis-cli SAVE
docker cp qweenone-redis:/data/dump.rdb $BACKUP_DIR/redis.rdb

# Backup RabbitMQ
docker-compose -f docker-compose.modern.yml exec -T rabbitmq \
  rabbitmqadmin export $BACKUP_DIR/rabbitmq.json

# Backup code and config
tar -czf $BACKUP_DIR/code.tar.gz src/ .env

echo "✅ Backup completed: $BACKUP_DIR"
```

### Recovery Procedure

```bash
# 1. Stop services
docker-compose -f docker-compose.modern.yml down

# 2. Restore PostgreSQL
cat backup_prefect_20240101.sql | \
  docker-compose -f docker-compose.modern.yml exec -T postgres \
  psql -U prefect prefect

# 3. Restore Redis
docker cp backup_redis_20240101.rdb qweenone-redis:/data/dump.rdb
docker-compose -f docker-compose.modern.yml restart redis

# 4. Restore code
tar -xzf backup_code_20240101.tar.gz

# 5. Start services
docker-compose -f docker-compose.modern.yml up -d

# 6. Verify
python src/modern_main.py --status
```

---

## 🚀 Scaling Guide

### Horizontal Scaling

```bash
# Scale Qweenone service
docker-compose -f docker-compose.modern.yml up -d --scale qweenone-modern=5

# Add load balancer (nginx example)
# See nginx.conf for configuration
```

### Vertical Scaling

```yaml
# In docker-compose.modern.yml
services:
  qweenone-modern:
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G
        reservations:
          cpus: '4'
          memory: 8G
```

### Database Scaling

```bash
# PostgreSQL: Enable connection pooling
# Add PgBouncer

# Redis: Enable clustering
# See redis-cluster.conf

# RabbitMQ: Enable clustering
# See rabbitmq-cluster.conf
```

---

## 📞 Support & Maintenance

### Regular Maintenance Tasks

**Daily:**
- Check service status
- Monitor error logs
- Review API costs

**Weekly:**
- Backup databases
- Review performance metrics
- Update dependencies (dev/staging)

**Monthly:**
- Security updates
- Capacity planning review
- Performance optimization
- Documentation updates

---

## 🎉 Success Criteria

After deployment, verify:

✅ All services running and healthy  
✅ Prefect UI accessible  
✅ Can execute test task successfully  
✅ Metrics visible in Prometheus  
✅ Logs flowing correctly  
✅ Backup automation working  
✅ Team can access and operate system  

---

**Deployment complete! 🚀**
