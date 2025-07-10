# Script 5: .env.example
# File Path: .env.example
#
# Datacenter Monitoring System - Environment Configuration Template
# Copy this file to .env and fill in your actual values

# ====================
# DATABASE CONFIGURATION
# ====================

# MySQL Database Settings
MYSQL_HOST=localhost
MYSQL_USER=datacenter_user
MYSQL_PASSWORD=your_secure_password_here
MYSQL_DATABASE=datacenter_monitoring
MYSQL_PORT=3306

# Database Pool Settings
DB_POOL_SIZE=32

# ====================
# MODEL CONFIGURATION
# ====================

# AI Model Paths
DETECTION_MODEL_PATH=models/yolov11l.pt
PPE_DETECTION_MODEL_PATH=models/ppe_detection.pt
POSE_ESTIMATION_MODEL_PATH=models/yolov11l-pose.pt

# Model Settings
POSE_ENABLED=false
PERSON_DETECTION_CONFIDENCE=0.5
GENERAL_DETECTION_CONFIDENCE=0.6
PPE_CONFIDENCE_THRESHOLD=0.7

# ====================
# PROCESSING CONFIGURATION
# ====================

# System Processing
MAX_RETRIES=3
PROCESSING_TIMEOUT=3600
NUMBER_OF_CAMERAS=8
BATCH_SIZE=8
BATCH_TIMEOUT=0.5
MAX_QUEUE_SIZE=200

# Camera Processing
READER_FPS_LIMIT=4
NUM_BATCH_THREADS=2
MAX_PARALLEL_CAMERAS=4

# Activity Level FPS
ACTIVITY_LEVEL_HIGH=10
ACTIVITY_LEVEL_MEDIUM=4
ACTIVITY_LEVEL_LOW=2
FPS_ACTIVITY_LEVEL=medium

# Tracking Configuration
TRACKING_THRESHOLD=3
MAX_AGE=30
MIN_MA=3

# ====================
# STORAGE CONFIGURATION
# ====================

# Local Storage
FRAMES_OUTPUT_DIR=frames
MEDIA_PREFERENCE=image
AUTO_RECORDING_ENABLED=false

# Video Recording
VIDEO_BUFFER_SIZE=30
VIDEO_FPS=4
PRE_EVENT_SECONDS=3
POST_EVENT_SECONDS=7
VIDEO_EXTENSION=mp4
VIDEO_CODEC=avc1

# Google Cloud Storage (Optional)
GCP_PROJECT=your-gcp-project-id
GCP_BUCKET_NAME=datacenter-monitoring-storage
GOOGLE_APPLICATION_CREDENTIALS=./secrets/gcp-credentials.json

# Database Writer
DB_WRITER_BATCH_SIZE=5

# ====================
# DATACENTER SPECIFIC SETTINGS
# ====================

# PPE Detection
PPE_DETECTION_ENABLED=true

# Tailgating Detection
TAILGATING_TIME_WINDOW=10
MAX_PEOPLE_PER_ENTRY=1
ENTRY_ZONE_BUFFER=2.0

# Loitering Detection
LOITERING_THRESHOLD=300
MOVEMENT_THRESHOLD=1.0
LOITERING_CHECK_INTERVAL=30

# Intrusion Detection
INTRUSION_SENSITIVITY=high
RESTRICTED_ZONE_BUFFER=0.5
INTRUSION_CONFIDENCE_THRESHOLD=0.8

# Camera Tamper Detection
TAMPER_DETECTION_ENABLED=true
FRAME_DIFF_THRESHOLD=0.8
OBSTRUCTION_THRESHOLD=0.9
TAMPER_CHECK_INTERVAL=60

# Occupancy Limits
SERVER_ROOM_OCCUPANCY=5
COMMON_AREA_OCCUPANCY=20
CORRIDOR_OCCUPANCY=10
CRITICAL_ZONE_OCCUPANCY=2
ENTRY_ZONE_OCCUPANCY=3

# ====================
# ALERT CONFIGURATION
# ====================

# Event Settings
EVENT_COOLDOWN=120

# SMS Notifications (Twilio)
SMS_ENABLED=true
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=+1234567890

# Alert Recipients (comma-separated)
ALERT_PHONE_NUMBERS=+919876543210,+919876543211,+919876543212

# ====================
# CONFIGURATION MANAGEMENT
# ====================

# Configuration Source
USE_CONFIG_FILES=false
CAMERA_CONFIG_DIR=configs/cameras

# ====================
# LOGGING CONFIGURATION
# ====================

# Log Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# ====================
# SECURITY SETTINGS
# ====================

# JWT Settings (for future API)
JWT_SECRET_KEY=your_super_secret_jwt_key_change_this_in_production
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=30

# API Settings (for future API)
API_PORT=8000
API_HOST=0.0.0.0
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# ====================
# DEVELOPMENT SETTINGS
# ====================

# Development Mode
DEBUG_MODE=false
TESTING_MODE=false

# Performance Monitoring
ENABLE_PERFORMANCE_LOGGING=true
ENABLE_AUDIT_LOGGING=true

# ====================
# HARDWARE SETTINGS
# ====================

# GPU Settings
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_FRACTION=0.8

# System Resource Limits
MAX_CPU_USAGE=80
MAX_MEMORY_USAGE=85

# ====================
# ADVANCED SETTINGS
# ====================

# Frame Processing
SKIP_FRAMES=0
FRAME_RESIZE_ENABLED=false
FRAME_RESIZE_WIDTH=1280
FRAME_RESIZE_HEIGHT=720

# Cache Settings
ENABLE_FRAME_CACHE=true
CACHE_SIZE_MB=512

# Network Settings
RTSP_TIMEOUT=30
RTSP_RETRY_ATTEMPTS=3
RTSP_RECONNECT_DELAY=5

# ====================
# MONITORING & HEALTH
# ====================

# Health Check
HEALTH_CHECK_INTERVAL=60
ENABLE_HEALTH_ENDPOINT=true

# Metrics
ENABLE_METRICS_COLLECTION=true
METRICS_EXPORT_INTERVAL=300

# System Monitoring
MONITOR_CPU=true
MONITOR_MEMORY=true
MONITOR_GPU=true
MONITOR_DISK=true

# ====================
# BACKUP & RECOVERY
# ====================

# Database Backup
AUTO_BACKUP_ENABLED=false
BACKUP_INTERVAL_HOURS=24
BACKUP_RETENTION_DAYS=30
BACKUP_LOCATION=./backups

# ====================
# EXTERNAL INTEGRATIONS
# ====================

# Email Notifications (future)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
EMAIL_FROM=datacenter-monitoring@yourcompany.com

# Webhook Notifications (future)
WEBHOOK_ENABLED=false
WEBHOOK_URL=https://your-webhook-endpoint.com/alerts
WEBHOOK_SECRET=your_webhook_secret

# Slack Integration (future)
SLACK_ENABLED=false
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
SLACK_CHANNEL=#security-alerts

# ====================
# DEPLOYMENT SETTINGS
# ====================

# Environment
ENVIRONMENT=development
VERSION=1.0.0

# Docker Settings
DOCKER_NETWORK=datacenter-monitoring-network
DOCKER_COMPOSE_PROJECT=datacenter-monitoring

# ====================
# COMPLIANCE & AUDIT
# ====================

# Data Retention
EVENT_RETENTION_DAYS=365
LOG_RETENTION_DAYS=90
VIDEO_RETENTION_DAYS=30
IMAGE_RETENTION_DAYS=60

# Privacy Settings
ANONYMIZE_LOGS=false
ENCRYPT_SENSITIVE_DATA=true

# Compliance
GDPR_COMPLIANCE=false
HIPAA_COMPLIANCE=false
SOC2_COMPLIANCE=true

# ====================
# NOTES
# ====================

# 1. Copy this file to .env and fill in your actual values
# 2. Never commit .env file to version control
# 3. Use strong passwords and secrets in production
# 4. Enable SSL/TLS for production deployments
# 5. Regularly rotate API keys and secrets
# 6. Monitor resource usage and adjust limits accordingly
# 7. Test backup and recovery procedures regularly
# 8. Keep model files updated for best detection accuracy
# 9. Configure proper firewall rules for camera access
# 10. Enable audit logging for compliance requirements