#!/usr/bin/env python3
"""
Script 2: config.py
File Path: src/config.py

Datacenter Monitoring System - Configuration Management

This module handles:
1. System configuration and environment variables
2. Datacenter-specific monitoring parameters
3. Detection model settings
4. Alert and notification configurations
5. Event type definitions for datacenter use cases
"""

import os
from dotenv import load_dotenv

load_dotenv()

class DatacenterConfig:
    """Main configuration class for datacenter monitoring system"""
    
    # GCP Storage settings
    GCP_PROJECT = os.getenv('GCP_PROJECT')
    BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")
    STORAGE_BASE_URL = f"https://storage.googleapis.com/{BUCKET_NAME}" if os.getenv("GCP_BUCKET_NAME") else None
    
    # Detection Model Paths
    DETECTION_MODEL_PATH = os.getenv('DETECTION_MODEL_PATH', 'models/yolov11l.pt')
    PPE_DETECTION_MODEL_PATH = os.getenv('PPE_DETECTION_MODEL_PATH', 'models/ppe_detection.pt')
    POSE_ENABLED = os.getenv('POSE_ENABLED', 'False').lower() == 'true'
    POSE_ESTIMATION_MODEL_PATH = os.getenv('POSE_ESTIMATION_MODEL_PATH', 'models/yolov11l-pose.pt')

    # MySQL Database Configuration
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_USER = os.getenv('MYSQL_USER')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'datacenter_monitoring')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', 3306))

    # Database Pool Configuration
    DB_POOL_NAME = 'datacenter_pool'
    DB_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', 32))

    # Processing Configuration
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
    PROCESSING_TIMEOUT = int(os.getenv('PROCESSING_TIMEOUT', 3600))  # 1 hour
    FRAMES_OUTPUT_DIR = os.getenv('FRAMES_OUTPUT_DIR', 'frames')

    # Camera and Tracking Configuration
    TRACKING_THRESHOLD = int(os.getenv('TRACKING_THRESHOLD', 3))
    MAX_AGE = int(os.getenv('MAX_AGE', 30))
    MIN_MA = int(os.getenv('MIN_MA', 3))
    
    # Media Storage Configuration
    MEDIA_PREFERENCE = os.getenv('MEDIA_PREFERENCE', 'image')  # "image" or "video"
    EVENT_COOLDOWN = int(os.getenv('EVENT_COOLDOWN', 120))  # Seconds between similar events
    AUTO_RECORDING_ENABLED = os.getenv('AUTO_RECORDING_ENABLED', 'False').lower() == 'true'
    
    # Batch Processing Configuration
    NUMBER_OF_CAMERAS = int(os.getenv('NUMBER_OF_CAMERAS', 4))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', NUMBER_OF_CAMERAS))
    BATCH_TIMEOUT = float(os.getenv('BATCH_TIMEOUT', 0.5))
    MAX_QUEUE_SIZE = int(os.getenv('MAX_QUEUE_SIZE', 200))
    
    # Multi-Camera Configuration  
    NUM_CAMERA_READERS = NUMBER_OF_CAMERAS
    READER_FPS_LIMIT = int(os.getenv('READER_FPS_LIMIT', 4))
    NUM_BATCH_THREADS = int(os.getenv('NUM_BATCH_THREADS', 2))
    
    # Activity Level FPS Settings
    ACTIVITY_LEVEL_HIGH = int(os.getenv('ACTIVITY_LEVEL_HIGH', 10))    # Critical zones
    ACTIVITY_LEVEL_MEDIUM = int(os.getenv('ACTIVITY_LEVEL_MEDIUM', 4)) # Server rooms
    ACTIVITY_LEVEL_LOW = int(os.getenv('ACTIVITY_LEVEL_LOW', 2))       # Corridors
    FPS_ACTIVITY_LEVEL = os.getenv('FPS_ACTIVITY_LEVEL', 'medium')
    
    # Parallel Processing Configuration
    MAX_PARALLEL_CAMERAS = int(os.getenv('MAX_PARALLEL_CAMERAS', 4))
    
    # Database Writer Configuration
    DB_WRITER_BATCH_SIZE = int(os.getenv('DB_WRITER_BATCH_SIZE', 5))
    
    # Video Recording Configuration
    VIDEO_BUFFER_SIZE = int(os.getenv('VIDEO_BUFFER_SIZE', 30))
    VIDEO_FPS = int(os.getenv('VIDEO_FPS', 4))
    PRE_EVENT_SECONDS = int(os.getenv('PRE_EVENT_SECONDS', 3))
    POST_EVENT_SECONDS = int(os.getenv('POST_EVENT_SECONDS', 7))
    VIDEO_EXTENSION = os.getenv('VIDEO_EXTENSION', 'mp4')
    VIDEO_CODEC = os.getenv('VIDEO_CODEC', 'avc1')
    
    # Configuration Management
    CAMERA_CONFIG_DIR = os.getenv('CAMERA_CONFIG_DIR', 'configs/cameras')
    USE_CONFIG_FILES = os.getenv('USE_CONFIG_FILES', 'False').lower() == 'true'
    
    # Twilio SMS Configuration for Alerts
    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
    TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
    
    # Alert phone numbers (comma-separated)
    _ALERT_NUMBERS = os.getenv('ALERT_PHONE_NUMBERS', '')
    ALERT_PHONE_NUMBERS = [num.strip() for num in _ALERT_NUMBERS.split(',') if num.strip()]
    
    # SMS Configuration
    SMS_ENABLED = os.getenv('SMS_ENABLED', 'True').lower() == 'true'
    
    # DATACENTER-SPECIFIC CONFIGURATIONS
    
    # PPE Detection Settings
    PPE_DETECTION_ENABLED = os.getenv('PPE_DETECTION_ENABLED', 'True').lower() == 'true'
    REQUIRED_PPE_CLASSES = ['hard_hat', 'safety_vest', 'safety_glasses']
    PPE_CONFIDENCE_THRESHOLD = float(os.getenv('PPE_CONFIDENCE_THRESHOLD', 0.7))
    
    # Tailgating Detection
    TAILGATING_TIME_WINDOW = int(os.getenv('TAILGATING_TIME_WINDOW', 10))  # seconds
    MAX_PEOPLE_PER_ENTRY = int(os.getenv('MAX_PEOPLE_PER_ENTRY', 1))
    ENTRY_ZONE_BUFFER = float(os.getenv('ENTRY_ZONE_BUFFER', 2.0))  # meters
    
    # Loitering Detection
    LOITERING_THRESHOLD = int(os.getenv('LOITERING_THRESHOLD', 300))  # seconds (5 minutes)
    MOVEMENT_THRESHOLD = float(os.getenv('MOVEMENT_THRESHOLD', 1.0))  # meters
    LOITERING_CHECK_INTERVAL = int(os.getenv('LOITERING_CHECK_INTERVAL', 30))  # seconds
    
    # Intrusion Detection
    INTRUSION_SENSITIVITY = os.getenv('INTRUSION_SENSITIVITY', 'high')  # low, medium, high
    RESTRICTED_ZONE_BUFFER = float(os.getenv('RESTRICTED_ZONE_BUFFER', 0.5))  # meters
    INTRUSION_CONFIDENCE_THRESHOLD = float(os.getenv('INTRUSION_CONFIDENCE_THRESHOLD', 0.8))
    
    # Camera Tamper Detection
    TAMPER_DETECTION_ENABLED = os.getenv('TAMPER_DETECTION_ENABLED', 'True').lower() == 'true'
    FRAME_DIFF_THRESHOLD = float(os.getenv('FRAME_DIFF_THRESHOLD', 0.8))
    OBSTRUCTION_THRESHOLD = float(os.getenv('OBSTRUCTION_THRESHOLD', 0.9))
    TAMPER_CHECK_INTERVAL = int(os.getenv('TAMPER_CHECK_INTERVAL', 60))  # seconds
    
    # People Counting and Occupancy
    OCCUPANCY_LIMITS = {
        'server_room': int(os.getenv('SERVER_ROOM_OCCUPANCY', 5)),
        'common_area': int(os.getenv('COMMON_AREA_OCCUPANCY', 20)),
        'corridor': int(os.getenv('CORRIDOR_OCCUPANCY', 10)),
        'critical_zone': int(os.getenv('CRITICAL_ZONE_OCCUPANCY', 2)),
        'entry_zone': int(os.getenv('ENTRY_ZONE_OCCUPANCY', 3))
    }
    
    # Zone Security Levels
    SECURITY_LEVELS = {
        'public': 1,
        'restricted': 2, 
        'high_security': 3,
        'critical': 4
    }
    
    # Detection Confidence Thresholds
    PERSON_DETECTION_CONFIDENCE = float(os.getenv('PERSON_DETECTION_CONFIDENCE', 0.5))
    GENERAL_DETECTION_CONFIDENCE = float(os.getenv('GENERAL_DETECTION_CONFIDENCE', 0.6))
    
    @classmethod
    def get_db_pool_config(cls):
        """Get database pool configuration dictionary"""
        return {
            'pool_name': cls.DB_POOL_NAME,
            'pool_size': cls.DB_POOL_SIZE,
            'host': cls.MYSQL_HOST,
            'user': cls.MYSQL_USER,
            'password': cls.MYSQL_PASSWORD,
            'database': cls.MYSQL_DATABASE,
            'port': cls.MYSQL_PORT
        }
    
    @classmethod
    def get_occupancy_limit(cls, zone_type: str) -> int:
        """Get occupancy limit for a specific zone type"""
        return cls.OCCUPANCY_LIMITS.get(zone_type, 10)  # Default to 10 if not specified
    
    @classmethod
    def get_activity_fps(cls, activity_level: str) -> int:
        """Get FPS based on activity level"""
        if activity_level == 'high':
            return cls.ACTIVITY_LEVEL_HIGH
        elif activity_level == 'low':
            return cls.ACTIVITY_LEVEL_LOW
        else:  # medium or default
            return cls.ACTIVITY_LEVEL_MEDIUM


class DatacenterEventTypes:
    """Event types specific to datacenter monitoring"""
    
    # Phase 1 - Core events (a1, a2)
    TAILGATING = "tailgating"
    INTRUSION = "intrusion" 
    PPE_VIOLATION = "ppe_violation"
    CAMERA_TAMPER = "camera_tamper"
    LOITERING = "loitering"
    PEOPLE_COUNTING = "people_counting"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    
    # Phase 2 - Advanced events (future)
    FACE_MISMATCH = "face_mismatch"
    SLIP_FALL = "slip_fall"
    HEAT_ANOMALY = "heat_anomaly"
    EQUIPMENT_TAMPERING = "equipment_tampering"
    EMERGENCY_EVACUATION = "emergency_evacuation"
    
    # Event metadata for UI and processing
    METADATA = {
        TAILGATING: {
            "display_name": "Tailgating Detected",
            "description": "Multiple people entered through single access point",
            "default_severity": "high",
            "camera_types": ["dc_entry_monitor"],
            "requires_immediate_action": True,
            "auto_notify": True
        },
        INTRUSION: {
            "display_name": "Intrusion Detected",
            "description": "Unauthorized access to restricted area",
            "default_severity": "critical", 
            "camera_types": ["dc_server_room", "dc_perimeter", "dc_critical_zone"],
            "requires_immediate_action": True,
            "auto_notify": True
        },
        PPE_VIOLATION: {
            "display_name": "PPE Compliance Violation",
            "description": "Required safety equipment not detected",
            "default_severity": "medium",
            "camera_types": ["dc_server_room", "dc_critical_zone"],
            "requires_immediate_action": False,
            "auto_notify": True
        },
        CAMERA_TAMPER: {
            "display_name": "Camera Tampering",
            "description": "Camera obstruction or movement detected", 
            "default_severity": "high",
            "camera_types": ["all"],
            "requires_immediate_action": True,
            "auto_notify": True
        },
        LOITERING: {
            "display_name": "Loitering Detected",
            "description": "Person remaining in area beyond allowed time",
            "default_severity": "medium",
            "camera_types": ["dc_corridor", "dc_common_area"],
            "requires_immediate_action": False,
            "auto_notify": False
        },
        PEOPLE_COUNTING: {
            "display_name": "Occupancy Alert", 
            "description": "Area occupancy exceeded maximum limit",
            "default_severity": "low",
            "camera_types": ["dc_common_area", "dc_server_room"],
            "requires_immediate_action": False,
            "auto_notify": False
        },
        UNAUTHORIZED_ACCESS: {
            "display_name": "Unauthorized Access",
            "description": "Access without proper authorization",
            "default_severity": "high",
            "camera_types": ["dc_entry_monitor", "dc_server_room", "dc_critical_zone"],
            "requires_immediate_action": True,
            "auto_notify": True
        }
    }
    
    @classmethod
    def get_all_events(cls):
        """Return all event types as a list"""
        return [attr for attr in dir(cls) 
                if not attr.startswith('_') and 
                attr not in ['METADATA', 'get_all_events', 'is_valid_event', 'get_event_metadata',
                           'get_critical_events', 'get_auto_notify_events']]
    
    @classmethod
    def is_valid_event(cls, event_type: str) -> bool:
        """Check if an event type is valid"""
        return event_type in cls.METADATA
        
    @classmethod
    def get_event_metadata(cls, event_type: str) -> dict:
        """Get metadata for a specific event type"""
        return cls.METADATA.get(event_type, {})
    
    @classmethod
    def get_critical_events(cls) -> list:
        """Get list of critical event types that require immediate action"""
        critical_events = []
        for event_type, metadata in cls.METADATA.items():
            if metadata.get('requires_immediate_action', False):
                critical_events.append(event_type)
        return critical_events
    
    @classmethod
    def get_auto_notify_events(cls) -> list:
        """Get list of event types that trigger automatic notifications"""
        auto_notify_events = []
        for event_type, metadata in cls.METADATA.items():
            if metadata.get('auto_notify', False):
                auto_notify_events.append(event_type)
        return auto_notify_events


class DatacenterCameraTypes:
    """Camera types for different datacenter monitoring scenarios"""
    
    # Phase 1 camera types
    ENTRY_MONITOR = "dc_entry_monitor"      # Entry/exit points - tailgating detection
    SERVER_ROOM = "dc_server_room"          # Server rooms - PPE compliance, intrusion
    CORRIDOR = "dc_corridor"                # Corridors - loitering, people counting  
    PERIMETER = "dc_perimeter"              # Perimeter - intrusion detection
    CRITICAL_ZONE = "dc_critical_zone"      # Critical infrastructure - high security
    COMMON_AREA = "dc_common_area"          # Common areas - occupancy monitoring
    
    # Camera type configurations
    TYPE_CONFIGS = {
        ENTRY_MONITOR: {
            "description": "Entry/exit point monitoring",
            "primary_events": ["tailgating", "unauthorized_access"],
            "secondary_events": ["people_counting"],
            "required_zones": ["entry_zone"],
            "activity_level": "high",
            "detection_classes": ["person"]
        },
        SERVER_ROOM: {
            "description": "Server room monitoring", 
            "primary_events": ["ppe_violation", "intrusion"],
            "secondary_events": ["people_counting", "unauthorized_access"],
            "required_zones": ["server_zone"],
            "activity_level": "medium",
            "detection_classes": ["person", "hard_hat", "safety_vest"]
        },
        CORRIDOR: {
            "description": "Corridor and hallway monitoring",
            "primary_events": ["loitering", "people_counting"],
            "secondary_events": ["unauthorized_access"],
            "required_zones": ["common_zone"],
            "activity_level": "low", 
            "detection_classes": ["person"]
        },
        PERIMETER: {
            "description": "Perimeter security monitoring",
            "primary_events": ["intrusion"],
            "secondary_events": ["people_counting"],
            "required_zones": ["perimeter_zone"],
            "activity_level": "medium",
            "detection_classes": ["person"]
        },
        CRITICAL_ZONE: {
            "description": "Critical infrastructure monitoring",
            "primary_events": ["intrusion", "ppe_violation"],
            "secondary_events": ["unauthorized_access", "people_counting"],
            "required_zones": ["critical_zone"],
            "activity_level": "high",
            "detection_classes": ["person", "hard_hat", "safety_vest"]
        },
        COMMON_AREA: {
            "description": "Common area monitoring",
            "primary_events": ["people_counting", "loitering"],
            "secondary_events": [],
            "required_zones": ["common_zone"],
            "activity_level": "low",
            "detection_classes": ["person"]
        }
    }
    
    @classmethod
    def get_all_types(cls):
        """Get all camera types"""
        return [cls.ENTRY_MONITOR, cls.SERVER_ROOM, cls.CORRIDOR, 
                cls.PERIMETER, cls.CRITICAL_ZONE, cls.COMMON_AREA]
    
    @classmethod
    def get_type_config(cls, camera_type: str) -> dict:
        """Get configuration for a specific camera type"""
        return cls.TYPE_CONFIGS.get(camera_type, {})
    
    @classmethod
    def is_valid_type(cls, camera_type: str) -> bool:
        """Check if camera type is valid"""
        return camera_type in cls.TYPE_CONFIGS


class DatacenterZoneTypes:
    """Zone types for different areas in datacenter"""
    
    ENTRY_ZONE = "entry_zone"
    SERVER_ZONE = "server_zone"
    RESTRICTED_ZONE = "restricted_zone"
    COMMON_ZONE = "common_zone"
    PERIMETER_ZONE = "perimeter_zone"
    CRITICAL_ZONE = "critical_zone"
    
    ZONE_CONFIGS = {
        ENTRY_ZONE: {
            "description": "Entry and exit points",
            "security_level": "restricted",
            "default_occupancy_limit": 3,
            "monitoring_rules": ["tailgating", "unauthorized_access"]
        },
        SERVER_ZONE: {
            "description": "Server room areas",
            "security_level": "high_security", 
            "default_occupancy_limit": 5,
            "monitoring_rules": ["ppe_violation", "intrusion", "people_counting"],
            "required_ppe": ["hard_hat", "safety_vest"]
        },
        RESTRICTED_ZONE: {
            "description": "Restricted access areas",
            "security_level": "high_security",
            "default_occupancy_limit": 2,
            "monitoring_rules": ["intrusion", "unauthorized_access"]
        },
        COMMON_ZONE: {
            "description": "Common areas and corridors",
            "security_level": "restricted",
            "default_occupancy_limit": 20,
            "monitoring_rules": ["loitering", "people_counting"]
        },
        PERIMETER_ZONE: {
            "description": "Perimeter and outdoor areas",
            "security_level": "restricted",
            "default_occupancy_limit": 10,
            "monitoring_rules": ["intrusion"]
        },
        CRITICAL_ZONE: {
            "description": "Mission critical infrastructure",
            "security_level": "critical",
            "default_occupancy_limit": 2,
            "monitoring_rules": ["intrusion", "ppe_violation", "unauthorized_access"],
            "required_ppe": ["hard_hat", "safety_vest", "safety_glasses"]
        }
    }
    
    @classmethod
    def get_zone_config(cls, zone_type: str) -> dict:
        """Get configuration for a specific zone type"""
        return cls.ZONE_CONFIGS.get(zone_type, {})
    
    @classmethod
    def get_all_zone_types(cls):
        """Get all zone types"""
        return list(cls.ZONE_CONFIGS.keys())


# Detection model class mappings - will be imported in video processor
DATACENTER_CAMERA_MODEL_MAPPING = {
    DatacenterCameraTypes.ENTRY_MONITOR: 'DatacenterEntryMonitor',
    DatacenterCameraTypes.SERVER_ROOM: 'ServerRoomMonitor', 
    DatacenterCameraTypes.CORRIDOR: 'CorridorMonitor',
    DatacenterCameraTypes.PERIMETER: 'PerimeterMonitor',
    DatacenterCameraTypes.CRITICAL_ZONE: 'CriticalZoneMonitor',
    DatacenterCameraTypes.COMMON_AREA: 'CommonAreaMonitor'
}

# Export main classes for easy importing
__all__ = [
    'DatacenterConfig',
    'DatacenterEventTypes', 
    'DatacenterCameraTypes',
    'DatacenterZoneTypes',
    'DATACENTER_CAMERA_MODEL_MAPPING'
]