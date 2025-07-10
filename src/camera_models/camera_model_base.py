#!/usr/bin/env python3
"""
Script 11: camera_model_base.py
File Path: src/camera_models/camera_model_base.py

Datacenter Monitoring System - Base Camera Model Class

This module provides:
1. Base class for all datacenter camera models
2. Common tracking and detection functionality
3. Event handling and media storage
4. PPE detection capabilities
5. Zone management and security level validation
6. Audit logging and compliance features
"""

import cv2
import numpy as np
import os
import uuid
import time
import threading
from collections import deque
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional, Any
from twilio.rest import Client
import pytz

from .kalman_track import DatacenterObjectTracker
from logger import setup_datacenter_logger, audit_logger, performance_logger
from config import DatacenterConfig, DatacenterEventTypes, DatacenterZoneTypes

class DatacenterCameraModelBase:
    """
    Base class for all datacenter camera models.
    Provides common functionality for tracking, detection, event handling, and compliance.
    """
    
    def __init__(self, camera_id: int, datacenter_id: int, zones: Optional[Dict] = None, 
                 rules: Optional[List] = None, settings: Optional[Dict] = None, 
                 db=None, db_writer=None, frames_base_dir: str = 'frames'):
        """
        Initialize the datacenter camera model base class
        
        Args:
            camera_id: Camera identifier
            datacenter_id: Datacenter identifier  
            zones: Zone definitions for this camera
            rules: Monitoring rules for this camera
            settings: Camera-specific settings
            db: Database instance
            db_writer: Database writer for batch operations
            frames_base_dir: Base directory for frame storage
        """
        # Initialize logger with datacenter context
        self.logger = setup_datacenter_logger(
            f'camera_model_{camera_id}', 
            f'camera_model_{camera_id}.log',
            datacenter_id=str(datacenter_id),
            camera_id=str(camera_id)
        )
        self.logger.info(f"Initializing datacenter camera model for camera {camera_id} in datacenter {datacenter_id}")
        
        # Store identifiers
        self.camera_id = camera_id
        self.datacenter_id = datacenter_id
        self.zones = zones or {}
        self.rules = rules or []
        self.settings = settings or {}
        
        # Database connections
        self.db = db
        self.db_writer = db_writer
        
        # Storage configuration
        self.frames_base_dir = frames_base_dir
        self.camera_output_dir = os.path.join(frames_base_dir, f"datacenter_{datacenter_id}", f"camera_{camera_id}")
        self.video_output_dir = os.path.join(self.camera_output_dir, "clips")
        os.makedirs(self.camera_output_dir, exist_ok=True)
        os.makedirs(self.video_output_dir, exist_ok=True)
        
        # Initialize object tracker
        self.object_tracker = DatacenterObjectTracker(
            max_age=DatacenterConfig.MAX_AGE,
            min_hits=DatacenterConfig.MIN_MA,
            iou_threshold=0.3
        )
        
        # Tracking state
        self.tracked_objects = {}
        self.total_object_count = 0
        self.frame_count = 0
        
        # PPE detection configuration
        self.ppe_detection_enabled = DatacenterConfig.PPE_DETECTION_ENABLED
        self.required_ppe_classes = DatacenterConfig.REQUIRED_PPE_CLASSES
        self.ppe_confidence_threshold = DatacenterConfig.PPE_CONFIDENCE_THRESHOLD
        
        # Event handling
        self.recent_events = {}
        self.event_cooldown = DatacenterConfig.EVENT_COOLDOWN
        self.tracking_threshold = DatacenterConfig.TRACKING_THRESHOLD
        
        # Media storage settings
        self.media_preference = self.settings.get('media_preference', DatacenterConfig.MEDIA_PREFERENCE)
        self.auto_recording_enabled = self.settings.get('auto_recording_enabled', DatacenterConfig.AUTO_RECORDING_ENABLED)
        
        # Frame buffer for video recording
        self.buffer_size = DatacenterConfig.VIDEO_BUFFER_SIZE
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.timestamp_buffer = deque(maxlen=self.buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Video recording settings
        self.video_fps = DatacenterConfig.VIDEO_FPS
        self.pre_event_seconds = DatacenterConfig.PRE_EVENT_SECONDS
        self.post_event_seconds = DatacenterConfig.POST_EVENT_SECONDS
        self.video_extension = DatacenterConfig.VIDEO_EXTENSION
        self.video_codec = DatacenterConfig.VIDEO_CODEC
        
        # Recording cooldown
        self.recording_cooldown = DatacenterConfig.EVENT_COOLDOWN
        self.last_recording_time = 0
        
        # Camera tamper detection
        self.tamper_detection_enabled = DatacenterConfig.TAMPER_DETECTION_ENABLED
        self.last_frame_hash = None
        self.tamper_check_count = 0
        
        # Statistics tracking
        self.stats = {
            'frames_processed': 0,
            'events_detected': 0,
            'people_detected': 0,
            'ppe_violations': 0,
            'security_events': 0,
            'frames_saved': 0,
            'videos_saved': 0,
            'last_processed_time': None,
            'start_time': time.time(),
            'average_fps': 0,
            'detection_accuracy': 0
        }
        
        # Multi-camera coordination
        self.enable_individual_events = True
        self.current_people_count = 0
        self.current_unauthorized_count = 0
        
        # Initialize SMS notifications
        self._init_sms_notifications()
        
        # Zone color mapping for visualization
        self.zone_colors = {
            'entry_zone': (255, 255, 0),      # Yellow
            'server_zone': (0, 255, 255),     # Cyan  
            'restricted_zone': (0, 0, 255),   # Red
            'common_zone': (0, 255, 0),       # Green
            'perimeter_zone': (255, 0, 255),  # Magenta
            'critical_zone': (128, 0, 128)    # Purple
        }
        
        self.logger.info(f"Datacenter camera model initialized successfully")
    
    def _init_sms_notifications(self):
        """Initialize Twilio SMS client for security alerts"""
        self.sms_enabled = False
        self.twilio_client = None
        
        try:
            if (DatacenterConfig.SMS_ENABLED and 
                DatacenterConfig.TWILIO_ACCOUNT_SID and 
                DatacenterConfig.TWILIO_AUTH_TOKEN and
                DatacenterConfig.TWILIO_PHONE_NUMBER and
                DatacenterConfig.ALERT_PHONE_NUMBERS):
                
                self.twilio_client = Client(
                    DatacenterConfig.TWILIO_ACCOUNT_SID, 
                    DatacenterConfig.TWILIO_AUTH_TOKEN
                )
                self.sms_enabled = True
                self.logger.info("SMS notifications initialized successfully")
                
            else:
                self.logger.warning("SMS notifications disabled: missing Twilio configuration")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize SMS notifications: {str(e)}")
            self.sms_enabled = False
    
    def process_frame(self, frame: np.ndarray, timestamp: float, 
                     detection_result: Any = None, ppe_result: Any = None) -> Tuple[np.ndarray, Dict]:
        """
        Main frame processing entry point
        
        Args:
            frame: Video frame to process
            timestamp: Frame timestamp
            detection_result: Person detection results
            ppe_result: PPE detection results
            
        Returns:
            Tuple of (annotated_frame, processing_results)
        """
        start_time = time.time()
        
        # Update frame statistics
        self.frame_count += 1
        self.stats['frames_processed'] += 1
        self.stats['last_processed_time'] = timestamp
        
        # Add frame to buffer for video recording
        with self.buffer_lock:
            self.frame_buffer.append(frame.copy())
            self.timestamp_buffer.append(timestamp)
        
        # Check for camera tampering
        if self.tamper_detection_enabled:
            tamper_detected = self._check_camera_tamper(frame)
            if tamper_detected:
                self._handle_tamper_event(frame, timestamp)
        
        # Perform camera-specific processing
        try:
            annotated_frame, results = self._process_frame_impl(
                frame, timestamp, detection_result, ppe_result
            )
            
            # Log performance metrics
            processing_time = time.time() - start_time
            performance_logger.log_processing_stats(
                camera_id=str(self.camera_id),
                fps=1.0 / processing_time if processing_time > 0 else 0,
                batch_size=1,
                processing_time=processing_time,
                queue_size=len(self.frame_buffer)
            )
            
            return annotated_frame, results
            
        except Exception as e:
            self.logger.error(f"Error in frame processing: {str(e)}", exc_info=True)
            return frame, {}
    
    def _process_frame_impl(self, frame: np.ndarray, timestamp: float, 
                           detection_result: Any, ppe_result: Any) -> Tuple[np.ndarray, Dict]:
        """
        Implementation-specific frame processing - to be overridden by subclasses
        
        Args:
            frame: Video frame
            timestamp: Frame timestamp  
            detection_result: Person detection results
            ppe_result: PPE detection results
            
        Returns:
            Tuple of (annotated_frame, results)
        """
        # Default implementation - should be overridden
        return frame, {}
    
    def detect_people_and_ppe(self, frame: np.ndarray, detection_result: Any, 
                             ppe_result: Any = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract people and PPE detections from model results
        
        Args:
            frame: Video frame
            detection_result: Person detection results from YOLO
            ppe_result: PPE detection results from PPE model
            
        Returns:
            Tuple of (people_detections, ppe_detections)
        """
        people_detections = []
        ppe_detections = []
        
        # Process person detections
        if detection_result and hasattr(detection_result, 'boxes'):
            for i, box in enumerate(detection_result.boxes):
                try:
                    class_id = int(box.cls[0]) if box.cls.numel() > 0 else -1
                    class_name = detection_result.names.get(class_id, "unknown")
                    
                    if class_name == 'person':
                        confidence = float(box.conf[0]) if box.conf.numel() > 0 else 0
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        
                        # Calculate center and dimensions
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        person_detection = {
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'center': (center_x, center_y),
                            'width': width,
                            'height': height,
                            'aspect_ratio': width / height if height > 0 else 1.0
                        }
                        
                        people_detections.append(person_detection)
                        
                except Exception as e:
                    self.logger.error(f"Error processing person detection: {e}")
        
        # Process PPE detections if available
        if self.ppe_detection_enabled and ppe_result and hasattr(ppe_result, 'boxes'):
            for i, box in enumerate(ppe_result.boxes):
                try:
                    class_id = int(box.cls[0]) if box.cls.numel() > 0 else -1
                    class_name = ppe_result.names.get(class_id, "unknown")
                    confidence = float(box.conf[0]) if box.conf.numel() > 0 else 0
                    
                    if confidence >= self.ppe_confidence_threshold:
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        
                        ppe_detection = {
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'center': ((x1 + x2) / 2, (y1 + y2) / 2)
                        }
                        
                        ppe_detections.append(ppe_detection)
                        
                except Exception as e:
                    self.logger.error(f"Error processing PPE detection: {e}")
        
        return people_detections, ppe_detections
    
    def update_object_tracking(self, people_detections: List[Dict]) -> List[Dict]:
        """
        Update object tracking with new people detections
        
        Args:
            people_detections: List of person detection dictionaries
            
        Returns:
            List of tracked objects with track IDs
        """
        try:
            # Convert detections to tracker format
            detection_array = []
            for detection in people_detections:
                center_x, center_y = detection['center']
                aspect_ratio = detection['aspect_ratio']
                height = detection['height']
                confidence = detection['confidence']
                
                # Format: [center_x, center_y, aspect_ratio, height, confidence]
                detection_array.append([center_x, center_y, aspect_ratio, height, confidence])
            
            # Update tracker
            if detection_array:
                tracked_objects_array, total_count = self.object_tracker.update(np.array(detection_array))
            else:
                tracked_objects_array, total_count = self.object_tracker.update()
            
            self.total_object_count = total_count
            
            # Convert tracker output back to detection format
            tracked_detections = []
            for i in range(tracked_objects_array.shape[0]):
                track = tracked_objects_array[i]
                center_x, center_y, aspect_ratio, height, track_id = track
                track_id = int(track_id)
                
                # Convert back to bbox format
                width = aspect_ratio * height
                x1 = center_x - width / 2
                y1 = center_y - height / 2
                x2 = center_x + width / 2
                y2 = center_y + height / 2
                
                # Update tracked objects dictionary
                if track_id not in self.tracked_objects:
                    self.tracked_objects[track_id] = {
                        'track_id': track_id,
                        'first_seen': time.time(),
                        'frames_tracked': 1,
                        'positions': [(center_x, center_y)]
                    }
                else:
                    self.tracked_objects[track_id]['frames_tracked'] += 1
                    self.tracked_objects[track_id]['positions'].append((center_x, center_y))
                    # Keep only recent positions
                    if len(self.tracked_objects[track_id]['positions']) > 10:
                        self.tracked_objects[track_id]['positions'] = self.tracked_objects[track_id]['positions'][-10:]
                
                # Update object properties
                self.tracked_objects[track_id].update({
                    'bbox': [x1, y1, x2, y2],
                    'center': (center_x, center_y),
                    'last_seen': time.time(),
                    'width': width,
                    'height': height
                })
                
                # Create tracked detection
                tracked_detection = {
                    'track_id': track_id,
                    'bbox': [x1, y1, x2, y2],
                    'center': (center_x, center_y),
                    'confidence': people_detections[min(i, len(people_detections) - 1)]['confidence'] if people_detections else 0.8,
                    'frames_tracked': self.tracked_objects[track_id]['frames_tracked']
                }
                
                tracked_detections.append(tracked_detection)
            
            return tracked_detections
            
        except Exception as e:
            self.logger.error(f"Error in object tracking: {str(e)}", exc_info=True)
            return []
    
    def check_ppe_compliance(self, people_detections: List[Dict], 
                           ppe_detections: List[Dict]) -> List[Dict]:
        """
        Check PPE compliance for detected people
        
        Args:
            people_detections: List of person detections
            ppe_detections: List of PPE detections
            
        Returns:
            List of people with PPE compliance status
        """
        if not self.ppe_detection_enabled or not ppe_detections:
            # Mark all people as non-compliant if PPE detection disabled or no PPE detected
            for person in people_detections:
                person['ppe_compliant'] = False
                person['missing_ppe'] = self.required_ppe_classes.copy()
                person['detected_ppe'] = []
            return people_detections
        
        # Check PPE compliance for each person
        for person in people_detections:
            person_bbox = person['bbox']
            detected_ppe = []
            missing_ppe = self.required_ppe_classes.copy()
            
            # Find PPE items near this person
            for ppe in ppe_detections:
                if self._is_ppe_associated_with_person(person_bbox, ppe['bbox']):
                    ppe_class = ppe['class_name']
                    detected_ppe.append(ppe_class)
                    if ppe_class in missing_ppe:
                        missing_ppe.remove(ppe_class)
            
            person['detected_ppe'] = detected_ppe
            person['missing_ppe'] = missing_ppe
            person['ppe_compliant'] = len(missing_ppe) == 0
        
        return people_detections
    
    def _is_ppe_associated_with_person(self, person_bbox: List[float], 
                                     ppe_bbox: List[float], 
                                     proximity_threshold: float = 50.0) -> bool:
        """
        Check if PPE item is associated with a person based on proximity
        
        Args:
            person_bbox: Person bounding box [x1, y1, x2, y2]
            ppe_bbox: PPE bounding box [x1, y1, x2, y2]
            proximity_threshold: Maximum distance for association
            
        Returns:
            True if PPE is associated with person
        """
        # Calculate centers
        person_center = ((person_bbox[0] + person_bbox[2]) / 2, (person_bbox[1] + person_bbox[3]) / 2)
        ppe_center = ((ppe_bbox[0] + ppe_bbox[2]) / 2, (ppe_bbox[1] + ppe_bbox[3]) / 2)
        
        # Calculate distance
        distance = np.sqrt((person_center[0] - ppe_center[0])**2 + (person_center[1] - ppe_center[1])**2)
        
        return distance <= proximity_threshold
    
    def check_zone_violations(self, tracked_people: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Check for zone violations (unauthorized access, occupancy limits, etc.)
        
        Args:
            tracked_people: List of tracked person detections
            
        Returns:
            Dictionary mapping zone names to violations
        """
        zone_violations = {}
        
        for zone_type, zone_list in self.zones.items():
            for zone in zone_list:
                zone_name = zone.get('name', f'Zone_{zone_type}')
                people_in_zone = []
                
                # Check each person against this zone
                for person in tracked_people:
                    if self._is_point_in_zone(person['center'], zone):
                        people_in_zone.append(person)
                
                # Check for violations
                violations = self._check_zone_specific_violations(zone, people_in_zone, zone_type)
                
                if violations:
                    zone_violations[zone_name] = violations
        
        return zone_violations
    
    def _is_point_in_zone(self, point: Tuple[float, float], zone: Dict) -> bool:
        """
        Check if a point is inside a zone using polygon containment
        
        Args:
            point: (x, y) coordinates
            zone: Zone definition with coordinates
            
        Returns:
            True if point is inside zone
        """
        try:
            coordinates = zone.get('coordinates', [])
            if len(coordinates) < 3:
                return False
            
            polygon = np.array(coordinates, dtype=np.int32)
            result = cv2.pointPolygonTest(polygon, point, False)
            return result >= 0
            
        except Exception as e:
            self.logger.error(f"Error checking point in zone: {e}")
            return False
    
    def _check_zone_specific_violations(self, zone: Dict, people_in_zone: List[Dict], 
                                      zone_type: str) -> List[Dict]:
        """
        Check for zone-specific violations
        
        Args:
            zone: Zone definition
            people_in_zone: People detected in this zone
            zone_type: Type of zone (entry_zone, server_zone, etc.)
            
        Returns:
            List of violation dictionaries
        """
        violations = []
        zone_config = DatacenterZoneTypes.get_zone_config(zone_type)
        
        # Check occupancy limits
        occupancy_limit = zone.get('occupancy_limit', zone_config.get('default_occupancy_limit', 10))
        if len(people_in_zone) > occupancy_limit:
            violations.append({
                'type': 'occupancy_exceeded',
                'severity': 'medium',
                'details': {
                    'current_count': len(people_in_zone),
                    'limit': occupancy_limit,
                    'people': people_in_zone
                }
            })
        
        # Check security level violations
        security_level = zone.get('security_level', zone_config.get('security_level', 'restricted'))
        if security_level in ['high_security', 'critical'] and people_in_zone:
            violations.append({
                'type': 'unauthorized_access',
                'severity': 'high' if security_level == 'high_security' else 'critical',
                'details': {
                    'security_level': security_level,
                    'people_count': len(people_in_zone),
                    'people': people_in_zone
                }
            })
        
        # Check PPE requirements for zones that require it
        required_ppe = zone.get('required_ppe', zone_config.get('required_ppe', []))
        if required_ppe:
            for person in people_in_zone:
                if not person.get('ppe_compliant', True):
                    violations.append({
                        'type': 'ppe_violation',
                        'severity': 'medium',
                        'details': {
                            'person': person,
                            'required_ppe': required_ppe,
                            'missing_ppe': person.get('missing_ppe', [])
                        }
                    })
        
        return violations
    
    def _check_camera_tamper(self, frame: np.ndarray) -> bool:
        """
        Check for camera tampering (obstruction, movement, etc.)
        
        Args:
            frame: Current video frame
            
        Returns:
            True if tampering detected
        """
        try:
            # Calculate frame hash for comparison
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_hash = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
            
            if self.last_frame_hash is not None:
                # Calculate correlation between frames
                correlation = cv2.compareHist(self.last_frame_hash, frame_hash, cv2.HISTCMP_CORREL)
                
                # Check for significant change (possible tampering)
                if correlation < DatacenterConfig.FRAME_DIFF_THRESHOLD:
                    self.tamper_check_count += 1
                    
                    # Require multiple consecutive detections to avoid false positives
                    if self.tamper_check_count >= 3:
                        self.tamper_check_count = 0
                        return True
                else:
                    self.tamper_check_count = 0
            
            self.last_frame_hash = frame_hash
            return False
            
        except Exception as e:
            self.logger.error(f"Error in camera tamper detection: {e}")
            return False
    
    def _handle_tamper_event(self, frame: np.ndarray, timestamp: float):
        """Handle camera tampering event"""
        try:
            event_data = {
                'event_type': 'camera_tamper',
                'severity': 'high',
                'timestamp': timestamp,
                'camera_id': self.camera_id,
                'datacenter_id': self.datacenter_id,
                'details': {
                    'detection_method': 'frame_correlation',
                    'confidence': 0.9
                }
            }
            
            # Log security event
            audit_logger.log_event_detection(
                event_type='camera_tamper',
                camera_id=str(self.camera_id),
                datacenter_id=str(self.datacenter_id),
                severity='high',
                detection_data=event_data['details']
            )
            
            # Send immediate alert
            if self.sms_enabled:
                self._send_security_alert(
                    event_type='Camera Tampering',
                    details=f"Camera {self.camera_id} tampering detected",
                    timestamp=timestamp
                )
            
            # Save event
            if self.db_writer:
                self._save_security_event(event_data, frame, timestamp)
            
            self.logger.warning(f"Camera tampering detected on camera {self.camera_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling tamper event: {e}")
    
    def _send_security_alert(self, event_type: str, details: str, timestamp: float):
        """Send SMS security alert"""
        try:
            if not self.sms_enabled:
                return False
            
            # Format timestamp
            ist_tz = pytz.timezone('Asia/Kolkata')
            dt = datetime.fromtimestamp(timestamp, tz=ist_tz)
            formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S IST')
            
            # Create alert message
            message = (
                f"🚨 DATACENTER SECURITY ALERT 🚨\n\n"
                f"Event: {event_type}\n"
                f"Datacenter: {self.datacenter_id}\n"
                f"Camera: {self.camera_id}\n"
                f"Details: {details}\n"
                f"Time: {formatted_time}\n\n"
                f"Immediate attention required!"
            )
            
            # Send to all configured numbers
            sent_count = 0
            for phone_number in DatacenterConfig.ALERT_PHONE_NUMBERS:
                try:
                    message_obj = self.twilio_client.messages.create(
                        body=message,
                        from_=DatacenterConfig.TWILIO_PHONE_NUMBER,
                        to=phone_number
                    )
                    sent_count += 1
                    self.logger.info(f"Security alert sent to {phone_number}, SID: {message_obj.sid}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to send alert to {phone_number}: {e}")
            
            return sent_count > 0
            
        except Exception as e:
            self.logger.error(f"Error sending security alert: {e}")
            return False
    
    def get_current_people_count(self) -> int:
        """Get current number of people detected"""
        return len(self.tracked_objects)
    
    def get_current_stats(self) -> Dict:
        """Get current processing statistics"""
        current_time = time.time()
        uptime = current_time - self.stats['start_time']
        
        return {
            'camera_id': self.camera_id,
            'datacenter_id': self.datacenter_id,
            'frames_processed': self.stats['frames_processed'],
            'events_detected': self.stats['events_detected'],
            'people_detected': self.stats['people_detected'],
            'tracked_objects': len(self.tracked_objects),
            'total_object_count': self.total_object_count,
            'uptime_seconds': uptime,
            'average_fps': self.stats['frames_processed'] / max(uptime, 1),
            'last_processed': self.stats['last_processed_time']
        }
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.logger.info(f"Cleaning up camera model for camera {self.camera_id}")
            
            # Clear buffers
            with self.buffer_lock:
                self.frame_buffer.clear()
                self.timestamp_buffer.clear()
            
            # Clear tracking data
            self.tracked_objects.clear()
            
            self.logger.info("Camera model cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

# Export the base class
__all__ = ['DatacenterCameraModelBase']