#!/usr/bin/env python3
"""
Script 15: perimeter_monitor.py
File Path: src/camera_models/perimeter_monitor.py

Datacenter Monitoring System - Perimeter Security Monitoring

This module implements Phase 1 perimeter use cases:
1. Intrusion Detection (a1 - scalable, low effort)
2. After-hours monitoring with enhanced sensitivity
3. Vehicle detection in restricted areas
4. Motion pattern analysis for suspicious activity
5. Perimeter breach detection

Perimeter monitoring focuses on:
- Outdoor boundary security
- Fence line monitoring  
- Vehicle approach detection
- After-hours enhanced security
- Weather-resistant detection
- Long-range motion analysis
"""

import cv2
import numpy as np
import time
import uuid
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, time as dt_time
import math

from .camera_model_base import DatacenterCameraModelBase
from config import DatacenterConfig, DatacenterEventTypes
from logger import audit_logger

class PerimeterMonitor(DatacenterCameraModelBase):
    """
    Monitor for datacenter perimeter security implementing:
    - Intrusion detection along fence lines and boundaries
    - Vehicle detection in restricted perimeter areas
    - After-hours enhanced monitoring
    - Motion pattern analysis
    - Long-range object detection and tracking
    """

    def __init__(self, camera_id: int, datacenter_id: int, zones: Optional[Dict] = None, 
                 rules: Optional[List] = None, settings: Optional[Dict] = None, 
                 db=None, db_writer=None, frames_base_dir: str = 'frames'):
        """
        Initialize the perimeter monitor
        
        Args:
            camera_id: Camera identifier
            datacenter_id: Datacenter identifier
            zones: Zone definitions (perimeter_zones, vehicle_zones)
            rules: Monitoring rules for perimeter events
            settings: Camera-specific settings
            db: Database instance
            db_writer: Database writer
            frames_base_dir: Base directory for frame storage
        """
        
        super().__init__(camera_id, datacenter_id, zones, rules, settings, 
                        db, db_writer, frames_base_dir)
        
        self.logger.info("Initializing PerimeterMonitor")
        
        # Perimeter-specific configuration
        self.intrusion_sensitivity = settings.get('intrusion_sensitivity', DatacenterConfig.INTRUSION_SENSITIVITY)
        self.motion_threshold = settings.get('motion_threshold', 0.3)
        self.min_object_size = settings.get('min_object_size', 50)  # pixels
        self.max_detection_distance = settings.get('max_detection_distance', 100)  # meters (estimated)
        
        # After-hours monitoring
        self.enhanced_after_hours = settings.get('enhanced_after_hours', True)
        self.business_start_hour = settings.get('business_start_hour', 8)  # 8 AM
        self.business_end_hour = settings.get('business_end_hour', 18)    # 6 PM
        
        # Vehicle detection parameters
        self.vehicle_detection_enabled = settings.get('vehicle_detection_enabled', True)
        self.vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
        self.vehicle_confidence_threshold = 0.6
        
        # Get zone configurations
        self.perimeter_zones = self._get_zones_by_type('perimeter_zone')
        self.vehicle_zones = self._get_zones_by_type('vehicle_zone')
        self.restricted_zones = self._get_zones_by_type('restricted_zone')
        
        # Motion detection for intrusion
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50, history=500
        )
        self.motion_detection_enabled = True
        
        # Tracking state
        self.perimeter_breaches = {}  # Track ongoing breaches
        self.vehicle_detections = {}  # Track vehicles in restricted areas
        self.motion_alerts = {}       # Track motion-based alerts
        
        # Pattern analysis
        self.suspicious_patterns = []
        self.movement_history = {}    # Track movement patterns
        self.loitering_threshold = 120  # 2 minutes for perimeter loitering
        
        # Environmental considerations
        self.weather_compensation = True
        self.wind_motion_threshold = 0.1  # Lower threshold to ignore wind
        self.rain_detection_enabled = True
        
        # Statistics
        self.perimeter_stats = {
            'intrusion_attempts': 0,
            'vehicle_violations': 0,
            'motion_alerts': 0,
            'after_hours_events': 0,
            'false_positives': 0,
            'patrol_detections': 0  # Authorized security patrols
        }
        
        self.logger.info(f"Perimeter monitor initialized - Sensitivity: {self.intrusion_sensitivity}")
        self.logger.info(f"Monitoring {len(self.perimeter_zones)} perimeter zones, {len(self.vehicle_zones)} vehicle zones")
        self.logger.info(f"After-hours monitoring: {self.enhanced_after_hours} (business hours: {self.business_start_hour}:00-{self.business_end_hour}:00)")
    
    def _get_zones_by_type(self, zone_type: str) -> List[Dict]:
        """Extract zones of specific type from zone configuration"""
        zones_of_type = []
        
        if zone_type in self.zones:
            zones_of_type = self.zones[zone_type]
        else:
            for zt, zone_list in self.zones.items():
                for zone in zone_list:
                    if zone.get('type') == zone_type or zone.get('zone_type') == zone_type:
                        zones_of_type.append(zone)
        
        return zones_of_type
    
    def _process_frame_impl(self, frame: np.ndarray, timestamp: float, 
                           detection_result: Any, ppe_result: Any = None) -> Tuple[np.ndarray, Dict]:
        """
        Process frame for perimeter monitoring
        
        Args:
            frame: Video frame
            timestamp: Frame timestamp
            detection_result: Person and vehicle detection results from YOLO
            ppe_result: Not used for perimeter monitoring
            
        Returns:
            Tuple of (annotated_frame, processing_results)
        """
        
        # Determine if this is after-hours
        is_after_hours = self._is_after_hours(timestamp)
        
        # Extract detections (people and vehicles)
        people_detections, vehicle_detections = self._extract_perimeter_detections(frame, detection_result)
        
        # Motion-based detection for enhanced sensitivity
        motion_detections = []
        if self.motion_detection_enabled:
            motion_detections = self._detect_motion_intrusions(frame, timestamp)
        
        # Update object tracking for people
        tracked_people = self.update_object_tracking(people_detections)
        
        # Track vehicles separately
        tracked_vehicles = self._track_vehicles(vehicle_detections, timestamp)
        
        # Store current counts
        self.current_people_count = len(tracked_people)
        
        # Analyze perimeter zones
        perimeter_analysis = self._analyze_perimeter_zones(tracked_people, tracked_vehicles, timestamp, is_after_hours)
        
        # Detect intrusions
        intrusion_events = self._detect_perimeter_intrusions(tracked_people, timestamp, is_after_hours)
        
        # Detect vehicle violations
        vehicle_events = self._detect_vehicle_violations(tracked_vehicles, timestamp)
        
        # Analyze motion patterns
        motion_events = self._analyze_motion_patterns(motion_detections, timestamp)
        
        # Check for suspicious behavior patterns
        pattern_events = self._detect_suspicious_patterns(tracked_people, timestamp)
        
        # Enhanced after-hours monitoring
        after_hours_events = []
        if is_after_hours and self.enhanced_after_hours:
            after_hours_events = self._enhanced_after_hours_monitoring(
                tracked_people, tracked_vehicles, motion_detections, timestamp
            )
        
        # Annotate frame with all detections
        annotated_frame = self._annotate_perimeter_frame(
            frame, tracked_people, tracked_vehicles, motion_detections, perimeter_analysis, is_after_hours
        )
        
        # Handle all detected events
        all_events = intrusion_events + vehicle_events + motion_events + pattern_events + after_hours_events
        for event in all_events:
            self._handle_perimeter_event(event, annotated_frame, timestamp)
        
        # Prepare processing results
        processing_results = {
            'people_count': len(tracked_people),
            'vehicle_count': len(tracked_vehicles),
            'motion_detections': len(motion_detections),
            'perimeter_analysis': perimeter_analysis,
            'events': all_events,
            'is_after_hours': is_after_hours,
            'perimeter_stats': self.perimeter_stats.copy()
        }
        
        # Update statistics
        self.stats['people_detected'] += len(tracked_people)
        if all_events:
            self.stats['events_detected'] += len(all_events)
            if is_after_hours:
                self.perimeter_stats['after_hours_events'] += len(all_events)
        
        return annotated_frame, processing_results
    
    def _is_after_hours(self, timestamp: float) -> bool:
        """
        Check if current time is after business hours
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            True if after hours
        """
        try:
            dt = datetime.fromtimestamp(timestamp)
            current_hour = dt.hour
            
            # Check if weekend
            is_weekend = dt.weekday() >= 5  # Saturday = 5, Sunday = 6
            
            # After hours if weekend or outside business hours
            if is_weekend:
                return True
            
            return current_hour < self.business_start_hour or current_hour >= self.business_end_hour
            
        except Exception as e:
            self.logger.error(f"Error checking after hours: {e}")
            return False
    
    def _extract_perimeter_detections(self, frame: np.ndarray, detection_result: Any) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract people and vehicle detections from YOLO results
        
        Args:
            frame: Video frame
            detection_result: YOLO detection results
            
        Returns:
            Tuple of (people_detections, vehicle_detections)
        """
        people_detections = []
        vehicle_detections = []
        
        if detection_result and hasattr(detection_result, 'boxes'):
            for i, box in enumerate(detection_result.boxes):
                try:
                    class_id = int(box.cls[0]) if box.cls.numel() > 0 else -1
                    class_name = detection_result.names.get(class_id, "unknown")
                    confidence = float(box.conf[0]) if box.conf.numel() > 0 else 0
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    
                    # Calculate object properties
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    # Filter by minimum size (remove very small detections)
                    if area < self.min_object_size:
                        continue
                    
                    detection = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'center': (center_x, center_y),
                        'width': width,
                        'height': height,
                        'area': area,
                        'aspect_ratio': width / height if height > 0 else 1.0
                    }
                    
                    # Categorize detection
                    if class_name == 'person':
                        people_detections.append(detection)
                    elif class_name in self.vehicle_classes and confidence >= self.vehicle_confidence_threshold:
                        detection['vehicle_type'] = class_name
                        vehicle_detections.append(detection)
                        
                except Exception as e:
                    self.logger.error(f"Error processing detection: {e}")
        
        return people_detections, vehicle_detections
    
    def _detect_motion_intrusions(self, frame: np.ndarray, timestamp: float) -> List[Dict]:
        """
        Detect motion-based intrusions using background subtraction
        
        Args:
            frame: Video frame
            timestamp: Current timestamp
            
        Returns:
            List of motion detection dictionaries
        """
        motion_detections = []
        
        try:
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame)
            
            # Remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (ignore small movements like leaves, etc.)
                min_motion_area = self.min_object_size * 2
                if area < min_motion_area:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w / 2
                center_y = y + h / 2
                
                # Check if motion is in perimeter zones
                motion_in_perimeter = False
                for zone in self.perimeter_zones:
                    if self._is_point_in_zone((center_x, center_y), zone):
                        motion_in_perimeter = True
                        break
                
                if motion_in_perimeter:
                    motion_detection = {
                        'type': 'motion',
                        'bbox': [x, y, x + w, y + h],
                        'center': (center_x, center_y),
                        'area': area,
                        'timestamp': timestamp,
                        'confidence': min(area / 1000.0, 1.0)  # Confidence based on size
                    }
                    motion_detections.append(motion_detection)
        
        except Exception as e:
            self.logger.error(f"Error in motion detection: {e}")
        
        return motion_detections
    
    def _track_vehicles(self, vehicle_detections: List[Dict], timestamp: float) -> List[Dict]:
        """
        Track vehicles separately from people (vehicles have different movement patterns)
        
        Args:
            vehicle_detections: List of vehicle detections
            timestamp: Current timestamp
            
        Returns:
            List of tracked vehicles
        """
        tracked_vehicles = []
        
        for vehicle in vehicle_detections:
            # Simple tracking based on position proximity
            # In production, use a separate tracker for vehicles
            vehicle_id = self._assign_vehicle_id(vehicle, timestamp)
            
            tracked_vehicle = vehicle.copy()
            tracked_vehicle['vehicle_id'] = vehicle_id
            tracked_vehicle['timestamp'] = timestamp
            
            # Store in vehicle tracking
            self.vehicle_detections[vehicle_id] = {
                'last_seen': timestamp,
                'vehicle_type': vehicle.get('vehicle_type', 'unknown'),
                'positions': [vehicle['center']],
                'first_detected': timestamp
            }
            
            tracked_vehicles.append(tracked_vehicle)
        
        return tracked_vehicles
    
    def _assign_vehicle_id(self, vehicle: Dict, timestamp: float) -> str:
        """Assign unique ID to vehicle based on position proximity"""
        vehicle_center = vehicle['center']
        proximity_threshold = 100  # pixels
        
        # Find existing vehicle within proximity
        for vehicle_id, track_data in self.vehicle_detections.items():
            if timestamp - track_data['last_seen'] > 10:  # Remove old tracks
                continue
                
            last_position = track_data['positions'][-1] if track_data['positions'] else (0, 0)
            distance = math.sqrt((vehicle_center[0] - last_position[0])**2 + 
                               (vehicle_center[1] - last_position[1])**2)
            
            if distance < proximity_threshold:
                # Update existing track
                track_data['positions'].append(vehicle_center)
                track_data['last_seen'] = timestamp
                return vehicle_id
        
        # Create new vehicle ID
        new_id = f"vehicle_{len(self.vehicle_detections)}_{int(timestamp)}"
        return new_id
    
    def _analyze_perimeter_zones(self, tracked_people: List[Dict], tracked_vehicles: List[Dict], 
                                timestamp: float, is_after_hours: bool) -> Dict:
        """
        Analyze activity in perimeter zones
        
        Args:
            tracked_people: List of tracked people
            tracked_vehicles: List of tracked vehicles
            timestamp: Current timestamp
            is_after_hours: Whether it's after business hours
            
        Returns:
            Perimeter analysis dictionary
        """
        analysis = {
            'zone_activity': {},
            'security_status': 'normal',
            'threats_detected': [],
            'patrol_activity': False
        }
        
        for zone in self.perimeter_zones:
            zone_name = zone.get('name', 'Perimeter Zone')
            people_in_zone = []
            vehicles_in_zone = []
            
            # Count people in zone
            for person in tracked_people:
                if self._is_point_in_zone(person['center'], zone):
                    people_in_zone.append(person)
            
            # Count vehicles in zone
            for vehicle in tracked_vehicles:
                if self._is_point_in_zone(vehicle['center'], zone):
                    vehicles_in_zone.append(vehicle)
            
            zone_activity = {
                'people_count': len(people_in_zone),
                'vehicle_count': len(vehicles_in_zone),
                'activity_level': self._calculate_activity_level(people_in_zone, vehicles_in_zone, is_after_hours),
                'threat_level': 'low'
            }
            
            # Assess threat level
            if is_after_hours and (people_in_zone or vehicles_in_zone):
                zone_activity['threat_level'] = 'high'
                analysis['threats_detected'].append({
                    'zone': zone_name,
                    'type': 'after_hours_activity',
                    'people': len(people_in_zone),
                    'vehicles': len(vehicles_in_zone)
                })
            elif len(people_in_zone) > 3 or len(vehicles_in_zone) > 1:
                zone_activity['threat_level'] = 'medium'
            
            analysis['zone_activity'][zone_name] = zone_activity
        
        # Determine overall security status
        threat_levels = [activity['threat_level'] for activity in analysis['zone_activity'].values()]
        if 'high' in threat_levels:
            analysis['security_status'] = 'alert'
        elif 'medium' in threat_levels:
            analysis['security_status'] = 'caution'
        
        return analysis
    
    def _calculate_activity_level(self, people: List[Dict], vehicles: List[Dict], is_after_hours: bool) -> str:
        """Calculate activity level for a zone"""
        total_objects = len(people) + len(vehicles)
        
        if is_after_hours:
            return 'high' if total_objects > 0 else 'none'
        else:
            if total_objects == 0:
                return 'none'
            elif total_objects <= 2:
                return 'low'
            elif total_objects <= 5:
                return 'medium'
            else:
                return 'high'
    
    def _detect_perimeter_intrusions(self, tracked_people: List[Dict], timestamp: float, 
                                   is_after_hours: bool) -> List[Dict]:
        """
        Detect perimeter intrusions
        
        Args:
            tracked_people: List of tracked people
            timestamp: Current timestamp
            is_after_hours: Whether it's after business hours
            
        Returns:
            List of intrusion events
        """
        intrusion_events = []
        
        for zone in self.perimeter_zones:
            zone_name = zone.get('name', 'Perimeter Zone')
            security_level = zone.get('security_level', 'restricted')
            
            for person in tracked_people:
                if self._is_point_in_zone(person['center'], zone):
                    
                    # Check if this person has been tracked long enough to be considered a real intrusion
                    track_id = person['track_id']
                    frames_tracked = person.get('frames_tracked', 0)
                    
                    if frames_tracked >= self.tracking_threshold:
                        
                        # Enhanced sensitivity after hours
                        is_intrusion = False
                        severity = 'medium'
                        
                        if is_after_hours:
                            is_intrusion = True
                            severity = 'high'
                        elif security_level in ['high_security', 'critical']:
                            is_intrusion = True
                            severity = 'high' if security_level == 'critical' else 'medium'
                        elif not self._is_authorized_patrol(person, timestamp):
                            is_intrusion = True
                            severity = 'medium'
                        
                        if is_intrusion:
                            event_key = f"perimeter_intrusion_{zone_name}_{track_id}"
                            
                            if self._should_trigger_event(event_key, timestamp):
                                
                                intrusion_event = {
                                    'type': DatacenterEventTypes.INTRUSION,
                                    'severity': severity,
                                    'zone_name': zone_name,
                                    'zone_type': 'perimeter_zone',
                                    'security_level': security_level,
                                    'person': person,
                                    'timestamp': timestamp,
                                    'is_after_hours': is_after_hours,
                                    'details': {
                                        'detection_method': 'perimeter_tracking',
                                        'frames_tracked': frames_tracked,
                                        'confidence': person.get('confidence', 0.8),
                                        'patrol_check': False
                                    }
                                }
                                
                                intrusion_events.append(intrusion_event)
                                self.perimeter_stats['intrusion_attempts'] += 1
                                
                                self.logger.warning(f"Perimeter intrusion detected in {zone_name}: Person {track_id}")
        
        return intrusion_events
    
    def _detect_vehicle_violations(self, tracked_vehicles: List[Dict], timestamp: float) -> List[Dict]:
        """
        Detect vehicle violations in restricted areas
        
        Args:
            tracked_vehicles: List of tracked vehicles
            timestamp: Current timestamp
            
        Returns:
            List of vehicle violation events
        """
        vehicle_events = []
        
        # Check vehicles in restricted zones
        for zone in self.restricted_zones + self.perimeter_zones:
            zone_name = zone.get('name', 'Restricted Zone')
            vehicle_allowed = zone.get('vehicle_access', False)
            
            if not vehicle_allowed:
                for vehicle in tracked_vehicles:
                    if self._is_point_in_zone(vehicle['center'], zone):
                        
                        vehicle_id = vehicle['vehicle_id']
                        event_key = f"vehicle_violation_{zone_name}_{vehicle_id}"
                        
                        if self._should_trigger_event(event_key, timestamp):
                            
                            vehicle_event = {
                                'type': 'vehicle_violation',
                                'severity': 'medium',
                                'zone_name': zone_name,
                                'zone_type': zone.get('zone_type', 'restricted_zone'),
                                'vehicle': vehicle,
                                'timestamp': timestamp,
                                'details': {
                                    'vehicle_type': vehicle.get('vehicle_type', 'unknown'),
                                    'vehicle_id': vehicle_id,
                                    'zone_restriction': 'no_vehicles_allowed'
                                }
                            }
                            
                            vehicle_events.append(vehicle_event)
                            self.perimeter_stats['vehicle_violations'] += 1
                            
                            self.logger.warning(f"Vehicle violation in {zone_name}: {vehicle.get('vehicle_type', 'vehicle')} {vehicle_id}")
        
        return vehicle_events
    
    def _analyze_motion_patterns(self, motion_detections: List[Dict], timestamp: float) -> List[Dict]:
        """
        Analyze motion patterns for suspicious activity
        
        Args:
            motion_detections: List of motion detections
            timestamp: Current timestamp
            
        Returns:
            List of suspicious motion events
        """
        motion_events = []
        
        # Check for significant motion in perimeter
        if len(motion_detections) > 3:  # Multiple motion areas
            
            event_key = f"motion_pattern_{int(timestamp)}"
            if self._should_trigger_event(event_key, timestamp, cooldown=30):  # 30s cooldown
                
                motion_event = {
                    'type': 'suspicious_motion',
                    'severity': 'low',
                    'zone_name': 'Perimeter',
                    'zone_type': 'perimeter_zone',
                    'timestamp': timestamp,
                    'details': {
                        'motion_areas': len(motion_detections),
                        'detection_method': 'background_subtraction',
                        'pattern_type': 'multiple_motion_areas'
                    }
                }
                
                motion_events.append(motion_event)
                self.perimeter_stats['motion_alerts'] += 1
        
        return motion_events
    
    def _detect_suspicious_patterns(self, tracked_people: List[Dict], timestamp: float) -> List[Dict]:
        """
        Detect suspicious behavior patterns in perimeter
        
        Args:
            tracked_people: List of tracked people
            timestamp: Current timestamp
            
        Returns:
            List of suspicious pattern events
        """
        pattern_events = []
        
        for person in tracked_people:
            track_id = person['track_id']
            
            # Check for loitering (stationary behavior)
            if track_id in self.tracked_objects:
                track_data = self.tracked_objects[track_id]
                positions = track_data.get('positions', [])
                
                if len(positions) >= 10:  # Need sufficient position history
                    # Calculate movement distance
                    total_distance = 0
                    for i in range(1, len(positions)):
                        dx = positions[i][0] - positions[i-1][0]
                        dy = positions[i][1] - positions[i-1][1]
                        total_distance += math.sqrt(dx*dx + dy*dy)
                    
                    avg_movement = total_distance / len(positions)
                    
                    # Check for loitering (very little movement)
                    if avg_movement < 5.0:  # Less than 5 pixels average movement
                        time_tracked = timestamp - track_data.get('first_seen', timestamp)
                        
                        if time_tracked > self.loitering_threshold:
                            event_key = f"perimeter_loitering_{track_id}"
                            
                            if self._should_trigger_event(event_key, timestamp):
                                
                                loitering_event = {
                                    'type': DatacenterEventTypes.LOITERING,
                                    'severity': 'medium',
                                    'zone_name': 'Perimeter',
                                    'zone_type': 'perimeter_zone',
                                    'person': person,
                                    'timestamp': timestamp,
                                    'details': {
                                        'loitering_duration': time_tracked,
                                        'average_movement': avg_movement,
                                        'threshold': self.loitering_threshold
                                    }
                                }
                                
                                pattern_events.append(loitering_event)
                                
                                self.logger.warning(f"Perimeter loitering detected: Person {track_id} for {time_tracked:.1f}s")
        
        return pattern_events
    
    def _enhanced_after_hours_monitoring(self, tracked_people: List[Dict], tracked_vehicles: List[Dict], 
                                        motion_detections: List[Dict], timestamp: float) -> List[Dict]:
        """
        Enhanced monitoring during after-hours with higher sensitivity
        
        Args:
            tracked_people: List of tracked people
            tracked_vehicles: List of tracked vehicles  
            motion_detections: List of motion detections
            timestamp: Current timestamp
            
        Returns:
            List of after-hours events
        """
        after_hours_events = []
        
        # Any human activity after hours is suspicious
        if tracked_people:
            for person in tracked_people:
                # Check if this might be authorized patrol
                if not self._is_authorized_patrol(person, timestamp):
                    
                    event_key = f"after_hours_person_{person['track_id']}"
                    if self._should_trigger_event(event_key, timestamp):
                        
                        after_hours_event = {
                            'type': 'after_hours_activity',
                            'severity': 'high',
                            'zone_name': 'Perimeter',
                            'zone_type': 'perimeter_zone',
                            'person': person,
                            'timestamp': timestamp,
                            'details': {
                                'activity_type': 'unauthorized_presence',
                                'detection_confidence': person.get('confidence', 0.8)
                            }
                        }
                        
                        after_hours_events.append(after_hours_event)
        
        # Vehicle activity after hours
        if tracked_vehicles:
            event_key = f"after_hours_vehicles_{int(timestamp)}"
            if self._should_trigger_event(event_key, timestamp, cooldown=60):
                
                vehicle_event = {
                    'type': 'after_hours_activity',
                    'severity': 'medium',
                    'zone_name': 'Perimeter',
                    'zone_type': 'perimeter_zone',
                    'timestamp': timestamp,
                    'details': {
                        'activity_type': 'vehicle_presence',
                        'vehicle_count': len(tracked_vehicles),
                        'vehicles': [v.get('vehicle_type', 'unknown') for v in tracked_vehicles]
                    }
                }
                
                after_hours_events.append(vehicle_event)
        
        return after_hours_events
    
    def _is_authorized_patrol(self, person: Dict, timestamp: float) -> bool:
        """
        Check if person might be authorized security patrol
        
        Args:
            person: Person detection dictionary
            timestamp: Current timestamp
            
        Returns:
            True if likely authorized patrol
        """
        
        # Simple heuristics for patrol detection
        # In production, integrate with security patrol scheduling system
        
        track_id = person['track_id']
        
        if track_id in self.tracked_objects:
            track_data = self.tracked_objects[track_id]
            positions = track_data.get('positions', [])
            
            # Check for patrol-like movement pattern (regular, purposeful movement)
            if len(positions) >= 5:
                # Calculate movement consistency
                distances = []
                for i in range(1, len(positions)):
                    dx = positions[i][0] - positions[i-1][0]
                    dy = positions[i][1] - positions[i-1][1]
                    distance = math.sqrt(dx*dx + dy*dy)
                    distances.append(distance)
                
                if distances:
                    avg_distance = sum(distances) / len(distances)
                    # Consistent movement suggests patrol
                    if 10 < avg_distance < 50:  # Regular walking pace
                        self.perimeter_stats['patrol_detections'] += 1
                        return True
        
        return False
    
    def _should_trigger_event(self, event_key: str, timestamp: float, cooldown: Optional[float] = None) -> bool:
        """
        Check if enough time has passed since last event to trigger a new one
        
        Args:
            event_key: Unique key for this event type
            timestamp: Current timestamp
            cooldown: Optional custom cooldown period
            
        Returns:
            True if event should be triggered
        """
        cooldown_period = cooldown or self.event_cooldown
        
        if event_key in self.recent_events:
            time_since_last = timestamp - self.recent_events[event_key]
            if time_since_last < cooldown_period:
                return False
        
        self.recent_events[event_key] = timestamp
        return True
    
    def _handle_perimeter_event(self, event: Dict, frame: np.ndarray, timestamp: float):
        """
        Handle detected perimeter events (save, alert, log)
        
        Args:
            event: Event dictionary
            frame: Current video frame
            timestamp: Event timestamp
        """
        try:
            event_type = event['type']
            severity = event['severity']
            zone_name = event.get('zone_name', 'Perimeter')
            
            # Log security event for audit
            audit_logger.log_event_detection(
                event_type=event_type,
                camera_id=str(self.camera_id),
                datacenter_id=str(self.datacenter_id),
                severity=severity,
                detection_data=event.get('details', {})
            )
            
            # Generate unique event ID
            event_id = str(uuid.uuid4())
            
            # Save event media for high/critical severity
            if severity in ['high', 'critical']:
                if self.db_writer:
                    self._save_event_media(event_type, event, frame, timestamp, zone_name, event_id)
                
                # Send immediate alert for critical perimeter events
                if severity == 'critical' and self.sms_enabled:
                    self._send_security_alert(
                        event_type=event_type.replace('_', ' ').title(),
                        details=f"Perimeter breach in {zone_name}",
                        timestamp=timestamp
                    )
            
            # Always save motion and vehicle events as low priority
            elif event_type in ['suspicious_motion', 'vehicle_violation'] and self.db_writer:
                self._save_event_media(event_type, event, frame, timestamp, zone_name, event_id)
            
            self.logger.info(f"Perimeter event handled: {event_type} in {zone_name} (severity: {severity})")
            
        except Exception as e:
            self.logger.error(f"Error handling perimeter event: {e}", exc_info=True)
    
    def _save_event_media(self, event_type: str, event_data: Dict, frame: np.ndarray, 
                         timestamp: float, zone_name: str, event_id: str):
        """Save event media and queue for database storage"""
        try:
            if self.media_preference == "image":
                # Save annotated frame
                frame_path = self._save_frame_with_detections(event_id, frame, timestamp)
                if frame_path:
                    self.stats['frames_saved'] += 1
            else:
                # Queue video recording for critical events
                if event_data.get('severity') in ['high', 'critical']:
                    self._trigger_video_recording(event_id, event_data, timestamp)
                    self.stats['videos_saved'] += 1
            
            # Queue event for database storage
            if self.db_writer:
                db_event_data = {
                    'event_id': event_id,
                    'camera_id': self.camera_id,
                    'datacenter_id': self.datacenter_id,
                    'event_type': event_type,
                    'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
                    'zone_name': zone_name,
                    'metadata': event_data
                }
                
                self.db_writer.queue_event(db_event_data)
                
        except Exception as e:
            self.logger.error(f"Error saving perimeter event media: {e}")
    
    def _annotate_perimeter_frame(self, frame: np.ndarray, tracked_people: List[Dict], 
                                 tracked_vehicles: List[Dict], motion_detections: List[Dict],
                                 perimeter_analysis: Dict, is_after_hours: bool) -> np.ndarray:
        """
        Annotate frame with perimeter detections and analysis
        
        Args:
            frame: Original video frame
            tracked_people: List of tracked people
            tracked_vehicles: List of tracked vehicles
            motion_detections: List of motion detections
            perimeter_analysis: Perimeter analysis results
            is_after_hours: Whether it's after hours
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw perimeter zones
        for zone in self.perimeter_zones:
            zone_name = zone.get('name', 'Perimeter Zone')
            zone_activity = perimeter_analysis['zone_activity'].get(zone_name, {})
            threat_level = zone_activity.get('threat_level', 'low')
            
            # Color based on threat level
            if threat_level == 'high':
                color = (0, 0, 255)  # Red
            elif threat_level == 'medium':
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 0)  # Green
                
            self._draw_zone(annotated_frame, zone, color, alpha=0.2, label=f"{zone_name} ({threat_level})")
        
        # Draw vehicle zones
        for zone in self.vehicle_zones:
            self._draw_zone(annotated_frame, zone, (255, 0, 255), alpha=0.1, label=zone.get('name', 'Vehicle Zone'))
        
        # Draw people detections
        for person in tracked_people:
            bbox = person['bbox']
            track_id = person['track_id']
            confidence = person.get('confidence', 0.0)
            
            # Color based on authorization and time
            if is_after_hours:
                color = (0, 0, 255)  # Red for after hours
                label = f"ID:{track_id} (AFTER HOURS)"
            elif self._is_authorized_patrol(person, time.time()):
                color = (0, 255, 255)  # Cyan for patrol
                label = f"ID:{track_id} (PATROL)"
            else:
                color = (255, 255, 0)  # Yellow for regular detection
                label = f"ID:{track_id}"
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
            
            # Draw label with confidence
            label_with_conf = f"{label} ({confidence:.2f})"
            cv2.putText(annotated_frame, label_with_conf,
                       (int(bbox[0]), int(bbox[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw vehicle detections
        for vehicle in tracked_vehicles:
            bbox = vehicle['bbox']
            vehicle_id = vehicle.get('vehicle_id', 'V?')
            vehicle_type = vehicle.get('vehicle_type', 'vehicle')
            
            color = (255, 0, 0)  # Blue for vehicles
            label = f"{vehicle_type.upper()} {vehicle_id}"
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 3)
            
            # Draw label
            cv2.putText(annotated_frame, label,
                       (int(bbox[0]), int(bbox[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw motion detections
        for motion in motion_detections:
            bbox = motion['bbox']
            color = (128, 128, 128)  # Gray for motion
            
            cv2.rectangle(annotated_frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 1)
            
            cv2.putText(annotated_frame, "MOTION",
                       (int(bbox[0]), int(bbox[1]) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw status information
        status_y = 30
        
        # Time and mode indicator
        time_mode = "AFTER HOURS" if is_after_hours else "BUSINESS HOURS"
        time_color = (0, 0, 255) if is_after_hours else (0, 255, 0)
        cv2.putText(annotated_frame, f"MODE: {time_mode}",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, time_color, 2)
        status_y += 30
        
        # Security status
        security_status = perimeter_analysis['security_status'].upper()
        security_color = (0, 0, 255) if security_status == 'ALERT' else (255, 165, 0) if security_status == 'CAUTION' else (0, 255, 0)
        cv2.putText(annotated_frame, f"SECURITY: {security_status}",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, security_color, 2)
        status_y += 30
        
        # Object counts
        cv2.putText(annotated_frame, f"PEOPLE: {len(tracked_people)} | VEHICLES: {len(tracked_vehicles)} | MOTION: {len(motion_detections)}",
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Statistics at bottom
        stats_text = f"Intrusions: {self.perimeter_stats['intrusion_attempts']} | Vehicles: {self.perimeter_stats['vehicle_violations']} | After-hrs: {self.perimeter_stats['after_hours_events']}"
        cv2.putText(annotated_frame, stats_text,
                   (10, annotated_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return annotated_frame
    
    def _draw_zone(self, frame: np.ndarray, zone: Dict, color: Tuple[int, int, int], 
                  alpha: float = 0.3, label: Optional[str] = None):
        """Draw zone overlay on frame"""
        try:
            coordinates = zone.get('coordinates', [])
            if len(coordinates) < 3:
                return
            
            # Create overlay
            overlay = frame.copy()
            polygon = np.array(coordinates, dtype=np.int32)
            cv2.fillPoly(overlay, [polygon], color)
            
            # Apply transparency
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Draw border
            cv2.polylines(frame, [polygon], True, color, 2)
            
            # Draw label
            if label and len(coordinates) > 0:
                label_pos = (int(coordinates[0][0]), int(coordinates[0][1]))
                cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
        except Exception as e:
            self.logger.error(f"Error drawing perimeter zone: {e}")
    
    def get_perimeter_status(self) -> Dict:
        """Get current perimeter security status"""
        return {
            'camera_id': self.camera_id,
            'datacenter_id': self.datacenter_id,
            'security_mode': 'after_hours' if self._is_after_hours(time.time()) else 'business_hours',
            'active_tracks': len(self.tracked_objects),
            'vehicle_tracks': len(self.vehicle_detections),
            'perimeter_stats': self.perimeter_stats.copy(),
            'zones_monitored': len(self.perimeter_zones),
            'last_intrusion': max([t for t in self.recent_events.values()] + [0]),
            'motion_detection_active': self.motion_detection_enabled
        }

# Export the monitor class
__all__ = ['PerimeterMonitor']