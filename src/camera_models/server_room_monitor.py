#!/usr/bin/env python3
"""
Script 13: server_room_monitor.py
File Path: src/camera_models/server_room_monitor.py

Datacenter Monitoring System - Server Room Monitoring

This module implements Phase 1 use cases for server rooms:
1. PPE Detection (a1 - scalable, low effort) - Hard hat, safety vest, safety glasses
2. Intrusion Detection (a1 - scalable, low effort) - Unauthorized server room access
3. People Counting (a2 - scalable, medium effort) - Server room occupancy limits
4. Equipment Proximity Monitoring - Prevent unauthorized server rack access

Server room monitoring focuses on:
- PPE compliance validation (hard hat, safety vest, safety glasses)
- Authorized personnel verification
- Occupancy limit enforcement
- Equipment protection zones
- Air-gapped zone monitoring
- Emergency egress path monitoring
"""

import cv2
import numpy as np
import time
import uuid
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import os

from .camera_model_base import DatacenterCameraModelBase
from config import DatacenterConfig, DatacenterEventTypes
from logger import audit_logger

class ServerRoomMonitor(DatacenterCameraModelBase):
    """
    Monitor for datacenter server rooms implementing Phase 1 use cases:
    - PPE compliance detection (hard hat, safety vest, safety glasses)
    - Server room intrusion detection
    - Occupancy monitoring and limits
    - Equipment proximity protection
    - Air-gapped zone security
    """

    def __init__(self, camera_id: int, datacenter_id: int, zones: Optional[Dict] = None, 
                 rules: Optional[List] = None, settings: Optional[Dict] = None, 
                 db=None, db_writer=None, frames_base_dir: str = 'frames'):
        """
        Initialize the server room monitor
        
        Args:
            camera_id: Camera identifier
            datacenter_id: Datacenter identifier
            zones: Zone definitions (server_zones, equipment_zones, air_gap_zones)
            rules: Monitoring rules for server room events
            settings: Camera-specific settings
            db: Database instance
            db_writer: Database writer
            frames_base_dir: Base directory for frame storage
        """
        
        super().__init__(camera_id, datacenter_id, zones, rules, settings, 
                        db, db_writer, frames_base_dir)
        
        self.logger.info("Initializing ServerRoomMonitor")
        
        # PPE Detection Configuration
        self.ppe_detection_enabled = DatacenterConfig.PPE_DETECTION_ENABLED
        self.required_ppe_classes = settings.get('required_ppe', ['hard_hat', 'safety_vest', 'safety_glasses'])
        self.ppe_confidence_threshold = settings.get('ppe_confidence', DatacenterConfig.PPE_CONFIDENCE_THRESHOLD)
        self.ppe_grace_period = settings.get('ppe_grace_period', 30)  # seconds to put on PPE
        
        # Server Room Specific Settings
        self.max_occupancy = settings.get('max_occupancy', DatacenterConfig.get_occupancy_limit('server_room'))
        self.equipment_proximity_threshold = settings.get('equipment_proximity', 1.0)  # meters
        self.air_gap_violation_threshold = settings.get('air_gap_threshold', 0.5)  # meters
        
        # Get zone configurations
        self.server_zones = self._get_zones_by_type('server_zone')
        self.equipment_zones = self._get_zones_by_type('equipment_zone')
        self.air_gap_zones = self._get_zones_by_type('air_gap_zone')
        self.restricted_zones = self._get_zones_by_type('restricted_zone')
        
        # PPE Tracking State
        self.person_ppe_status = {}  # Track PPE status per person
        self.ppe_violation_timers = {}  # Track how long someone has been non-compliant
        self.entry_timestamps = {}  # Track when people entered server room
        
        # Equipment Protection
        self.equipment_access_log = {}  # Track who accessed which equipment
        self.critical_equipment_zones = []  # Define critical server racks
        
        # Occupancy Tracking
        self.current_occupancy = 0
        self.max_occupancy_violations = 0
        
        # Statistics for server room monitoring
        self.server_stats = {
            'ppe_violations': 0,
            'equipment_violations': 0,
            'occupancy_violations': 0,
            'air_gap_violations': 0,
            'authorized_entries': 0,
            'total_server_time': 0  # Total person-minutes in server room
        }
        
        # PPE Detection Color Coding
        self.ppe_colors = {
            'hard_hat': (0, 255, 255),      # Cyan
            'safety_vest': (0, 165, 255),   # Orange
            'safety_glasses': (255, 0, 255), # Magenta
            'compliant': (0, 255, 0),       # Green
            'non_compliant': (0, 0, 255)    # Red
        }
        
        self.logger.info(f"Server room monitor initialized - Required PPE: {self.required_ppe_classes}")
        self.logger.info(f"Max occupancy: {self.max_occupancy}, Equipment zones: {len(self.equipment_zones)}")
    
    def _get_zones_by_type(self, zone_type: str) -> List[Dict]:
        """Extract zones of specific type from zone configuration"""
        zones_of_type = []
        
        # Handle both database format and config file format
        if zone_type in self.zones:
            zones_of_type = self.zones[zone_type]
        else:
            # Search through all zone types
            for zt, zone_list in self.zones.items():
                for zone in zone_list:
                    if zone.get('type') == zone_type or zone.get('zone_type') == zone_type:
                        zones_of_type.append(zone)
        
        return zones_of_type
    
    def _process_frame_impl(self, frame: np.ndarray, timestamp: float, 
                           detection_result: Any, ppe_result: Any = None) -> Tuple[np.ndarray, Dict]:
        """
        Process frame for server room monitoring
        
        Args:
            frame: Video frame
            timestamp: Frame timestamp
            detection_result: Person detection results from YOLO
            ppe_result: PPE detection results from PPE model
            
        Returns:
            Tuple of (annotated_frame, processing_results)
        """
        
        # Extract people and PPE detections
        people_detections, ppe_detections = self.detect_people_and_ppe(frame, detection_result, ppe_result)
        
        # Update object tracking
        tracked_people = self.update_object_tracking(people_detections)
        
        # Check PPE compliance for each person
        ppe_compliance_results = self._check_comprehensive_ppe_compliance(tracked_people, ppe_detections, timestamp)
        
        # Update current people count
        self.current_people_count = len(tracked_people)
        self.current_occupancy = len(tracked_people)
        
        # Check server room violations
        occupancy_events = self._check_occupancy_violations(tracked_people, timestamp)
        equipment_events = self._check_equipment_violations(tracked_people, timestamp)
        air_gap_events = self._check_air_gap_violations(tracked_people, timestamp)
        intrusion_events = self._check_server_intrusion(tracked_people, timestamp)
        ppe_events = self._process_ppe_violations(ppe_compliance_results, timestamp)
        
        # Update server room statistics
        self._update_server_statistics(tracked_people, timestamp)
        
        # Annotate frame with all detections
        annotated_frame = self._annotate_server_room_frame(
            frame, tracked_people, ppe_detections, ppe_compliance_results
        )
        
        # Handle all detected events
        all_events = occupancy_events + equipment_events + air_gap_events + intrusion_events + ppe_events
        for event in all_events:
            self._handle_server_room_event(event, annotated_frame, timestamp)
        
        # Prepare processing results
        processing_results = {
            'people_count': len(tracked_people),
            'occupancy_status': {
                'current': self.current_occupancy,
                'max_allowed': self.max_occupancy,
                'violation': self.current_occupancy > self.max_occupancy
            },
            'ppe_compliance': ppe_compliance_results,
            'events': all_events,
            'server_stats': self.server_stats.copy()
        }
        
        # Update base statistics
        self.stats['people_detected'] += len(tracked_people)
        if all_events:
            self.stats['events_detected'] += len(all_events)
        
        return annotated_frame, processing_results
    
    def _check_comprehensive_ppe_compliance(self, tracked_people: List[Dict], 
                                          ppe_detections: List[Dict], 
                                          timestamp: float) -> Dict[int, Dict]:
        """
        Comprehensive PPE compliance check for all people in server room
        
        Args:
            tracked_people: List of tracked person detections
            ppe_detections: List of PPE item detections
            timestamp: Current timestamp
            
        Returns:
            Dictionary mapping track_id to PPE compliance status
        """
        compliance_results = {}
        
        for person in tracked_people:
            track_id = person['track_id']
            person_bbox = person['bbox']
            person_center = person['center']
            
            # Initialize PPE status for this person
            ppe_status = {
                'hard_hat': {'detected': False, 'confidence': 0.0},
                'safety_vest': {'detected': False, 'confidence': 0.0},
                'safety_glasses': {'detected': False, 'confidence': 0.0}
            }
            
            # Find PPE items associated with this person
            for ppe_item in ppe_detections:
                if self._is_ppe_associated_with_person(person_bbox, ppe_item['bbox']):
                    ppe_class = ppe_item['class_name']
                    if ppe_class in ppe_status:
                        ppe_status[ppe_class]['detected'] = True
                        ppe_status[ppe_class]['confidence'] = max(
                            ppe_status[ppe_class]['confidence'], 
                            ppe_item['confidence']
                        )
            
            # Determine overall compliance
            missing_ppe = []
            detected_ppe = []
            
            for ppe_class in self.required_ppe_classes:
                if ppe_class in ppe_status:
                    if ppe_status[ppe_class]['detected']:
                        detected_ppe.append(ppe_class)
                    else:
                        missing_ppe.append(ppe_class)
            
            is_compliant = len(missing_ppe) == 0
            
            # Track compliance time for grace period
            if track_id not in self.person_ppe_status:
                self.person_ppe_status[track_id] = {
                    'entry_time': timestamp,
                    'first_violation_time': None,
                    'compliant': is_compliant
                }
            
            # Update violation timing
            if not is_compliant:
                if self.person_ppe_status[track_id]['first_violation_time'] is None:
                    self.person_ppe_status[track_id]['first_violation_time'] = timestamp
            else:
                self.person_ppe_status[track_id]['first_violation_time'] = None
            
            self.person_ppe_status[track_id]['compliant'] = is_compliant
            
            # Calculate time in violation
            violation_duration = 0
            if self.person_ppe_status[track_id]['first_violation_time']:
                violation_duration = timestamp - self.person_ppe_status[track_id]['first_violation_time']
            
            # Determine if violation should trigger alert (after grace period)
            should_alert = (not is_compliant and 
                          violation_duration > self.ppe_grace_period)
            
            compliance_results[track_id] = {
                'person': person,
                'ppe_status': ppe_status,
                'detected_ppe': detected_ppe,
                'missing_ppe': missing_ppe,
                'is_compliant': is_compliant,
                'violation_duration': violation_duration,
                'should_alert': should_alert,
                'grace_period_remaining': max(0, self.ppe_grace_period - violation_duration) if not is_compliant else 0
            }
        
        return compliance_results
    
    def _check_occupancy_violations(self, tracked_people: List[Dict], timestamp: float) -> List[Dict]:
        """Check for server room occupancy violations"""
        occupancy_events = []
        
        current_count = len(tracked_people)
        
        if current_count > self.max_occupancy:
            event_key = "server_occupancy_violation"
            
            if self._should_trigger_event(event_key, timestamp):
                occupancy_event = {
                    'type': DatacenterEventTypes.PEOPLE_COUNTING,
                    'severity': 'medium',
                    'zone_name': 'Server Room',
                    'zone_type': 'server_zone',
                    'people_count': current_count,
                    'occupancy_limit': self.max_occupancy,
                    'people': tracked_people,
                    'timestamp': timestamp,
                    'details': {
                        'violation_type': 'occupancy_exceeded',
                        'excess_count': current_count - self.max_occupancy,
                        'risk_level': 'high' if current_count > self.max_occupancy * 1.5 else 'medium'
                    }
                }
                
                occupancy_events.append(occupancy_event)
                self.server_stats['occupancy_violations'] += 1
                
                self.logger.warning(f"Server room occupancy exceeded: {current_count}/{self.max_occupancy}")
        
        return occupancy_events
    
    def _check_equipment_violations(self, tracked_people: List[Dict], timestamp: float) -> List[Dict]:
        """Check for unauthorized equipment access"""
        equipment_events = []
        
        for zone in self.equipment_zones:
            zone_name = zone.get('name', 'Equipment Zone')
            equipment_type = zone.get('equipment_type', 'server_rack')
            security_level = zone.get('security_level', 'high_security')
            
            people_near_equipment = []
            
            for person in tracked_people:
                if self._is_point_in_zone(person['center'], zone):
                    people_near_equipment.append(person)
                    
                    # Log equipment access
                    track_id = person['track_id']
                    access_key = f"{track_id}_{zone_name}"
                    
                    if access_key not in self.equipment_access_log:
                        self.equipment_access_log[access_key] = {
                            'first_access': timestamp,
                            'zone_name': zone_name,
                            'equipment_type': equipment_type,
                            'person': person
                        }
            
            # Check for violations based on security level
            if people_near_equipment and security_level in ['high_security', 'critical']:
                event_key = f"equipment_access_{zone_name}"
                
                if self._should_trigger_event(event_key, timestamp):
                    
                    # Check if people have proper authorization (simulate)
                    unauthorized_people = []
                    for person in people_near_equipment:
                        # In real system, check against access control database
                        if not self._check_equipment_authorization(person, zone):
                            unauthorized_people.append(person)
                    
                    if unauthorized_people:
                        equipment_event = {
                            'type': DatacenterEventTypes.UNAUTHORIZED_ACCESS,
                            'severity': 'critical' if security_level == 'critical' else 'high',
                            'zone_name': zone_name,
                            'zone_type': 'equipment_zone',
                            'equipment_type': equipment_type,
                            'security_level': security_level,
                            'people': unauthorized_people,
                            'timestamp': timestamp,
                            'details': {
                                'equipment_access': True,
                                'authorization_checked': True,
                                'people_count': len(unauthorized_people)
                            }
                        }
                        
                        equipment_events.append(equipment_event)
                        self.server_stats['equipment_violations'] += 1
                        
                        self.logger.error(f"Unauthorized equipment access in {zone_name}: {len(unauthorized_people)} people")
        
        return equipment_events
    
    def _check_air_gap_violations(self, tracked_people: List[Dict], timestamp: float) -> List[Dict]:
        """Check for air-gapped zone violations"""
        air_gap_events = []
        
        for zone in self.air_gap_zones:
            zone_name = zone.get('name', 'Air-Gap Zone')
            
            people_in_air_gap = []
            for person in tracked_people:
                if self._is_point_in_zone(person['center'], zone):
                    people_in_air_gap.append(person)
            
            # Air-gapped zones should have minimal access
            if people_in_air_gap:
                event_key = f"air_gap_violation_{zone_name}"
                
                if self._should_trigger_event(event_key, timestamp):
                    air_gap_event = {
                        'type': DatacenterEventTypes.UNAUTHORIZED_ACCESS,
                        'severity': 'critical',
                        'zone_name': zone_name,
                        'zone_type': 'air_gap_zone',
                        'people': people_in_air_gap,
                        'timestamp': timestamp,
                        'details': {
                            'air_gap_violation': True,
                            'security_level': 'critical',
                            'people_count': len(people_in_air_gap)
                        }
                    }
                    
                    air_gap_events.append(air_gap_event)
                    self.server_stats['air_gap_violations'] += 1
                    
                    self.logger.critical(f"Air-gap zone violation in {zone_name}: {len(people_in_air_gap)} people")
        
        return air_gap_events
    
    def _check_server_intrusion(self, tracked_people: List[Dict], timestamp: float) -> List[Dict]:
        """Check for general server room intrusion"""
        intrusion_events = []
        
        for zone in self.server_zones:
            zone_name = zone.get('name', 'Server Zone')
            
            for person in tracked_people:
                if self._is_point_in_zone(person['center'], zone):
                    
                    # Check authorization (simplified for demo)
                    track_id = person['track_id']
                    if not self._check_server_room_authorization(person, timestamp):
                        event_key = f"server_intrusion_{track_id}"
                        
                        if self._should_trigger_event(event_key, timestamp):
                            intrusion_event = {
                                'type': DatacenterEventTypes.INTRUSION,
                                'severity': 'high',
                                'zone_name': zone_name,
                                'zone_type': 'server_zone',
                                'person': person,
                                'timestamp': timestamp,
                                'details': {
                                    'server_room_access': True,
                                    'authorization_status': 'denied',
                                    'confidence': person.get('confidence', 0.8)
                                }
                            }
                            
                            intrusion_events.append(intrusion_event)
                            
                            self.logger.error(f"Server room intrusion detected: Person {track_id} in {zone_name}")
        
        return intrusion_events
    
    def _process_ppe_violations(self, ppe_compliance_results: Dict[int, Dict], timestamp: float) -> List[Dict]:
        """Process PPE compliance violations"""
        ppe_events = []
        
        for track_id, compliance in ppe_compliance_results.items():
            if compliance['should_alert']:  # Only alert after grace period
                event_key = f"ppe_violation_{track_id}"
                
                if self._should_trigger_event(event_key, timestamp):
                    ppe_event = {
                        'type': DatacenterEventTypes.PPE_VIOLATION,
                        'severity': 'medium',
                        'zone_name': 'Server Room',
                        'zone_type': 'server_zone',
                        'person': compliance['person'],
                        'timestamp': timestamp,
                        'details': {
                            'missing_ppe': compliance['missing_ppe'],
                            'detected_ppe': compliance['detected_ppe'],
                            'required_ppe': self.required_ppe_classes,
                            'violation_duration': compliance['violation_duration'],
                            'grace_period_expired': True
                        }
                    }
                    
                    ppe_events.append(ppe_event)
                    self.server_stats['ppe_violations'] += 1
                    
                    self.logger.warning(f"PPE violation: Person {track_id} missing {compliance['missing_ppe']}")
        
        return ppe_events
    
    def _check_equipment_authorization(self, person: Dict, equipment_zone: Dict) -> bool:
        """Check if person is authorized to access equipment (simulation)"""
        # In real system, integrate with access control database
        # For demo, simulate authorization based on equipment criticality
        equipment_type = equipment_zone.get('equipment_type', 'server_rack')
        security_level = equipment_zone.get('security_level', 'high_security')
        
        if security_level == 'critical':
            return False  # Very restricted access for demo
        
        # Simulate 70% authorization rate for high security equipment
        import random
        return random.random() < 0.7
    
    def _check_server_room_authorization(self, person: Dict, timestamp: float) -> bool:
        """Check if person is authorized for server room access"""
        # Simulate authorization check
        # In real system, integrate with badge readers and access control
        track_id = person['track_id']
        
        # Simulate 80% authorization rate for server room access
        import random
        return random.random() < 0.8
    
    def _update_server_statistics(self, tracked_people: List[Dict], timestamp: float):
        """Update server room monitoring statistics"""
        # Update total server time (person-minutes)
        if tracked_people:
            time_interval = 1/60  # Assume 1 second intervals, convert to minutes
            self.server_stats['total_server_time'] += len(tracked_people) * time_interval
        
        # Count authorized vs total entries
        for person in tracked_people:
            track_id = person['track_id']
            if track_id not in self.entry_timestamps:
                self.entry_timestamps[track_id] = timestamp
                # Simulate authorization check for entry counting
                if self._check_server_room_authorization(person, timestamp):
                    self.server_stats['authorized_entries'] += 1
    
    def _annotate_server_room_frame(self, frame: np.ndarray, tracked_people: List[Dict], 
                                   ppe_detections: List[Dict], 
                                   ppe_compliance: Dict[int, Dict]) -> np.ndarray:
        """
        Annotate frame with server room specific information
        
        Args:
            frame: Original video frame
            tracked_people: List of tracked person detections
            ppe_detections: List of PPE detections
            ppe_compliance: PPE compliance results per person
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw server zones
        for zone in self.server_zones:
            self._draw_zone(annotated_frame, zone, self.zone_colors.get('server_zone', (0, 255, 255)), 
                          alpha=0.2, label=zone.get('name', 'Server Zone'))
        
        # Draw equipment zones
        for zone in self.equipment_zones:
            self._draw_zone(annotated_frame, zone, (255, 165, 0), 
                          alpha=0.3, label=zone.get('name', 'Equipment Zone'))
        
        # Draw air-gap zones
        for zone in self.air_gap_zones:
            self._draw_zone(annotated_frame, zone, (128, 0, 128), 
                          alpha=0.4, label=zone.get('name', 'Air-Gap Zone'))
        
        # Draw PPE detections
        for ppe_item in ppe_detections:
            bbox = ppe_item['bbox']
            ppe_class = ppe_item['class_name']
            confidence = ppe_item['confidence']
            
            color = self.ppe_colors.get(ppe_class, (255, 255, 255))
            
            cv2.rectangle(annotated_frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 1)
            
            cv2.putText(annotated_frame, f"{ppe_class} {confidence:.2f}",
                       (int(bbox[0]), int(bbox[1]) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw people with PPE compliance status
        for person in tracked_people:
            bbox = person['bbox']
            track_id = person['track_id']
            
            # Get PPE compliance for this person
            compliance = ppe_compliance.get(track_id, {})
            is_compliant = compliance.get('is_compliant', False)
            missing_ppe = compliance.get('missing_ppe', [])
            
            # Color based on PPE compliance
            color = self.ppe_colors['compliant'] if is_compliant else self.ppe_colors['non_compliant']
            thickness = 2 if not is_compliant else 2
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, thickness)
            
            # Create label with PPE status
            label = f"ID:{track_id}"
            if not is_compliant:
                label += f" Missing: {', '.join(missing_ppe)}"
            else:
                label += " PPE OK"
            
            # Draw label with background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(annotated_frame,
                         (int(bbox[0]), int(bbox[1]) - label_size[1] - 10),
                         (int(bbox[0]) + label_size[0], int(bbox[1])),
                         color, -1)
            
            cv2.putText(annotated_frame, label,
                       (int(bbox[0]), int(bbox[1]) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw server room statistics
        stats_y = 30
        
        # Occupancy status
        occupancy_color = (0, 0, 255) if self.current_occupancy > self.max_occupancy else (0, 255, 0)
        occupancy_text = f"Occupancy: {self.current_occupancy}/{self.max_occupancy}"
        cv2.putText(annotated_frame, occupancy_text, (10, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, occupancy_color, 2)
        stats_y += 25
        
        # PPE compliance summary
        total_people = len(tracked_people)
        compliant_people = sum(1 for c in ppe_compliance.values() if c.get('is_compliant', False))
        ppe_text = f"PPE Compliance: {compliant_people}/{total_people}"
        ppe_color = (0, 255, 0) if compliant_people == total_people else (0, 165, 255)
        cv2.putText(annotated_frame, ppe_text, (10, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ppe_color, 2)
        stats_y += 25
        
        # Violation summary
        violation_text = f"Violations: PPE:{self.server_stats['ppe_violations']} Equipment:{self.server_stats['equipment_violations']}"
        cv2.putText(annotated_frame, violation_text, (10, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame
    
    def _handle_server_room_event(self, event: Dict, frame: np.ndarray, timestamp: float):
        """Handle server room events with appropriate severity"""
        try:
            event_type = event['type']
            severity = event['severity']
            zone_name = event.get('zone_name', 'Server Room')
            
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
            
            # Save event media for all server room events
            if self.db_writer:
                self._save_event_media(event_type, event, frame, timestamp, zone_name, event_id)
            
            # Send alerts for critical events
            if severity == 'critical' and self.sms_enabled:
                details = self._format_event_details(event)
                self._send_security_alert(
                    event_type=event_type.replace('_', ' ').title(),
                    details=details,
                    timestamp=timestamp
                )
            
            self.logger.info(f"Server room event handled: {event_type} in {zone_name} (severity: {severity})")
            
        except Exception as e:
            self.logger.error(f"Error handling server room event: {e}", exc_info=True)
    
    def _format_event_details(self, event: Dict) -> str:
        """Format event details for alerts"""
        event_type = event['type']
        details = []
        
        if event_type == DatacenterEventTypes.PPE_VIOLATION:
            missing_ppe = event['details'].get('missing_ppe', [])
            details.append(f"Missing PPE: {', '.join(missing_ppe)}")
        elif event_type == DatacenterEventTypes.PEOPLE_COUNTING:
            people_count = event.get('people_count', 0)
            limit = event.get('occupancy_limit', 0)
            details.append(f"Occupancy: {people_count}/{limit}")
        elif event_type == DatacenterEventTypes.UNAUTHORIZED_ACCESS:
            equipment_type = event['details'].get('equipment_type', 'equipment')
            details.append(f"Unauthorized {equipment_type} access")
        
        return "; ".join(details) if details else "Server room violation"
    
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
                # Queue video recording
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
            self.logger.error(f"Error saving server room event media: {e}")
    
    def _save_frame_with_detections(self, event_id: str, frame: np.ndarray, timestamp: float) -> Optional[str]:
        """Save frame with detections for event documentation"""
        try:
            # Generate filename with timestamp
            timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S_%f')
            frame_filename = f"server_room_cam{self.camera_id}_{timestamp_str}_{event_id[:8]}.jpg"
            frame_path = os.path.join(self.camera_output_dir, frame_filename)
            
            # Save the frame
            success = cv2.imwrite(frame_path, frame)
            if success:
                self.logger.debug(f"Saved server room event frame: {frame_path}")
                return frame_path
            else:
                self.logger.error(f"Failed to save frame: {frame_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error saving frame: {e}")
            return None
    
    def _trigger_video_recording(self, event_id: str, event_data: Dict, timestamp: float):
        """Trigger video recording for server room events"""
        try:
            # In a full implementation, this would use the frame buffer
            # to create a video clip around the event time
            self.logger.info(f"Video recording triggered for event {event_id}")
            
            # Queue video metadata for database
            if self.db_writer:
                video_metadata = {
                    'event_id': event_id,
                    'camera_id': self.camera_id,
                    'datacenter_id': self.datacenter_id,
                    'timestamp': timestamp,
                    'event_type': event_data.get('type', 'unknown'),
                    'duration': self.pre_event_seconds + self.post_event_seconds
                }
                self.db_writer.queue_video_metadata(video_metadata)
                
        except Exception as e:
            self.logger.error(f"Error triggering video recording: {e}")
    
    def _should_trigger_event(self, event_key: str, timestamp: float) -> bool:
        """Check if event should be triggered based on cooldown"""
        if event_key in self.recent_events:
            time_since_last = timestamp - self.recent_events[event_key]
            if time_since_last < self.event_cooldown:
                return False
        
        self.recent_events[event_key] = timestamp
        return True
    
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
                # Add background for label
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame,
                             (label_pos[0] - 2, label_pos[1] - label_size[1] - 2),
                             (label_pos[0] + label_size[0] + 2, label_pos[1] + 2),
                             (0, 0, 0), -1)
                cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
        except Exception as e:
            self.logger.error(f"Error drawing server room zone: {e}")
    
    def get_server_room_status(self) -> Dict:
        """Get current server room monitoring status"""
        return {
            'camera_id': self.camera_id,
            'datacenter_id': self.datacenter_id,
            'current_occupancy': self.current_occupancy,
            'max_occupancy': self.max_occupancy,
            'occupancy_percentage': (self.current_occupancy / max(self.max_occupancy, 1)) * 100,
            'ppe_enabled': self.ppe_detection_enabled,
            'required_ppe': self.required_ppe_classes,
            'active_people': len(self.tracked_objects),
            'statistics': self.server_stats.copy(),
            'zones_monitored': {
                'server_zones': len(self.server_zones),
                'equipment_zones': len(self.equipment_zones),
                'air_gap_zones': len(self.air_gap_zones)
            }
        }
    
    def cleanup(self):
        """Cleanup server room monitor resources"""
        try:
            super().cleanup()
            
            # Clear server room specific data
            self.person_ppe_status.clear()
            self.ppe_violation_timers.clear()
            self.equipment_access_log.clear()
            self.entry_timestamps.clear()
            
            self.logger.info("Server room monitor cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during server room monitor cleanup: {e}")

# Export the monitor class
__all__ = ['ServerRoomMonitor']