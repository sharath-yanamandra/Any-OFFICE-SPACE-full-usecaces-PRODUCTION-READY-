#!/usr/bin/env python3
"""
Script 16: critical_zone_monitor.py
File Path: src/camera_models/critical_zone_monitor.py

Datacenter Monitoring System - Critical Zone Monitoring

This module implements Phase 1 use cases for critical zones:
1. PPE Violation Detection (a1 - scalable, low effort)
2. Intrusion Detection (a1 - scalable, low effort) 
3. Unauthorized Access (derived from intrusion)
4. People Counting with strict limits (a2 - scalable, medium effort)
5. Equipment Tampering Detection (future - Phase 2)

Critical zones include:
- UPS rooms and power infrastructure
- Network operations centers (NOC)
- Core networking equipment areas
- Emergency systems and controls
- Fire suppression system areas
- Main electrical panels and distribution
"""

import cv2
import numpy as np
import time
import uuid
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from .camera_model_base import DatacenterCameraModelBase
from config import DatacenterConfig, DatacenterEventTypes, DatacenterZoneTypes
from logger import audit_logger

class CriticalZoneMonitor(DatacenterCameraModelBase):
    """
    Monitor for critical datacenter infrastructure zones implementing:
    - Strict PPE compliance (hard hat, safety vest, safety glasses)
    - Zero-tolerance intrusion detection
    - Equipment tampering detection
    - Minimal occupancy limits (typically 1-2 people max)
    - Enhanced security logging and alerts
    """

    def __init__(self, camera_id: int, datacenter_id: int, zones: Optional[Dict] = None, 
                 rules: Optional[List] = None, settings: Optional[Dict] = None, 
                 db=None, db_writer=None, frames_base_dir: str = 'frames'):
        """
        Initialize the critical zone monitor
        
        Args:
            camera_id: Camera identifier
            datacenter_id: Datacenter identifier
            zones: Zone definitions (critical_zones, equipment_zones)
            rules: Monitoring rules with enhanced security parameters
            settings: Camera-specific settings for critical monitoring
            db: Database instance
            db_writer: Database writer
            frames_base_dir: Base directory for frame storage
        """
        
        super().__init__(camera_id, datacenter_id, zones, rules, settings, 
                        db, db_writer, frames_base_dir)
        
        self.logger.info("Initializing CriticalZoneMonitor for mission-critical infrastructure")
        
        # Critical zone specific configuration
        self.critical_security_level = 'critical'
        self.enhanced_ppe_required = True
        self.zero_tolerance_intrusion = True
        self.equipment_tampering_detection = settings.get('equipment_tampering_detection', True)
        
        # Strict occupancy limits for critical zones
        self.max_occupancy_critical = settings.get('max_occupancy_critical', DatacenterConfig.get_occupancy_limit('critical_zone'))
        self.max_dwell_time = settings.get('max_dwell_time', 1800)  # 30 minutes max stay
        self.escort_required = settings.get('escort_required', True)
        
        # Enhanced PPE requirements for critical zones
        self.required_ppe_critical = settings.get('required_ppe_critical', [
            'hard_hat', 'safety_vest', 'safety_glasses', 'safety_gloves'
        ])
        self.ppe_grace_period = settings.get('ppe_grace_period', 15)  # 15 seconds to comply
        
        # Get zone configurations
        self.critical_zones = self._get_zones_by_type('critical_zone')
        self.equipment_zones = self._get_zones_by_type('equipment_zone') 
        
        # Critical zone tracking state
        self.zone_access_log = {}  # Track all access to critical zones
        self.ppe_compliance_tracking = {}  # Track PPE compliance per person
        self.dwell_time_tracking = {}  # Track how long people stay
        self.equipment_baseline = {}  # Baseline for equipment tampering detection
        
        # Enhanced security parameters
        self.immediate_alert_threshold = 0.9  # Confidence threshold for immediate alerts
        self.motion_sensitivity_critical = 0.3  # Lower threshold = higher sensitivity
        self.unauthorized_tolerance = 0  # Zero tolerance for unauthorized access
        
        # Equipment monitoring (for Phase 2 - basic implementation)
        self.equipment_monitoring_enabled = settings.get('equipment_monitoring', False)
        self.equipment_areas = []  # Will be populated from zone config
        
        # Critical zone statistics
        self.critical_stats = {
            'total_critical_access': 0,
            'ppe_violations': 0,
            'unauthorized_access': 0,
            'equipment_alerts': 0,
            'emergency_evacuations': 0,
            'compliance_violations': 0,
            'security_breaches': 0
        }
        
        # Enhanced alerting for critical zones
        self.escalation_contacts = settings.get('escalation_contacts', [])
        self.management_alert_enabled = settings.get('management_alert', True)
        self.security_team_alert = settings.get('security_team_alert', True)
        
        self.logger.info(f"Critical zone monitor initialized - Monitoring {len(self.critical_zones)} critical zones")
        self.logger.info(f"Enhanced PPE requirements: {self.required_ppe_critical}")
        self.logger.info(f"Max occupancy: {self.max_occupancy_critical}, Escort required: {self.escort_required}")
    
    def _get_zones_by_type(self, zone_type: str) -> List[Dict]:
        """Extract zones of specific type from zone configuration"""
        zones_of_type = []
        
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
        Process frame for critical zone monitoring with enhanced security
        
        Args:
            frame: Video frame
            timestamp: Frame timestamp
            detection_result: Person detection results from YOLO
            ppe_result: PPE detection results (mandatory for critical zones)
            
        Returns:
            Tuple of (annotated_frame, processing_results)
        """
        
        # Extract people and PPE detections
        people_detections, ppe_detections = self.detect_people_and_ppe(frame, detection_result, ppe_result)
        
        # Update object tracking with enhanced parameters
        tracked_people = self.update_object_tracking(people_detections)
        
        # Enhanced PPE compliance checking for critical zones
        tracked_people_with_ppe = self.check_enhanced_ppe_compliance(tracked_people, ppe_detections)
        
        # Store current people count
        self.current_people_count = len(tracked_people_with_ppe)
        
        # Critical zone access analysis
        access_analysis = self._analyze_critical_zone_access(tracked_people_with_ppe, timestamp)
        
        # Detect PPE violations with enhanced requirements
        ppe_violation_events = self._detect_ppe_violations(tracked_people_with_ppe, timestamp)
        
        # Detect unauthorized access with zero tolerance
        intrusion_events = self._detect_critical_intrusion(tracked_people_with_ppe, timestamp)
        
        # Check strict occupancy limits
        occupancy_events = self._check_critical_occupancy(tracked_people_with_ppe, timestamp)
        
        # Monitor dwell time limits
        dwell_time_events = self._check_dwell_time_violations(tracked_people_with_ppe, timestamp)
        
        # Equipment tampering detection (basic implementation for Phase 2)
        equipment_events = []
        if self.equipment_monitoring_enabled:
            equipment_events = self._detect_equipment_tampering(frame, tracked_people_with_ppe, timestamp)
        
        # Update critical zone tracking
        self._update_critical_tracking(tracked_people_with_ppe, timestamp)
        
        # Annotate frame with critical zone information
        annotated_frame = self._annotate_critical_frame(frame, tracked_people_with_ppe, access_analysis)
        
        # Handle all critical events with enhanced alerting
        all_events = ppe_violation_events + intrusion_events + occupancy_events + dwell_time_events + equipment_events
        
        for event in all_events:
            self._handle_critical_event(event, annotated_frame, timestamp)
        
        # Prepare comprehensive processing results
        processing_results = {
            'people_count': len(tracked_people_with_ppe),
            'critical_access_analysis': access_analysis,
            'events': all_events,
            'ppe_compliance_status': self._get_ppe_compliance_summary(tracked_people_with_ppe),
            'security_status': self._get_security_status_summary(),
            'critical_stats': self.critical_stats.copy()
        }
        
        # Update statistics
        self.stats['people_detected'] += len(tracked_people_with_ppe)
        if all_events:
            self.stats['events_detected'] += len(all_events)
            self.stats['security_events'] += len([e for e in all_events if e['severity'] in ['high', 'critical']])
        
        return annotated_frame, processing_results
    
    def check_enhanced_ppe_compliance(self, tracked_people: List[Dict], 
                                    ppe_detections: List[Dict]) -> List[Dict]:
        """
        Enhanced PPE compliance checking for critical zones
        
        Args:
            tracked_people: List of tracked person detections
            ppe_detections: List of PPE detections
            
        Returns:
            List of people with enhanced PPE compliance status
        """
        
        for person in tracked_people:
            person_bbox = person['bbox']
            track_id = person['track_id']
            
            # Enhanced PPE detection for critical zones
            detected_ppe = []
            missing_ppe = self.required_ppe_critical.copy()
            ppe_confidence_scores = {}
            
            # Find PPE items associated with this person
            for ppe in ppe_detections:
                if self._is_ppe_associated_with_person(person_bbox, ppe['bbox'], proximity_threshold=75.0):
                    ppe_class = ppe['class_name']
                    confidence = ppe['confidence']
                    
                    if confidence >= self.ppe_confidence_threshold:
                        detected_ppe.append(ppe_class)
                        ppe_confidence_scores[ppe_class] = confidence
                        
                        if ppe_class in missing_ppe:
                            missing_ppe.remove(ppe_class)
            
            # Enhanced compliance status
            person['detected_ppe'] = detected_ppe
            person['missing_ppe'] = missing_ppe
            person['ppe_confidence_scores'] = ppe_confidence_scores
            person['ppe_compliant'] = len(missing_ppe) == 0
            person['ppe_compliance_level'] = len(detected_ppe) / len(self.required_ppe_critical)
            
            # Track PPE compliance over time
            if track_id not in self.ppe_compliance_tracking:
                self.ppe_compliance_tracking[track_id] = {
                    'first_detection': time.time(),
                    'compliance_history': [],
                    'grace_period_used': False
                }
            
            self.ppe_compliance_tracking[track_id]['compliance_history'].append({
                'timestamp': time.time(),
                'compliant': person['ppe_compliant'],
                'detected_ppe': detected_ppe,
                'missing_ppe': missing_ppe
            })
            
            # Check grace period for PPE compliance
            time_since_detection = time.time() - self.ppe_compliance_tracking[track_id]['first_detection']
            person['ppe_grace_period_active'] = time_since_detection < self.ppe_grace_period
            
        return tracked_people
    
    def _analyze_critical_zone_access(self, tracked_people: List[Dict], timestamp: float) -> Dict:
        """
        Analyze access to critical zones with enhanced security logging
        
        Args:
            tracked_people: List of tracked person detections
            timestamp: Current timestamp
            
        Returns:
            Dictionary with critical zone access analysis
        """
        access_analysis = {
            'critical_zone_occupancy': {},
            'security_violations': [],
            'access_events': [],
            'compliance_summary': {}
        }
        
        for zone in self.critical_zones:
            zone_name = zone.get('name', 'Critical Zone')
            security_level = zone.get('security_level', 'critical')
            people_in_zone = []
            
            # Find people in this critical zone
            for person in tracked_people:
                if self._is_point_in_zone(person['center'], zone):
                    people_in_zone.append(person)
                    
                    # Log critical zone access
                    track_id = person['track_id']
                    self._log_critical_access(track_id, zone_name, timestamp, person)
                    
                    # Check escort requirements
                    if self.escort_required:
                        escort_present = self._check_escort_presence(people_in_zone, person)
                        person['escort_present'] = escort_present
                        
                        if not escort_present and len(people_in_zone) == 1:
                            access_analysis['security_violations'].append({
                                'type': 'no_escort',
                                'person': person,
                                'zone': zone_name,
                                'severity': 'high'
                            })
            
            access_analysis['critical_zone_occupancy'][zone_name] = len(people_in_zone)
            
            if people_in_zone:
                access_analysis['access_events'].append({
                    'zone_name': zone_name,
                    'security_level': security_level,
                    'people_count': len(people_in_zone),
                    'people': people_in_zone,
                    'timestamp': timestamp,
                    'compliance_rate': self._calculate_zone_compliance_rate(people_in_zone)
                })
        
        return access_analysis
    
    def _log_critical_access(self, track_id: int, zone_name: str, timestamp: float, person: Dict):
        """Log access to critical zones for audit trail"""
        
        if track_id not in self.zone_access_log:
            self.zone_access_log[track_id] = []
        
        # Check if this is a new access event
        recent_access = [log for log in self.zone_access_log[track_id] 
                        if log['zone_name'] == zone_name and timestamp - log['timestamp'] < 60]
        
        if not recent_access:
            access_event = {
                'zone_name': zone_name,
                'timestamp': timestamp,
                'entry_time': timestamp,
                'ppe_compliant': person.get('ppe_compliant', False),
                'detected_ppe': person.get('detected_ppe', []),
                'missing_ppe': person.get('missing_ppe', [])
            }
            
            self.zone_access_log[track_id].append(access_event)
            self.critical_stats['total_critical_access'] += 1
            
            # Audit logging for critical access
            audit_logger.log_access_event(
                datacenter_id=str(self.datacenter_id),
                zone_id=zone_name,
                person_id=str(track_id),
                access_type='entry',
                result='allowed'  # In real system, check authorization
            )
    
    def _check_escort_presence(self, people_in_zone: List[Dict], target_person: Dict) -> bool:
        """Check if person has required escort in critical zone"""
        
        if len(people_in_zone) < 2:
            return False
        
        # Simple escort detection - in real system, identify authorized escorts
        # For now, assume if there are 2+ people, one could be an escort
        return len(people_in_zone) >= 2
    
    def _calculate_zone_compliance_rate(self, people_in_zone: List[Dict]) -> float:
        """Calculate PPE compliance rate for people in zone"""
        
        if not people_in_zone:
            return 1.0
        
        compliant_count = sum(1 for person in people_in_zone if person.get('ppe_compliant', False))
        return compliant_count / len(people_in_zone)
    
    def _detect_ppe_violations(self, tracked_people: List[Dict], timestamp: float) -> List[Dict]:
        """
        Detect PPE violations in critical zones with enhanced requirements
        
        Args:
            tracked_people: List of tracked person detections
            timestamp: Current timestamp
            
        Returns:
            List of PPE violation events
        """
        ppe_violation_events = []
        
        for zone in self.critical_zones:
            zone_name = zone.get('name', 'Critical Zone')
            
            for person in tracked_people:
                if self._is_point_in_zone(person['center'], zone):
                    track_id = person['track_id']
                    
                    # Check PPE compliance
                    if not person.get('ppe_compliant', False):
                        
                        # Check if grace period has expired
                        grace_period_active = person.get('ppe_grace_period_active', True)
                        
                        if not grace_period_active:
                            event_key = f"ppe_violation_{zone_name}_{track_id}"
                            
                            if self._should_trigger_critical_event(event_key, timestamp):
                                
                                ppe_violation_event = {
                                    'type': DatacenterEventTypes.PPE_VIOLATION,
                                    'severity': 'high',  # Always high in critical zones
                                    'zone_name': zone_name,
                                    'zone_type': 'critical_zone',
                                    'person': person,
                                    'timestamp': timestamp,
                                    'details': {
                                        'required_ppe': self.required_ppe_critical,
                                        'detected_ppe': person.get('detected_ppe', []),
                                        'missing_ppe': person.get('missing_ppe', []),
                                        'compliance_level': person.get('ppe_compliance_level', 0),
                                        'grace_period_expired': True,
                                        'confidence_scores': person.get('ppe_confidence_scores', {})
                                    }
                                }
                                
                                ppe_violation_events.append(ppe_violation_event)
                                self.critical_stats['ppe_violations'] += 1
                                
                                self.logger.error(f"PPE violation in critical zone {zone_name}: "
                                                f"Person {track_id} missing {person.get('missing_ppe', [])}")
        
        return ppe_violation_events
    
    def _detect_critical_intrusion(self, tracked_people: List[Dict], timestamp: float) -> List[Dict]:
        """
        Detect intrusion in critical zones with zero tolerance
        
        Args:
            tracked_people: List of tracked person detections
            timestamp: Current timestamp
            
        Returns:
            List of critical intrusion events
        """
        intrusion_events = []
        
        for zone in self.critical_zones:
            zone_name = zone.get('name', 'Critical Zone')
            security_level = zone.get('security_level', 'critical')
            
            for person in tracked_people:
                if self._is_point_in_zone(person['center'], zone):
                    track_id = person['track_id']
                    
                    # Zero tolerance - any unauthorized access is intrusion
                    is_authorized = self._check_critical_authorization(person, zone, timestamp)
                    
                    if not is_authorized:
                        event_key = f"critical_intrusion_{zone_name}_{track_id}"
                        
                        # Immediate triggering for critical intrusion
                        intrusion_event = {
                            'type': DatacenterEventTypes.INTRUSION,
                            'severity': 'critical',
                            'zone_name': zone_name,
                            'zone_type': 'critical_zone',
                            'security_level': security_level,
                            'person': person,
                            'timestamp': timestamp,
                            'details': {
                                'authorization_status': 'denied',
                                'security_clearance_required': True,
                                'escort_required': self.escort_required,
                                'escort_present': person.get('escort_present', False),
                                'detection_confidence': person.get('confidence', 0.0),
                                'immediate_response_required': True
                            }
                        }
                        
                        intrusion_events.append(intrusion_event)
                        self.critical_stats['unauthorized_access'] += 1
                        self.critical_stats['security_breaches'] += 1
                        
                        self.logger.critical(f"CRITICAL INTRUSION: Unauthorized access to {zone_name} "
                                           f"by person {track_id}")
        
        return intrusion_events
    
    def _check_critical_authorization(self, person: Dict, zone: Dict, timestamp: float) -> bool:
        """
        Check authorization for critical zone access
        
        Args:
            person: Person detection dictionary
            zone: Critical zone configuration
            timestamp: Current timestamp
            
        Returns:
            True if access is authorized
        """
        
        # In a real system, this would check:
        # 1. Badge scan records
        # 2. Security clearance levels
        # 3. Time-based access permissions
        # 4. Escort requirements
        # 5. Special authorization for critical zones
        
        track_id = person['track_id']
        
        # Simulate strict authorization check
        # Check PPE compliance as prerequisite
        if not person.get('ppe_compliant', False):
            return False
        
        # Check escort requirement
        if self.escort_required and not person.get('escort_present', False):
            return False
        
        # For demo: very strict authorization (only 10% pass)
        import random
        return random.random() < 0.1
    
    def _check_critical_occupancy(self, tracked_people: List[Dict], timestamp: float) -> List[Dict]:
        """
        Check strict occupancy limits for critical zones
        
        Args:
            tracked_people: List of tracked person detections
            timestamp: Current timestamp
            
        Returns:
            List of occupancy violation events
        """
        occupancy_events = []
        
        for zone in self.critical_zones:
            zone_name = zone.get('name', 'Critical Zone')
            max_occupancy = min(zone.get('max_occupancy', self.max_occupancy_critical), 
                              self.max_occupancy_critical)
            
            people_in_zone = []
            for person in tracked_people:
                if self._is_point_in_zone(person['center'], zone):
                    people_in_zone.append(person)
            
            if len(people_in_zone) > max_occupancy:
                event_key = f"critical_occupancy_{zone_name}"
                
                if self._should_trigger_critical_event(event_key, timestamp):
                    
                    occupancy_event = {
                        'type': DatacenterEventTypes.PEOPLE_COUNTING,
                        'severity': 'high',  # Always high for critical zones
                        'zone_name': zone_name,
                        'zone_type': 'critical_zone',
                        'people_count': len(people_in_zone),
                        'max_occupancy': max_occupancy,
                        'people': people_in_zone,
                        'timestamp': timestamp,
                        'details': {
                            'violation_type': 'critical_occupancy_exceeded',
                            'excess_count': len(people_in_zone) - max_occupancy,
                            'security_risk_level': 'high',
                            'immediate_action_required': True
                        }
                    }
                    
                    occupancy_events.append(occupancy_event)
                    self.critical_stats['compliance_violations'] += 1
                    
                    self.logger.error(f"Critical occupancy violation in {zone_name}: "
                                    f"{len(people_in_zone)}/{max_occupancy} people")
        
        return occupancy_events
    
    def _check_dwell_time_violations(self, tracked_people: List[Dict], timestamp: float) -> List[Dict]:
        """
        Check for excessive dwell time in critical zones
        
        Args:
            tracked_people: List of tracked person detections
            timestamp: Current timestamp
            
        Returns:
            List of dwell time violation events
        """
        dwell_time_events = []
        
        for person in tracked_people:
            track_id = person['track_id']
            
            # Check if person is in any critical zone
            for zone in self.critical_zones:
                if self._is_point_in_zone(person['center'], zone):
                    zone_name = zone.get('name', 'Critical Zone')
                    
                    # Track dwell time
                    if track_id not in self.dwell_time_tracking:
                        self.dwell_time_tracking[track_id] = {
                            'zone_name': zone_name,
                            'entry_time': timestamp,
                            'last_seen': timestamp
                        }
                    else:
                        self.dwell_time_tracking[track_id]['last_seen'] = timestamp
                        
                        # Check if dwell time exceeds limit
                        dwell_time = timestamp - self.dwell_time_tracking[track_id]['entry_time']
                        
                        if dwell_time > self.max_dwell_time:
                            event_key = f"dwell_time_{zone_name}_{track_id}"
                            
                            if self._should_trigger_critical_event(event_key, timestamp, cooldown=300):  # 5 min cooldown
                                
                                dwell_event = {
                                    'type': 'excessive_dwell_time',
                                    'severity': 'medium',
                                    'zone_name': zone_name,
                                    'zone_type': 'critical_zone',
                                    'person': person,
                                    'timestamp': timestamp,
                                    'details': {
                                        'dwell_time_seconds': dwell_time,
                                        'max_allowed_seconds': self.max_dwell_time,
                                        'entry_time': self.dwell_time_tracking[track_id]['entry_time'],
                                        'requires_escort_check': True
                                    }
                                }
                                
                                dwell_time_events.append(dwell_event)
                                
                                self.logger.warning(f"Excessive dwell time in {zone_name}: "
                                                  f"Person {track_id} for {dwell_time/60:.1f} minutes")
        
        return dwell_time_events
    
    def _detect_equipment_tampering(self, frame: np.ndarray, tracked_people: List[Dict], 
                                  timestamp: float) -> List[Dict]:
        """
        Basic equipment tampering detection for Phase 2 preparation
        
        Args:
            frame: Current video frame
            tracked_people: List of tracked person detections
            timestamp: Current timestamp
            
        Returns:
            List of equipment tampering events
        """
        equipment_events = []
        
        # Basic implementation - detect people near equipment areas
        # In Phase 2, this would include:
        # - Motion detection around equipment
        # - Object removal/addition detection
        # - Abnormal interaction patterns
        # - Integration with equipment sensors
        
        for zone in self.equipment_zones:
            zone_name = zone.get('name', 'Equipment Zone')
            
            people_near_equipment = []
            for person in tracked_people:
                if self._is_point_in_zone(person['center'], zone):
                    people_near_equipment.append(person)
            
            # Basic tampering detection - unauthorized people near equipment
            if people_near_equipment:
                for person in people_near_equipment:
                    if not self._check_equipment_access_authorization(person, zone):
                        
                        equipment_event = {
                            'type': 'equipment_tampering',
                            'severity': 'high',
                            'zone_name': zone_name,
                            'zone_type': 'equipment_zone',
                            'person': person,
                            'timestamp': timestamp,
                            'details': {
                                'equipment_type': zone.get('equipment_type', 'unknown'),
                                'unauthorized_proximity': True,
                                'detection_method': 'proximity_based'
                            }
                        }
                        
                        equipment_events.append(equipment_event)
                        self.critical_stats['equipment_alerts'] += 1
        
        return equipment_events
    
    def _check_equipment_access_authorization(self, person: Dict, equipment_zone: Dict) -> bool:
        """Check if person is authorized to access equipment"""
        
        # Basic authorization check - in real system would check:
        # - Technical clearance levels
        # - Equipment-specific permissions
        # - Maintenance schedules
        # - Escort requirements for equipment access
        
        # For now, require PPE compliance as minimum
        return person.get('ppe_compliant', False)
    
    def _should_trigger_critical_event(self, event_key: str, timestamp: float, 
                                     cooldown: int = None) -> bool:
        """
        Check if critical event should be triggered (shorter cooldown than normal)
        
        Args:
            event_key: Unique event identifier
            timestamp: Current timestamp
            cooldown: Custom cooldown period (default: half of normal cooldown)
            
        Returns:
            True if event should be triggered
        """
        if cooldown is None:
            cooldown = self.event_cooldown // 2  # Shorter cooldown for critical zones
        
        if event_key in self.recent_events:
            time_since_last = timestamp - self.recent_events[event_key]
            if time_since_last < cooldown:
                return False
        
        self.recent_events[event_key] = timestamp
        return True
    
    def _update_critical_tracking(self, tracked_people: List[Dict], timestamp: float):
        """Update critical zone tracking state"""
        
        # Clean up old dwell time tracking
        active_track_ids = {person['track_id'] for person in tracked_people}
        
        for track_id in list(self.dwell_time_tracking.keys()):
            if track_id not in active_track_ids:
                # Person left the area - log exit
                exit_time = timestamp
                entry_time = self.dwell_time_tracking[track_id]['entry_time']
                total_dwell_time = exit_time - entry_time
                
                # Log exit event for audit
                audit_logger.log_access_event(
                    datacenter_id=str(self.datacenter_id),
                    zone_id=self.dwell_time_tracking[track_id]['zone_name'],
                    person_id=str(track_id),
                    access_type='exit',
                    result='logged'
                )
                
                # Clean up tracking
                del self.dwell_time_tracking[track_id]
                
                self.logger.info(f"Person {track_id} exited critical zone after {total_dwell_time/60:.1f} minutes")
    
    def _handle_critical_event(self, event: Dict, frame: np.ndarray, timestamp: float):
        """
        Handle critical zone events with enhanced alerting and escalation
        
        Args:
            event: Event dictionary
            frame: Current video frame
            timestamp: Event timestamp
        """
        try:
            event_type = event['type']
            severity = event['severity']
            zone_name = event.get('zone_name', 'Unknown Critical Zone')
            
            # Enhanced audit logging for critical events
            audit_logger.log_event_detection(
                event_type=event_type,
                camera_id=str(self.camera_id),
                datacenter_id=str(self.datacenter_id),
                severity=severity,
                detection_data=event.get('details', {})
            )
            
            # Generate unique event ID
            event_id = str(uuid.uuid4())
            
            # Always save media for critical zone events
            if self.db_writer:
                self._save_critical_event_media(event_type, event, frame, timestamp, zone_name, event_id)
            
            # Enhanced alerting for critical events
            if severity in ['high', 'critical']:
                
                # Immediate SMS alert for critical events
                if self.sms_enabled:
                    self._send_critical_security_alert(event_type, event, timestamp)
                
                # Management escalation for critical intrusions
                if event_type == DatacenterEventTypes.INTRUSION and severity == 'critical':
                    self._escalate_to_management(event, timestamp)
                
                # Security team notification
                if self.security_team_alert:
                    self._notify_security_team(event, timestamp)
            
            # Log critical event
            self.logger.critical(f"Critical zone event: {event_type} in {zone_name} (severity: {severity})")
            
        except Exception as e:
            self.logger.error(f"Error handling critical event: {e}", exc_info=True)
    
    def _save_critical_event_media(self, event_type: str, event_data: Dict, frame: np.ndarray, 
                                 timestamp: float, zone_name: str, event_id: str):
        """Save event media with enhanced metadata for critical zones"""
        try:
            # Always save both image and video for critical events
            frame_path = None
            video_path = None
            
            # Save annotated frame
            frame_path = self._save_frame_with_detections(event_id, frame, timestamp)
            if frame_path:
                self.stats['frames_saved'] += 1
            
            # Always trigger video recording for critical events
            self._trigger_video_recording(event_id, event_data, timestamp)
            self.stats['videos_saved'] += 1
            
            # Queue enhanced event data for database
            if self.db_writer:
                db_event_data = {
                    'event_id': event_id,
                    'camera_id': self.camera_id,
                    'datacenter_id': self.datacenter_id,
                    'event_type': event_type,
                    'severity': event_data.get('severity', 'high'),
                    'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
                    'zone_name': zone_name,
                    'zone_type': 'critical_zone',
                    'metadata': {
                        **event_data,
                        'critical_zone_event': True,
                        'requires_immediate_response': True,
                        'compliance_impact': 'high',
                        'security_clearance_required': True
                    }
                }
                
                self.db_writer.queue_event(db_event_data)
                
        except Exception as e:
            self.logger.error(f"Error saving critical event media: {e}")
    
    def _send_critical_security_alert(self, event_type: str, event_data: Dict, timestamp: float):
        """Send enhanced SMS alert for critical zone events"""
        try:
            if not self.sms_enabled:
                return False
            
            # Format timestamp
            import pytz
            ist_tz = pytz.timezone('Asia/Kolkata')
            dt = datetime.fromtimestamp(timestamp, tz=ist_tz)
            formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S IST')
            
            # Create detailed alert message
            zone_name = event_data.get('zone_name', 'Critical Zone')
            severity = event_data.get('severity', 'high')
            people_count = event_data.get('people_count', 1)
            
            message = (
                f"🚨 CRITICAL DATACENTER ALERT 🚨\n\n"
                f"Event: {event_type.replace('_', ' ').title()}\n"
                f"Severity: {severity.upper()}\n"
                f"Zone: {zone_name}\n"
                f"Datacenter: {self.datacenter_id}\n"
                f"Camera: {self.camera_id}\n"
                f"People: {people_count}\n"
                f"Time: {formatted_time}\n\n"
            )
            
            # Add event-specific details
            if event_type == DatacenterEventTypes.PPE_VIOLATION:
                missing_ppe = event_data.get('details', {}).get('missing_ppe', [])
                message += f"Missing PPE: {', '.join(missing_ppe)}\n"
            elif event_type == DatacenterEventTypes.INTRUSION:
                message += "UNAUTHORIZED ACCESS TO CRITICAL ZONE\n"
            
            message += "\n⚠️ IMMEDIATE RESPONSE REQUIRED ⚠️"
            
            # Send to all alert numbers
            sent_count = 0
            for phone_number in DatacenterConfig.ALERT_PHONE_NUMBERS:
                try:
                    message_obj = self.twilio_client.messages.create(
                        body=message,
                        from_=DatacenterConfig.TWILIO_PHONE_NUMBER,
                        to=phone_number
                    )
                    sent_count += 1
                    self.logger.info(f"Critical alert sent to {phone_number}, SID: {message_obj.sid}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to send critical alert to {phone_number}: {e}")
            
            return sent_count > 0
            
        except Exception as e:
            self.logger.error(f"Error sending critical security alert: {e}")
            return False
    
    def _escalate_to_management(self, event: Dict, timestamp: float):
        """Escalate critical intrusion to management"""
        try:
            if not self.management_alert_enabled:
                return
            
            # Log management escalation
            self.logger.critical(f"MANAGEMENT ESCALATION: Critical intrusion in {event.get('zone_name')}")
            
            # In real system, this would:
            # 1. Send email to management
            # 2. Create high-priority ticket
            # 3. Activate emergency protocols
            # 4. Notify security contractors
            
            # For now, just log the escalation
            audit_logger.log_system_event(
                component='critical_zone_monitor',
                event='management_escalation',
                status='triggered',
                details={
                    'event_type': event['type'],
                    'zone_name': event.get('zone_name'),
                    'severity': event.get('severity'),
                    'timestamp': timestamp
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error escalating to management: {e}")
    
    def _notify_security_team(self, event: Dict, timestamp: float):
        """Notify security team of critical event"""
        try:
            # Log security team notification
            audit_logger.log_system_event(
                component='critical_zone_monitor',
                event='security_team_notification',
                status='sent',
                details={
                    'event_type': event['type'],
                    'zone_name': event.get('zone_name'),
                    'severity': event.get('severity'),
                    'timestamp': timestamp
                }
            )
            
            self.logger.info(f"Security team notified of {event['type']} in {event.get('zone_name')}")
            
        except Exception as e:
            self.logger.error(f"Error notifying security team: {e}")
    
    def _get_ppe_compliance_summary(self, tracked_people: List[Dict]) -> Dict:
        """Get PPE compliance summary for critical zones"""
        
        if not tracked_people:
            return {'compliance_rate': 1.0, 'violations': 0, 'compliant_people': 0}
        
        compliant_count = sum(1 for person in tracked_people if person.get('ppe_compliant', False))
        violation_count = len(tracked_people) - compliant_count
        
        return {
            'compliance_rate': compliant_count / len(tracked_people),
            'violations': violation_count,
            'compliant_people': compliant_count,
            'total_people': len(tracked_people),
            'required_ppe': self.required_ppe_critical
        }
    
    def _get_security_status_summary(self) -> Dict:
        """Get overall security status summary"""
        
        return {
            'security_level': 'critical',
            'zero_tolerance_mode': self.zero_tolerance_intrusion,
            'escort_required': self.escort_required,
            'enhanced_monitoring': True,
            'immediate_response_enabled': True,
            'management_alerts_enabled': self.management_alert_enabled,
            'current_threat_level': self._assess_current_threat_level()
        }
    
    def _assess_current_threat_level(self) -> str:
        """Assess current threat level based on recent events"""
        
        # Simple threat assessment based on recent violations
        recent_violations = (
            self.critical_stats['unauthorized_access'] + 
            self.critical_stats['security_breaches'] + 
            self.critical_stats['equipment_alerts']
        )
        
        if recent_violations == 0:
            return 'green'
        elif recent_violations <= 2:
            return 'yellow'
        elif recent_violations <= 5:
            return 'orange'
        else:
            return 'red'
    
    def _annotate_critical_frame(self, frame: np.ndarray, tracked_people: List[Dict], 
                                access_analysis: Dict) -> np.ndarray:
        """
        Annotate frame with critical zone information and security status
        
        Args:
            frame: Original video frame
            tracked_people: List of tracked person detections
            access_analysis: Critical zone access analysis
            
        Returns:
            Annotated frame with critical zone visualization
        """
        annotated_frame = frame.copy()
        
        # Draw critical zones with enhanced visualization
        for zone in self.critical_zones:
            zone_name = zone.get('name', 'Critical Zone')
            
            # Use red color for critical zones
            color = self.zone_colors.get('critical_zone', (128, 0, 128))
            
            # Enhanced zone drawing with security level indicator
            self._draw_critical_zone(annotated_frame, zone, color, alpha=0.4, 
                                   label=f"CRITICAL: {zone_name}")
        
        # Draw equipment zones if enabled
        for zone in self.equipment_zones:
            zone_name = zone.get('name', 'Equipment Zone')
            self._draw_critical_zone(annotated_frame, zone, (255, 165, 0), alpha=0.3, 
                                   label=f"EQUIPMENT: {zone_name}")
        
        # Draw people with enhanced PPE status
        for person in tracked_people:
            bbox = person['bbox']
            track_id = person['track_id']
            ppe_compliant = person.get('ppe_compliant', False)
            
            # Color coding based on compliance and location
            if ppe_compliant:
                color = (0, 255, 0)  # Green for compliant
                label = f"ID:{track_id} ✓ PPE"
            else:
                color = (0, 0, 255)  # Red for non-compliant
                missing_ppe = person.get('missing_ppe', [])
                label = f"ID:{track_id} ✗ Missing: {', '.join(missing_ppe[:2])}"
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 3)  # Thicker border for critical zones
            
            # Draw enhanced label with PPE status
            cv2.putText(annotated_frame, label,
                       (int(bbox[0]), int(bbox[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw PPE compliance indicator
            compliance_level = person.get('ppe_compliance_level', 0)
            compliance_text = f"PPE: {compliance_level*100:.0f}%"
            cv2.putText(annotated_frame, compliance_text,
                       (int(bbox[0]), int(bbox[3]) + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw critical zone occupancy and status
        y_offset = 30
        for zone_name, count in access_analysis['critical_zone_occupancy'].items():
            max_occupancy = self.max_occupancy_critical
            occupancy_color = (0, 255, 0) if count <= max_occupancy else (0, 0, 255)
            
            occupancy_text = f"{zone_name}: {count}/{max_occupancy} (CRITICAL)"
            cv2.putText(annotated_frame, occupancy_text,
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, occupancy_color, 2)
            y_offset += 25
        
        # Draw security status
        threat_level = self._assess_current_threat_level()
        threat_colors = {'green': (0, 255, 0), 'yellow': (0, 255, 255), 
                        'orange': (0, 165, 255), 'red': (0, 0, 255)}
        
        threat_text = f"THREAT LEVEL: {threat_level.upper()}"
        cv2.putText(annotated_frame, threat_text,
                   (10, annotated_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, threat_colors.get(threat_level, (255, 255, 255)), 2)
        
        # Draw critical statistics
        stats_text = (f"Critical Events: {self.critical_stats['unauthorized_access']} | "
                     f"PPE Violations: {self.critical_stats['ppe_violations']} | "
                     f"Security Breaches: {self.critical_stats['security_breaches']}")
        cv2.putText(annotated_frame, stats_text,
                   (10, annotated_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 255, 255), 1)
        
        return annotated_frame
    
    def _draw_critical_zone(self, frame: np.ndarray, zone: Dict, color: Tuple[int, int, int], 
                           alpha: float = 0.4, label: Optional[str] = None):
        """Draw critical zone with enhanced security visualization"""
        try:
            coordinates = zone.get('coordinates', [])
            if len(coordinates) < 3:
                return
            
            # Create overlay with higher opacity for critical zones
            overlay = frame.copy()
            polygon = np.array(coordinates, dtype=np.int32)
            cv2.fillPoly(overlay, [polygon], color)
            
            # Apply transparency
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Draw thick border for critical zones
            cv2.polylines(frame, [polygon], True, color, 3)
            
            # Draw security level indicator
            if label and len(coordinates) > 0:
                label_pos = (int(coordinates[0][0]), int(coordinates[0][1]))
                
                # Draw background for label
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, 
                             (label_pos[0], label_pos[1] - label_size[1] - 5),
                             (label_pos[0] + label_size[0], label_pos[1] + 5),
                             (0, 0, 0), -1)
                
                # Draw label
                cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
        except Exception as e:
            self.logger.error(f"Error drawing critical zone: {e}")
    
    def get_critical_zone_stats(self) -> Dict:
        """Get comprehensive statistics for critical zone monitoring"""
        
        base_stats = self.get_current_stats()
        
        critical_specific_stats = {
            'critical_stats': self.critical_stats,
            'ppe_compliance_summary': self._get_ppe_compliance_summary([]),
            'security_status': self._get_security_status_summary(),
            'active_critical_zones': len(self.critical_zones),
            'equipment_zones_monitored': len(self.equipment_zones),
            'current_threat_level': self._assess_current_threat_level(),
            'dwell_time_violations': len(self.dwell_time_tracking),
            'enhanced_monitoring_active': True
        }
        
        return {**base_stats, **critical_specific_stats}

# Export the critical zone monitor class
__all__ = ['CriticalZoneMonitor']