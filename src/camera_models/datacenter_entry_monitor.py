#!/usr/bin/env python3
"""
Script 12: datacenter_entry_monitor.py
File Path: src/camera_models/datacenter_entry_monitor.py

Datacenter Monitoring System - Entry Point Monitoring

This module implements Phase 1 use cases for entry points:
1. Tailgating Detection (a1 - scalable, low effort)
2. Intrusion Detection (a1 - scalable, low effort) 
3. People Counting (a2 - scalable, medium effort)
4. Unauthorized Access (derived from intrusion)

Entry point monitoring focuses on:
- Single person entry validation
- Multiple people detection (tailgating)
- Badge-less entry detection
- Entry zone occupancy monitoring
- Time-based access control
"""

import cv2
import numpy as np
import time
import uuid
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from .camera_model_base import DatacenterCameraModelBase
from config import DatacenterConfig, DatacenterEventTypes
from logger import audit_logger

class DatacenterEntryMonitor(DatacenterCameraModelBase):
    """
    Monitor for datacenter entry points implementing Phase 1 use cases:
    - Tailgating detection (multiple people with single badge scan)
    - Intrusion detection (unauthorized entry attempts) 
    - People counting (entry zone occupancy)
    - Access control violations
    """

    def __init__(self, camera_id: int, datacenter_id: int, zones: Optional[Dict] = None, 
                 rules: Optional[List] = None, settings: Optional[Dict] = None, 
                 db=None, db_writer=None, frames_base_dir: str = 'frames'):
        """
        Initialize the datacenter entry monitor
        
        Args:
            camera_id: Camera identifier
            datacenter_id: Datacenter identifier
            zones: Zone definitions (entry_zones, restricted_zones)
            rules: Monitoring rules for entry events
            settings: Camera-specific settings
            db: Database instance
            db_writer: Database writer
            frames_base_dir: Base directory for frame storage
        """
        
        super().__init__(camera_id, datacenter_id, zones, rules, settings, 
                        db, db_writer, frames_base_dir)
        
        self.logger.info("Initializing DatacenterEntryMonitor")
        
        # Entry-specific configuration
        self.tailgating_enabled = True
        self.max_people_per_entry = settings.get('max_people_per_entry', DatacenterConfig.MAX_PEOPLE_PER_ENTRY)
        self.entry_time_window = settings.get('entry_time_window', DatacenterConfig.TAILGATING_TIME_WINDOW)
        self.entry_zone_buffer = settings.get('entry_zone_buffer', DatacenterConfig.ENTRY_ZONE_BUFFER)
        
        # Get zone configurations
        self.entry_zones = self._get_zones_by_type('entry_zone')
        self.restricted_zones = self._get_zones_by_type('restricted_zone')
        
        # Entry tracking state
        self.entry_events = {}  # Track people entering zones
        self.last_entry_times = {}  # Track last entry time per zone
        self.zone_occupancy = {}  # Current occupancy per zone
        
        # Tailgating detection parameters
        self.simultaneous_entry_threshold = 2.0  # seconds for simultaneous detection
        self.recent_entries = {}  # Track recent entries for tailgating detection
        
        # Access control simulation (in real system, integrate with badge readers)
        self.authorized_entry_simulation = True
        self.badge_scan_window = 5.0  # seconds to associate person with badge scan
        
        # Statistics for entry monitoring
        self.entry_stats = {
            'total_entries': 0,
            'authorized_entries': 0,
            'tailgating_events': 0,
            'intrusion_attempts': 0,
            'occupancy_violations': 0
        }
        
        self.logger.info(f"Entry monitor initialized - Max people per entry: {self.max_people_per_entry}")
        self.logger.info(f"Monitoring {len(self.entry_zones)} entry zones and {len(self.restricted_zones)} restricted zones")
    
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
        Process frame for entry point monitoring
        
        Args:
            frame: Video frame
            timestamp: Frame timestamp
            detection_result: Person detection results from YOLO
            ppe_result: PPE detection results (optional for entry points)
            
        Returns:
            Tuple of (annotated_frame, processing_results)
        """
        
        # Extract people detections
        people_detections, ppe_detections = self.detect_people_and_ppe(frame, detection_result, ppe_result)
        
        # Update object tracking
        tracked_people = self.update_object_tracking(people_detections)
        
        # Store current people count for multi-camera coordination
        self.current_people_count = len(tracked_people)
        
        # Analyze entry zones
        entry_analysis = self._analyze_entry_zones(tracked_people, timestamp)
        
        # Check for tailgating
        tailgating_events = self._detect_tailgating(tracked_people, timestamp)
        
        # Check for intrusion
        intrusion_events = self._detect_intrusion(tracked_people, timestamp)
        
        # Check occupancy limits
        occupancy_events = self._check_occupancy_limits(tracked_people, timestamp)
        
        # Update zone occupancy tracking
        self._update_zone_occupancy(tracked_people)
        
        # Annotate frame with detections and zones
        annotated_frame = self._annotate_frame(frame, tracked_people, entry_analysis)
        
        # Handle all detected events
        all_events = tailgating_events + intrusion_events + occupancy_events
        for event in all_events:
            self._handle_entry_event(event, annotated_frame, timestamp)
        
        # Prepare processing results
        processing_results = {
            'people_count': len(tracked_people),
            'entry_analysis': entry_analysis,
            'events': all_events,
            'zone_occupancy': self.zone_occupancy.copy(),
            'entry_stats': self.entry_stats.copy()
        }
        
        # Update statistics
        self.stats['people_detected'] += len(tracked_people)
        if all_events:
            self.stats['events_detected'] += len(all_events)
        
        return annotated_frame, processing_results
    
    def _analyze_entry_zones(self, tracked_people: List[Dict], timestamp: float) -> Dict:
        """
        Analyze people movement in entry zones
        
        Args:
            tracked_people: List of tracked person detections
            timestamp: Current timestamp
            
        Returns:
            Dictionary with entry zone analysis
        """
        entry_analysis = {
            'active_entries': [],
            'zone_occupancy': {},
            'movement_patterns': []
        }
        
        for zone in self.entry_zones:
            zone_name = zone.get('name', 'Entry Zone')
            people_in_zone = []
            
            # Find people in this entry zone
            for person in tracked_people:
                if self._is_point_in_zone(person['center'], zone):
                    people_in_zone.append(person)
                    
                    # Track entry timing
                    track_id = person['track_id']
                    if track_id not in self.entry_events:
                        self.entry_events[track_id] = {
                            'zone_name': zone_name,
                            'entry_time': timestamp,
                            'first_position': person['center']
                        }
                    
                    # Update position history for movement analysis
                    if track_id in self.tracked_objects:
                        positions = self.tracked_objects[track_id].get('positions', [])
                        if len(positions) >= 2:
                            # Analyze movement direction
                            movement = self._analyze_movement_direction(positions)
                            entry_analysis['movement_patterns'].append({
                                'track_id': track_id,
                                'zone': zone_name,
                                'direction': movement,
                                'position': person['center']
                            })
            
            entry_analysis['zone_occupancy'][zone_name] = len(people_in_zone)
            
            if people_in_zone:
                entry_analysis['active_entries'].append({
                    'zone_name': zone_name,
                    'people_count': len(people_in_zone),
                    'people': people_in_zone,
                    'timestamp': timestamp
                })
        
        return entry_analysis
    
    def _analyze_movement_direction(self, positions: List[Tuple[float, float]]) -> str:
        """
        Analyze movement direction from position history
        
        Args:
            positions: List of (x, y) positions
            
        Returns:
            Movement direction string
        """
        if len(positions) < 2:
            return 'stationary'
        
        # Calculate overall movement vector
        start_pos = positions[0]
        end_pos = positions[-1]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        # Determine primary direction
        if abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'down' if dy > 0 else 'up'
    
    def _detect_tailgating(self, tracked_people: List[Dict], timestamp: float) -> List[Dict]:
        """
        Detect tailgating events (multiple people entering simultaneously)
        
        Args:
            tracked_people: List of tracked person detections
            timestamp: Current timestamp
            
        Returns:
            List of tailgating event dictionaries
        """
        tailgating_events = []
        
        for zone in self.entry_zones:
            zone_name = zone.get('name', 'Entry Zone')
            people_in_zone = []
            
            # Find people currently in this entry zone
            for person in tracked_people:
                if self._is_point_in_zone(person['center'], zone):
                    people_in_zone.append(person)
            
            # Check for tailgating (multiple people in entry zone)
            if len(people_in_zone) > self.max_people_per_entry:
                
                # Verify these are simultaneous entries (not just people passing through)
                simultaneous_entries = self._verify_simultaneous_entry(people_in_zone, timestamp)
                
                if len(simultaneous_entries) > self.max_people_per_entry:
                    
                    # Check if this is a new tailgating event (cooldown)
                    event_key = f"tailgating_{zone_name}"
                    if self._should_trigger_event(event_key, timestamp):
                        
                        tailgating_event = {
                            'type': DatacenterEventTypes.TAILGATING,
                            'severity': 'high',
                            'zone_name': zone_name,
                            'zone_type': 'entry_zone',
                            'people_count': len(simultaneous_entries),
                            'people': simultaneous_entries,
                            'timestamp': timestamp,
                            'details': {
                                'max_allowed': self.max_people_per_entry,
                                'detection_method': 'simultaneous_zone_occupancy',
                                'time_window': self.entry_time_window
                            }
                        }
                        
                        tailgating_events.append(tailgating_event)
                        self.entry_stats['tailgating_events'] += 1
                        
                        self.logger.warning(f"Tailgating detected in {zone_name}: {len(simultaneous_entries)} people")
        
        return tailgating_events
    
    def _verify_simultaneous_entry(self, people_in_zone: List[Dict], timestamp: float) -> List[Dict]:
        """
        Verify that people entered the zone simultaneously (within time window)
        
        Args:
            people_in_zone: People currently in the zone
            timestamp: Current timestamp
            
        Returns:
            List of people who entered simultaneously
        """
        simultaneous_entries = []
        
        for person in people_in_zone:
            track_id = person['track_id']
            
            # Check when this person first entered the zone
            if track_id in self.entry_events:
                entry_time = self.entry_events[track_id]['entry_time']
                time_in_zone = timestamp - entry_time
                
                # Consider as simultaneous if entered within the time window
                if time_in_zone <= self.entry_time_window:
                    simultaneous_entries.append(person)
        
        return simultaneous_entries
    
    def _detect_intrusion(self, tracked_people: List[Dict], timestamp: float) -> List[Dict]:
        """
        Detect intrusion attempts (unauthorized access to restricted areas)
        
        Args:
            tracked_people: List of tracked person detections
            timestamp: Current timestamp
            
        Returns:
            List of intrusion event dictionaries
        """
        intrusion_events = []
        
        # Check restricted zones for unauthorized access
        for zone in self.restricted_zones:
            zone_name = zone.get('name', 'Restricted Zone')
            security_level = zone.get('security_level', 'high_security')
            
            for person in tracked_people:
                if self._is_point_in_zone(person['center'], zone):
                    
                    # In a real system, check against access control database
                    # For now, consider any entry to restricted zone as potential intrusion
                    is_authorized = self._check_authorization_simulation(person, zone, timestamp)
                    
                    if not is_authorized:
                        event_key = f"intrusion_{zone_name}_{person['track_id']}"
                        
                        if self._should_trigger_event(event_key, timestamp):
                            
                            intrusion_event = {
                                'type': DatacenterEventTypes.INTRUSION,
                                'severity': 'critical' if security_level == 'critical' else 'high',
                                'zone_name': zone_name,
                                'zone_type': 'restricted_zone',
                                'security_level': security_level,
                                'person': person,
                                'timestamp': timestamp,
                                'details': {
                                    'authorization_checked': True,
                                    'badge_scan_detected': False,
                                    'detection_confidence': person.get('confidence', 0.8)
                                }
                            }
                            
                            intrusion_events.append(intrusion_event)
                            self.entry_stats['intrusion_attempts'] += 1
                            
                            self.logger.error(f"Intrusion detected in {zone_name}: Person {person['track_id']}")
        
        return intrusion_events
    
    def _check_authorization_simulation(self, person: Dict, zone: Dict, timestamp: float) -> bool:
        """
        Simulate authorization check (in real system, integrate with access control)
        
        Args:
            person: Person detection dictionary
            zone: Zone configuration
            timestamp: Current timestamp
            
        Returns:
            True if access is authorized
        """
        
        # Simulation: Check if person was in entry zone recently (badge scan simulation)
        track_id = person['track_id']
        
        if track_id in self.entry_events:
            entry_time = self.entry_events[track_id]['entry_time']
            time_since_entry = timestamp - entry_time
            
            # Simulate badge scan validity window
            if time_since_entry <= self.badge_scan_window:
                return True
        
        # For demo purposes, randomly authorize some entries to reduce false alarms
        # In production, this would integrate with actual access control systems
        import random
        return random.random() < 0.3  # 30% authorization rate for demo
    
    def _check_occupancy_limits(self, tracked_people: List[Dict], timestamp: float) -> List[Dict]:
        """
        Check for occupancy limit violations in entry zones
        
        Args:
            tracked_people: List of tracked person detections
            timestamp: Current timestamp
            
        Returns:
            List of occupancy violation events
        """
        occupancy_events = []
        
        for zone in self.entry_zones:
            zone_name = zone.get('name', 'Entry Zone')
            occupancy_limit = zone.get('occupancy_limit', DatacenterConfig.get_occupancy_limit('entry_zone'))
            
            people_in_zone = []
            for person in tracked_people:
                if self._is_point_in_zone(person['center'], zone):
                    people_in_zone.append(person)
            
            if len(people_in_zone) > occupancy_limit:
                event_key = f"occupancy_{zone_name}"
                
                if self._should_trigger_event(event_key, timestamp):
                    
                    occupancy_event = {
                        'type': DatacenterEventTypes.PEOPLE_COUNTING,
                        'severity': 'medium',
                        'zone_name': zone_name,
                        'zone_type': 'entry_zone',
                        'people_count': len(people_in_zone),
                        'occupancy_limit': occupancy_limit,
                        'people': people_in_zone,
                        'timestamp': timestamp,
                        'details': {
                            'violation_type': 'occupancy_exceeded',
                            'excess_count': len(people_in_zone) - occupancy_limit
                        }
                    }
                    
                    occupancy_events.append(occupancy_event)
                    self.entry_stats['occupancy_violations'] += 1
                    
                    self.logger.warning(f"Occupancy limit exceeded in {zone_name}: {len(people_in_zone)}/{occupancy_limit}")
        
        return occupancy_events
    
    def _update_zone_occupancy(self, tracked_people: List[Dict]):
        """Update current zone occupancy tracking"""
        
        # Reset occupancy counters
        for zone in self.entry_zones + self.restricted_zones:
            zone_name = zone.get('name', 'Zone')
            self.zone_occupancy[zone_name] = 0
        
        # Count people in each zone
        for person in tracked_people:
            for zone in self.entry_zones + self.restricted_zones:
                zone_name = zone.get('name', 'Zone')
                if self._is_point_in_zone(person['center'], zone):
                    self.zone_occupancy[zone_name] += 1
    
    def _should_trigger_event(self, event_key: str, timestamp: float) -> bool:
        """
        Check if enough time has passed since last event to trigger a new one
        
        Args:
            event_key: Unique key for this event type
            timestamp: Current timestamp
            
        Returns:
            True if event should be triggered
        """
        if event_key in self.recent_events:
            time_since_last = timestamp - self.recent_events[event_key]
            if time_since_last < self.event_cooldown:
                return False
        
        self.recent_events[event_key] = timestamp
        return True
    
    def _handle_entry_event(self, event: Dict, frame: np.ndarray, timestamp: float):
        """
        Handle detected entry events (save, alert, log)
        
        Args:
            event: Event dictionary
            frame: Current video frame
            timestamp: Event timestamp
        """
        try:
            event_type = event['type']
            severity = event['severity']
            zone_name = event.get('zone_name', 'Unknown Zone')
            
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
            
            # Save event media based on severity
            if severity in ['high', 'critical']:
                if self.db_writer:
                    self._save_event_media(event_type, event, frame, timestamp, zone_name, event_id)
                
                # Send immediate alert for critical events
                if severity == 'critical' and self.sms_enabled:
                    self._send_security_alert(
                        event_type=event_type.replace('_', ' ').title(),
                        details=f"{event.get('people_count', 1)} people in {zone_name}",
                        timestamp=timestamp
                    )
            
            self.logger.info(f"Entry event handled: {event_type} in {zone_name} (severity: {severity})")
            
        except Exception as e:
            self.logger.error(f"Error handling entry event: {e}", exc_info=True)
    
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
            self.logger.error(f"Error saving event media: {e}")
    
    def _annotate_frame(self, frame: np.ndarray, tracked_people: List[Dict], 
                       entry_analysis: Dict) -> np.ndarray:
        """
        Annotate frame with detections, zones, and entry analysis
        
        Args:
            frame: Original video frame
            tracked_people: List of tracked person detections
            entry_analysis: Entry zone analysis results
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw entry zones
        for zone in self.entry_zones:
            self._draw_zone(annotated_frame, zone, self.zone_colors.get('entry_zone', (255, 255, 0)), 
                          alpha=0.2, label=zone.get('name', 'Entry Zone'))
        
        # Draw restricted zones
        for zone in self.restricted_zones:
            self._draw_zone(annotated_frame, zone, self.zone_colors.get('restricted_zone', (0, 0, 255)), 
                          alpha=0.3, label=zone.get('name', 'Restricted Zone'))
        
        # Draw people detections
        for person in tracked_people:
            bbox = person['bbox']
            track_id = person['track_id']
            confidence = person.get('confidence', 0.0)
            
            # Determine color based on zone
            color = (0, 255, 0)  # Green default
            label = f"ID:{track_id}"
            
            # Check if person is in restricted zone
            for zone in self.restricted_zones:
                if self._is_point_in_zone(person['center'], zone):
                    color = (0, 0, 255)  # Red for restricted
                    label += " (RESTRICTED)"
                    break
            
            # Check if person is in entry zone
            for zone in self.entry_zones:
                if self._is_point_in_zone(person['center'], zone):
                    if color == (0, 255, 0):  # If not already red
                        color = (255, 255, 0)  # Yellow for entry
                        label += " (ENTRY)"
                    break
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
            
            # Draw label
            cv2.putText(annotated_frame, label,
                       (int(bbox[0]), int(bbox[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw zone occupancy information
        y_offset = 30
        for zone_name, count in entry_analysis['zone_occupancy'].items():
            occupancy_text = f"{zone_name}: {count} people"
            cv2.putText(annotated_frame, occupancy_text,
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        # Draw statistics
        stats_text = f"Total: {len(tracked_people)} | Entries: {self.entry_stats['total_entries']} | Events: {self.stats['events_detected']}"
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
            self.logger.error(f"Error drawing zone: {e}")

# Export the monitor class
__all__ = ['DatacenterEntryMonitor']