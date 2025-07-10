#!/usr/bin/env python3
"""
Script 14: corridor_monitor.py
File Path: src/camera_models/corridor_monitor.py

Datacenter Monitoring System - Corridor Monitoring

This module implements Phase 1 use cases for corridors:
1. Loitering Detection (a2 - scalable, medium effort)
2. People Counting (a2 - scalable, medium effort) 
3. Flow Analysis (movement patterns)
4. Emergency Egress Path Monitoring

Corridor monitoring focuses on:
- Detecting people staying stationary too long (loitering)
- Monitoring corridor occupancy levels
- Analyzing movement patterns and flow
- Ensuring emergency exits remain clear
- Detecting unusual behavior in corridors
"""

import cv2
import numpy as np
import time
import uuid
import math
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict, deque

from .camera_model_base import DatacenterCameraModelBase
from config import DatacenterConfig, DatacenterEventTypes
from logger import audit_logger, performance_logger

class CorridorMonitor(DatacenterCameraModelBase):
    """
    Monitor for datacenter corridors implementing Phase 1 use cases:
    - Loitering detection (people staying stationary beyond threshold)
    - People counting (corridor occupancy monitoring)
    - Flow analysis (movement patterns and congestion)
    - Emergency egress monitoring (exit path clearance)
    """

    def __init__(self, camera_id: int, datacenter_id: int, zones: Optional[Dict] = None, 
                 rules: Optional[List] = None, settings: Optional[Dict] = None, 
                 db=None, db_writer=None, frames_base_dir: str = 'frames'):
        """
        Initialize the corridor monitor
        
        Args:
            camera_id: Camera identifier
            datacenter_id: Datacenter identifier
            zones: Zone definitions (common_zones, emergency_exits)
            rules: Monitoring rules for corridor events
            settings: Camera-specific settings
            db: Database instance
            db_writer: Database writer
            frames_base_dir: Base directory for frame storage
        """
        
        super().__init__(camera_id, datacenter_id, zones, rules, settings, 
                        db, db_writer, frames_base_dir)
        
        self.logger.info("Initializing CorridorMonitor")
        
        # Loitering detection configuration
        self.loitering_threshold = settings.get('loitering_threshold', DatacenterConfig.LOITERING_THRESHOLD)
        self.movement_threshold = settings.get('movement_threshold', DatacenterConfig.MOVEMENT_THRESHOLD)
        self.loitering_check_interval = settings.get('loitering_check_interval', DatacenterConfig.LOITERING_CHECK_INTERVAL)
        
        # Get zone configurations
        self.common_zones = self._get_zones_by_type('common_zone')
        self.emergency_exit_zones = self._get_zones_by_type('emergency_exit')
        self.corridor_zones = self._get_zones_by_type('corridor_zone')
        
        # Loitering tracking state
        self.stationary_people = {}  # track_id -> {'start_time', 'positions', 'zone', 'alerted'}
        self.last_loitering_check = 0
        
        # Movement analysis
        self.movement_history = defaultdict(lambda: deque(maxlen=20))  # track_id -> position history
        self.flow_zones = {}  # zone_name -> flow statistics
        
        # Occupancy monitoring
        self.zone_occupancy = {}
        self.occupancy_history = defaultdict(lambda: deque(maxlen=100))  # zone -> historical occupancy
        self.peak_occupancy_times = {}  # zone -> peak times
        
        # Emergency egress monitoring
        self.exit_blocked_alerts = {}
        self.exit_clearance_required = 2.0  # meters clearance required around exits
        
        # Flow analysis parameters
        self.congestion_threshold = 0.7  # density threshold for congestion detection
        self.flow_analysis_enabled = True
        self.direction_analysis_enabled = True
        
        # Statistics for corridor monitoring
        self.corridor_stats = {
            'total_people_passed': 0,
            'loitering_events': 0,
            'congestion_events': 0,
            'exit_blocked_events': 0,
            'average_occupancy': 0,
            'peak_occupancy': 0,
            'movement_patterns': {
                'north_south': 0,
                'south_north': 0,
                'east_west': 0,
                'west_east': 0,
                'stationary': 0
            }
        }
        
        self.logger.info(f"Corridor monitor initialized - Loitering threshold: {self.loitering_threshold}s")
        self.logger.info(f"Monitoring {len(self.common_zones)} common zones, {len(self.emergency_exit_zones)} emergency exits")
    
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
        Process frame for corridor monitoring
        
        Args:
            frame: Video frame
            timestamp: Frame timestamp
            detection_result: Person detection results from YOLO
            ppe_result: PPE detection results (not used for corridors)
            
        Returns:
            Tuple of (annotated_frame, processing_results)
        """
        
        # Extract people detections
        people_detections, _ = self.detect_people_and_ppe(frame, detection_result, ppe_result)
        
        # Update object tracking
        tracked_people = self.update_object_tracking(people_detections)
        
        # Store current people count for multi-camera coordination
        self.current_people_count = len(tracked_people)
        
        # Update movement history for flow analysis
        self._update_movement_history(tracked_people, timestamp)
        
        # Perform loitering detection
        loitering_events = self._detect_loitering(tracked_people, timestamp)
        
        # Check occupancy levels
        occupancy_events = self._check_occupancy_levels(tracked_people, timestamp)
        
        # Analyze movement flows and patterns
        flow_analysis = self._analyze_movement_flows(tracked_people, timestamp)
        
        # Check emergency exit clearance
        exit_events = self._check_emergency_exits(tracked_people, timestamp)
        
        # Detect congestion
        congestion_events = self._detect_congestion(tracked_people, timestamp)
        
        # Update zone occupancy tracking
        self._update_zone_occupancy(tracked_people, timestamp)
        
        # Annotate frame with detections and analysis
        annotated_frame = self._annotate_frame(frame, tracked_people, flow_analysis)
        
        # Handle all detected events
        all_events = loitering_events + occupancy_events + exit_events + congestion_events
        for event in all_events:
            self._handle_corridor_event(event, annotated_frame, timestamp)
        
        # Prepare processing results
        processing_results = {
            'people_count': len(tracked_people),
            'loitering_count': len([p for p in self.stationary_people.values() if not p.get('alerted', False)]),
            'flow_analysis': flow_analysis,
            'zone_occupancy': self.zone_occupancy.copy(),
            'events': all_events,
            'corridor_stats': self.corridor_stats.copy()
        }
        
        # Update statistics
        self.stats['people_detected'] += len(tracked_people)
        if all_events:
            self.stats['events_detected'] += len(all_events)
        
        return annotated_frame, processing_results
    
    def _update_movement_history(self, tracked_people: List[Dict], timestamp: float):
        """Update movement history for all tracked people"""
        
        current_tracks = set()
        
        for person in tracked_people:
            track_id = person['track_id']
            current_tracks.add(track_id)
            
            # Add current position to history
            position = person['center']
            self.movement_history[track_id].append({
                'position': position,
                'timestamp': timestamp
            })
        
        # Clean up movement history for tracks that are no longer active
        tracks_to_remove = []
        for track_id in self.movement_history:
            if track_id not in current_tracks:
                # Keep history for a while in case track comes back
                if len(self.movement_history[track_id]) > 0:
                    last_time = self.movement_history[track_id][-1]['timestamp']
                    if timestamp - last_time > 30:  # 30 seconds
                        tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.movement_history[track_id]
    
    def _detect_loitering(self, tracked_people: List[Dict], timestamp: float) -> List[Dict]:
        """
        Detect people who have been stationary for too long (loitering)
        
        Args:
            tracked_people: List of tracked person detections
            timestamp: Current timestamp
            
        Returns:
            List of loitering event dictionaries
        """
        loitering_events = []
        
        # Only check loitering at intervals to reduce computation
        if timestamp - self.last_loitering_check < self.loitering_check_interval:
            return loitering_events
        
        self.last_loitering_check = timestamp
        
        current_tracks = {person['track_id'] for person in tracked_people}
        
        # Update stationary people tracking
        for person in tracked_people:
            track_id = person['track_id']
            current_position = person['center']
            
            # Initialize tracking for new people
            if track_id not in self.stationary_people:
                self.stationary_people[track_id] = {
                    'start_time': timestamp,
                    'initial_position': current_position,
                    'positions': [current_position],
                    'zone': self._get_person_zone(person),
                    'alerted': False,
                    'total_movement': 0.0
                }
            else:
                # Update existing tracking
                stationary_info = self.stationary_people[track_id]
                stationary_info['positions'].append(current_position)
                
                # Calculate total movement
                if len(stationary_info['positions']) > 1:
                    last_pos = stationary_info['positions'][-2]
                    movement = math.sqrt(
                        (current_position[0] - last_pos[0])**2 + 
                        (current_position[1] - last_pos[1])**2
                    )
                    stationary_info['total_movement'] += movement
                
                # Keep only recent positions
                if len(stationary_info['positions']) > 50:
                    stationary_info['positions'] = stationary_info['positions'][-25:]
        
        # Check for loitering violations
        for track_id, stationary_info in self.stationary_people.items():
            if track_id not in current_tracks:
                continue
                
            time_stationary = timestamp - stationary_info['start_time']
            total_movement = stationary_info['total_movement']
            
            # Check if person has been stationary long enough and hasn't moved much
            if (time_stationary >= self.loitering_threshold and 
                total_movement < self.movement_threshold * time_stationary and
                not stationary_info['alerted']):
                
                # Find the person object for this track
                person = next((p for p in tracked_people if p['track_id'] == track_id), None)
                if person:
                    
                    zone_name = stationary_info['zone']
                    
                    loitering_event = {
                        'type': DatacenterEventTypes.LOITERING,
                        'severity': 'medium',
                        'zone_name': zone_name,
                        'zone_type': 'common_zone',
                        'person': person,
                        'stationary_time': time_stationary,
                        'total_movement': total_movement,
                        'timestamp': timestamp,
                        'details': {
                            'loitering_threshold': self.loitering_threshold,
                            'movement_threshold': self.movement_threshold,
                            'detection_method': 'movement_analysis',
                            'positions_tracked': len(stationary_info['positions'])
                        }
                    }
                    
                    loitering_events.append(loitering_event)
                    stationary_info['alerted'] = True
                    self.corridor_stats['loitering_events'] += 1
                    
                    self.logger.warning(f"Loitering detected: Person {track_id} stationary for {time_stationary:.1f}s in {zone_name}")
        
        # Clean up tracking for people who are no longer present
        tracks_to_remove = []
        for track_id in self.stationary_people:
            if track_id not in current_tracks:
                # Remove if not seen for a while
                if timestamp - self.stationary_people[track_id]['start_time'] > 120:  # 2 minutes
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.stationary_people[track_id]
        
        return loitering_events
    
    def _get_person_zone(self, person: Dict) -> str:
        """Determine which zone a person is currently in"""
        
        position = person['center']
        
        # Check common zones first
        for zone in self.common_zones:
            if self._is_point_in_zone(position, zone):
                return zone.get('name', 'Common Area')
        
        # Check corridor zones
        for zone in self.corridor_zones:
            if self._is_point_in_zone(position, zone):
                return zone.get('name', 'Corridor')
        
        return 'Unknown Zone'
    
    def _check_occupancy_levels(self, tracked_people: List[Dict], timestamp: float) -> List[Dict]:
        """Check for occupancy violations in corridor zones"""
        
        occupancy_events = []
        
        all_zones = self.common_zones + self.corridor_zones
        
        for zone in all_zones:
            zone_name = zone.get('name', 'Zone')
            occupancy_limit = zone.get('occupancy_limit', DatacenterConfig.get_occupancy_limit('corridor'))
            
            people_in_zone = []
            for person in tracked_people:
                if self._is_point_in_zone(person['center'], zone):
                    people_in_zone.append(person)
            
            if len(people_in_zone) > occupancy_limit:
                event_key = f"occupancy_{zone_name}"
                
                if self._should_trigger_event(event_key, timestamp):
                    
                    occupancy_event = {
                        'type': DatacenterEventTypes.PEOPLE_COUNTING,
                        'severity': 'low',
                        'zone_name': zone_name,
                        'zone_type': 'corridor',
                        'people_count': len(people_in_zone),
                        'occupancy_limit': occupancy_limit,
                        'people': people_in_zone,
                        'timestamp': timestamp,
                        'details': {
                            'violation_type': 'occupancy_exceeded',
                            'excess_count': len(people_in_zone) - occupancy_limit,
                            'occupancy_ratio': len(people_in_zone) / occupancy_limit
                        }
                    }
                    
                    occupancy_events.append(occupancy_event)
                    
                    self.logger.info(f"Occupancy exceeded in {zone_name}: {len(people_in_zone)}/{occupancy_limit}")
        
        return occupancy_events
    
    def _analyze_movement_flows(self, tracked_people: List[Dict], timestamp: float) -> Dict:
        """
        Analyze movement patterns and flows in the corridor
        
        Args:
            tracked_people: List of tracked person detections
            timestamp: Current timestamp
            
        Returns:
            Dictionary with flow analysis results
        """
        
        flow_analysis = {
            'movement_vectors': [],
            'flow_directions': defaultdict(int),
            'average_speed': 0,
            'congestion_areas': [],
            'flow_efficiency': 1.0
        }
        
        if not self.flow_analysis_enabled:
            return flow_analysis
        
        total_speed = 0
        speed_count = 0
        
        for person in tracked_people:
            track_id = person['track_id']
            
            if track_id in self.movement_history and len(self.movement_history[track_id]) >= 2:
                history = list(self.movement_history[track_id])
                
                # Calculate movement vector and speed
                if len(history) >= 2:
                    recent_positions = history[-5:]  # Last 5 positions
                    if len(recent_positions) >= 2:
                        
                        start_pos = recent_positions[0]['position']
                        end_pos = recent_positions[-1]['position']
                        time_diff = recent_positions[-1]['timestamp'] - recent_positions[0]['timestamp']
                        
                        if time_diff > 0:
                            # Calculate movement vector
                            dx = end_pos[0] - start_pos[0]
                            dy = end_pos[1] - start_pos[1]
                            distance = math.sqrt(dx**2 + dy**2)
                            
                            if distance > 5:  # Minimum movement threshold
                                speed = distance / time_diff  # pixels per second
                                direction = self._calculate_movement_direction(dx, dy)
                                
                                flow_analysis['movement_vectors'].append({
                                    'track_id': track_id,
                                    'vector': (dx, dy),
                                    'speed': speed,
                                    'direction': direction,
                                    'position': person['center']
                                })
                                
                                flow_analysis['flow_directions'][direction] += 1
                                total_speed += speed
                                speed_count += 1
                                
                                # Update global statistics
                                self.corridor_stats['movement_patterns'][direction] += 1
        
        # Calculate average speed
        if speed_count > 0:
            flow_analysis['average_speed'] = total_speed / speed_count
        
        # Detect congestion areas
        flow_analysis['congestion_areas'] = self._detect_congestion_areas(tracked_people)
        
        # Calculate flow efficiency (higher is better)
        if len(tracked_people) > 0:
            moving_people = len([v for v in flow_analysis['movement_vectors'] if v['speed'] > 10])
            flow_analysis['flow_efficiency'] = moving_people / len(tracked_people)
        
        return flow_analysis
    
    def _calculate_movement_direction(self, dx: float, dy: float) -> str:
        """Calculate primary movement direction from displacement vector"""
        
        # Determine primary direction
        if abs(dx) > abs(dy):
            if dx > 0:
                return 'east_west'  # Moving right
            else:
                return 'west_east'  # Moving left
        else:
            if dy > 0:
                return 'north_south'  # Moving down
            else:
                return 'south_north'  # Moving up
    
    def _detect_congestion_areas(self, tracked_people: List[Dict]) -> List[Dict]:
        """Detect areas of high density that might indicate congestion"""
        
        congestion_areas = []
        
        if len(tracked_people) < 3:
            return congestion_areas
        
        # Use simple clustering to find high-density areas
        positions = [person['center'] for person in tracked_people]
        
        # Grid-based density analysis
        grid_size = 100  # pixels
        density_grid = defaultdict(list)
        
        for i, pos in enumerate(positions):
            grid_x = int(pos[0] // grid_size)
            grid_y = int(pos[1] // grid_size)
            density_grid[(grid_x, grid_y)].append(i)
        
        # Find high-density grid cells
        for (grid_x, grid_y), person_indices in density_grid.items():
            if len(person_indices) >= 3:  # 3+ people in same grid cell
                
                # Calculate center of this congestion area
                center_x = sum(positions[i][0] for i in person_indices) / len(person_indices)
                center_y = sum(positions[i][1] for i in person_indices) / len(person_indices)
                
                congestion_areas.append({
                    'center': (center_x, center_y),
                    'people_count': len(person_indices),
                    'grid_position': (grid_x, grid_y),
                    'people_indices': person_indices
                })
        
        return congestion_areas
    
    def _check_emergency_exits(self, tracked_people: List[Dict], timestamp: float) -> List[Dict]:
        """Check if emergency exits are blocked or obstructed"""
        
        exit_events = []
        
        for exit_zone in self.emergency_exit_zones:
            exit_name = exit_zone.get('name', 'Emergency Exit')
            
            # Check if people are too close to exit
            people_near_exit = []
            for person in tracked_people:
                distance_to_exit = self._calculate_distance_to_zone(person['center'], exit_zone)
                if distance_to_exit < self.exit_clearance_required * 50:  # Convert to pixels approximately
                    people_near_exit.append(person)
            
            if len(people_near_exit) > 0:
                event_key = f"exit_blocked_{exit_name}"
                
                if self._should_trigger_event(event_key, timestamp):
                    
                    exit_event = {
                        'type': 'exit_obstruction',
                        'severity': 'medium',
                        'zone_name': exit_name,
                        'zone_type': 'emergency_exit',
                        'people_count': len(people_near_exit),
                        'people': people_near_exit,
                        'timestamp': timestamp,
                        'details': {
                            'clearance_required': self.exit_clearance_required,
                            'obstruction_type': 'people_too_close'
                        }
                    }
                    
                    exit_events.append(exit_event)
                    self.corridor_stats['exit_blocked_events'] += 1
                    
                    self.logger.warning(f"Emergency exit {exit_name} partially obstructed by {len(people_near_exit)} people")
        
        return exit_events
    
    def _calculate_distance_to_zone(self, point: Tuple[float, float], zone: Dict) -> float:
        """Calculate minimum distance from point to zone boundary"""
        
        coordinates = zone.get('coordinates', [])
        if len(coordinates) < 3:
            return float('inf')
        
        # Simple distance calculation to zone center
        zone_center_x = sum(coord[0] for coord in coordinates) / len(coordinates)
        zone_center_y = sum(coord[1] for coord in coordinates) / len(coordinates)
        
        return math.sqrt((point[0] - zone_center_x)**2 + (point[1] - zone_center_y)**2)
    
    def _detect_congestion(self, tracked_people: List[Dict], timestamp: float) -> List[Dict]:
        """Detect corridor congestion events"""
        
        congestion_events = []
        
        # Simple congestion detection based on density
        if len(tracked_people) > 8:  # Threshold for congestion
            
            # Calculate movement efficiency
            moving_people = 0
            for track_id in [p['track_id'] for p in tracked_people]:
                if track_id in self.movement_history and len(self.movement_history[track_id]) >= 2:
                    recent_history = list(self.movement_history[track_id])[-3:]
                    if len(recent_history) >= 2:
                        total_movement = 0
                        for i in range(1, len(recent_history)):
                            pos1 = recent_history[i-1]['position']
                            pos2 = recent_history[i]['position']
                            movement = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                            total_movement += movement
                        
                        if total_movement > 20:  # Significant movement
                            moving_people += 1
            
            movement_efficiency = moving_people / len(tracked_people) if len(tracked_people) > 0 else 1
            
            if movement_efficiency < 0.3:  # Less than 30% are moving
                event_key = "congestion_corridor"
                
                if self._should_trigger_event(event_key, timestamp):
                    
                    congestion_event = {
                        'type': 'congestion',
                        'severity': 'low',
                        'zone_name': 'Corridor',
                        'zone_type': 'corridor',
                        'people_count': len(tracked_people),
                        'movement_efficiency': movement_efficiency,
                        'timestamp': timestamp,
                        'details': {
                            'moving_people': moving_people,
                            'stationary_people': len(tracked_people) - moving_people,
                            'congestion_threshold': self.congestion_threshold
                        }
                    }
                    
                    congestion_events.append(congestion_event)
                    self.corridor_stats['congestion_events'] += 1
                    
                    self.logger.info(f"Corridor congestion detected: {len(tracked_people)} people, {movement_efficiency:.2f} efficiency")
        
        return congestion_events
    
    def _update_zone_occupancy(self, tracked_people: List[Dict], timestamp: float):
        """Update zone occupancy tracking and statistics"""
        
        # Reset occupancy counters
        all_zones = self.common_zones + self.corridor_zones + self.emergency_exit_zones
        for zone in all_zones:
            zone_name = zone.get('name', 'Zone')
            self.zone_occupancy[zone_name] = 0
        
        # Count people in each zone
        for person in tracked_people:
            for zone in all_zones:
                zone_name = zone.get('name', 'Zone')
                if self._is_point_in_zone(person['center'], zone):
                    self.zone_occupancy[zone_name] += 1
        
        # Update occupancy history and statistics
        total_occupancy = sum(self.zone_occupancy.values())
        
        for zone_name, count in self.zone_occupancy.items():
            self.occupancy_history[zone_name].append({
                'count': count,
                'timestamp': timestamp
            })
            
            # Update peak occupancy
            if zone_name not in self.peak_occupancy_times or count > self.peak_occupancy_times[zone_name]['count']:
                self.peak_occupancy_times[zone_name] = {
                    'count': count,
                    'timestamp': timestamp
                }
        
        # Update global statistics
        if total_occupancy > self.corridor_stats['peak_occupancy']:
            self.corridor_stats['peak_occupancy'] = total_occupancy
        
        # Calculate average occupancy (simple moving average)
        current_avg = self.corridor_stats['average_occupancy']
        self.corridor_stats['average_occupancy'] = (current_avg * 0.9) + (total_occupancy * 0.1)
    
    def _should_trigger_event(self, event_key: str, timestamp: float) -> bool:
        """Check if enough time has passed since last event to trigger a new one"""
        if not hasattr(self, 'recent_events'):
            self.recent_events = {}
            
        if event_key in self.recent_events:
            time_since_last = timestamp - self.recent_events[event_key]
            if time_since_last < self.event_cooldown:
                return False
        
        self.recent_events[event_key] = timestamp
        return True
    
    def _handle_corridor_event(self, event: Dict, frame: np.ndarray, timestamp: float):
        """Handle detected corridor events (save, alert, log)"""
        try:
            event_type = event['type']
            severity = event['severity']
            zone_name = event.get('zone_name', 'Unknown Zone')
            
            # Log event for audit
            audit_logger.log_event_detection(
                event_type=event_type,
                camera_id=str(self.camera_id),
                datacenter_id=str(self.datacenter_id),
                severity=severity,
                detection_data=event.get('details', {})
            )
            
            # Generate unique event ID
            event_id = str(uuid.uuid4())
            
            # Save event media for medium+ severity events
            if severity in ['medium', 'high', 'critical']:
                if self.db_writer:
                    self._save_event_media(event_type, event, frame, timestamp, zone_name, event_id)
            
            self.logger.info(f"Corridor event handled: {event_type} in {zone_name} (severity: {severity})")
            
        except Exception as e:
            self.logger.error(f"Error handling corridor event: {e}", exc_info=True)
    
    def _save_event_media(self, event_type: str, event_data: Dict, frame: np.ndarray, 
                         timestamp: float, zone_name: str, event_id: str):
        """Save event media and queue for database storage"""
        try:
            if self.media_preference == "image":
                frame_path = self._save_frame_with_detections(event_id, frame, timestamp)
                if frame_path:
                    self.stats['frames_saved'] += 1
            else:
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
                       flow_analysis: Dict) -> np.ndarray:
        """
        Annotate frame with detections, zones, and corridor analysis
        
        Args:
            frame: Original video frame
            tracked_people: List of tracked person detections
            flow_analysis: Movement flow analysis results
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw common zones
        for zone in self.common_zones:
            self._draw_zone(annotated_frame, zone, self.zone_colors.get('common_zone', (0, 255, 0)), 
                          alpha=0.2, label=zone.get('name', 'Common Zone'))
        
        # Draw corridor zones
        for zone in self.corridor_zones:
            self._draw_zone(annotated_frame, zone, (128, 128, 255), 
                          alpha=0.15, label=zone.get('name', 'Corridor'))
        
        # Draw emergency exit zones
        for zone in self.emergency_exit_zones:
            self._draw_zone(annotated_frame, zone, (255, 0, 255), 
                          alpha=0.3, label=f"EXIT: {zone.get('name', 'Emergency Exit')}")
        
        # Draw people detections with loitering status
        for person in tracked_people:
            bbox = person['bbox']
            track_id = person['track_id']
            confidence = person.get('confidence', 0.0)
            
            # Determine color based on status
            color = (0, 255, 0)  # Green default
            label = f"ID:{track_id}"
            
            # Check if person is loitering
            if track_id in self.stationary_people:
                stationary_info = self.stationary_people[track_id]
                time_stationary = time.time() - stationary_info['start_time']
                
                if time_stationary > self.loitering_threshold / 2:  # Warning threshold
                    color = (0, 255, 255)  # Yellow for potential loitering
                    label += f" ({time_stationary:.0f}s)"
                
                if stationary_info.get('alerted', False):
                    color = (0, 0, 255)  # Red for confirmed loitering
                    label += " LOITERING"
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
            
            # Draw label
            cv2.putText(annotated_frame, label,
                       (int(bbox[0]), int(bbox[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw movement vectors
        if self.direction_analysis_enabled:
            for vector_info in flow_analysis.get('movement_vectors', []):
                position = vector_info['position']
                vector = vector_info['vector']
                speed = vector_info['speed']
                
                # Scale vector for visualization
                scale = min(2.0, speed / 20.0)  # Scale based on speed
                end_point = (
                    int(position[0] + vector[0] * scale),
                    int(position[1] + vector[1] * scale)
                )
                
                # Draw movement arrow
                cv2.arrowedLine(annotated_frame, 
                              (int(position[0]), int(position[1])),
                              end_point,
                              (255, 255, 0), 2, tipLength=0.3)
        
        # Draw congestion areas
        for congestion_area in flow_analysis.get('congestion_areas', []):
            center = congestion_area['center']
            people_count = congestion_area['people_count']
            
            # Draw congestion indicator
            cv2.circle(annotated_frame, 
                      (int(center[0]), int(center[1])), 
                      30, (0, 0, 255), 3)
            cv2.putText(annotated_frame, f"CONGESTION: {people_count}",
                       (int(center[0]) - 50, int(center[1]) - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw zone occupancy information
        y_offset = 30
        for zone_name, count in self.zone_occupancy.items():
            occupancy_text = f"{zone_name}: {count} people"
            
            # Color code based on occupancy
            text_color = (255, 255, 255)  # White default
            if count > DatacenterConfig.get_occupancy_limit('corridor'):
                text_color = (0, 0, 255)  # Red for over capacity
            elif count > DatacenterConfig.get_occupancy_limit('corridor') * 0.8:
                text_color = (0, 255, 255)  # Yellow for near capacity
            
            cv2.putText(annotated_frame, occupancy_text,
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            y_offset += 25
        
        # Draw flow statistics
        flow_directions = flow_analysis.get('flow_directions', {})
        if flow_directions:
            y_offset += 10
            flow_text = "Flow: " + ", ".join([f"{direction}: {count}" for direction, count in flow_directions.items()])
            cv2.putText(annotated_frame, flow_text[:80],  # Truncate if too long
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y_offset += 20
        
        # Draw efficiency indicator
        flow_efficiency = flow_analysis.get('flow_efficiency', 1.0)
        efficiency_text = f"Flow Efficiency: {flow_efficiency:.2f}"
        efficiency_color = (0, 255, 0) if flow_efficiency > 0.7 else (0, 255, 255) if flow_efficiency > 0.4 else (0, 0, 255)
        cv2.putText(annotated_frame, efficiency_text,
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, efficiency_color, 1)
        
        # Draw overall statistics
        stats_text = (f"Total: {len(tracked_people)} | "
                     f"Loitering: {len([p for p in self.stationary_people.values() if not p.get('alerted', False)])} | "
                     f"Events: {self.stats['events_detected']}")
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
    
    def get_loitering_statistics(self) -> Dict:
        """Get detailed loitering statistics"""
        
        current_time = time.time()
        active_loiterers = []
        
        for track_id, stationary_info in self.stationary_people.items():
            time_stationary = current_time - stationary_info['start_time']
            
            if time_stationary >= self.loitering_threshold:
                active_loiterers.append({
                    'track_id': track_id,
                    'time_stationary': time_stationary,
                    'zone': stationary_info['zone'],
                    'alerted': stationary_info.get('alerted', False),
                    'total_movement': stationary_info['total_movement']
                })
        
        return {
            'active_loiterers': active_loiterers,
            'total_loitering_events': self.corridor_stats['loitering_events'],
            'loitering_threshold': self.loitering_threshold,
            'movement_threshold': self.movement_threshold
        }
    
    def get_flow_statistics(self) -> Dict:
        """Get detailed movement flow statistics"""
        
        return {
            'movement_patterns': self.corridor_stats['movement_patterns'].copy(),
            'congestion_events': self.corridor_stats['congestion_events'],
            'average_occupancy': self.corridor_stats['average_occupancy'],
            'peak_occupancy': self.corridor_stats['peak_occupancy'],
            'peak_occupancy_times': self.peak_occupancy_times.copy()
        }
    
    def reset_statistics(self):
        """Reset corridor monitoring statistics"""
        
        self.corridor_stats = {
            'total_people_passed': 0,
            'loitering_events': 0,
            'congestion_events': 0,
            'exit_blocked_events': 0,
            'average_occupancy': 0,
            'peak_occupancy': 0,
            'movement_patterns': {
                'north_south': 0,
                'south_north': 0,
                'east_west': 0,
                'west_east': 0,
                'stationary': 0
            }
        }
        
        self.stationary_people.clear()
        self.movement_history.clear()
        self.occupancy_history.clear()
        self.peak_occupancy_times.clear()
        
        self.logger.info("Corridor monitoring statistics reset")

# Export the monitor class
__all__ = ['CorridorMonitor']