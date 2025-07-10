#!/usr/bin/env python3
"""
Script 17: common_area_monitor.py
File Path: src/camera_models/common_area_monitor.py

Datacenter Monitoring System - Common Area Monitoring

This module implements Phase 1 use cases for common areas:
1. People Counting (a2 - scalable, medium effort)
2. Loitering Detection (a2 - scalable, medium effort)
3. Occupancy Monitoring (derived from people counting)
4. Flow Analysis (crowd management)

Common area monitoring focuses on:
- Reception area occupancy
- Waiting area management
- Visitor flow monitoring
- Long-term presence detection
- Crowd density analysis
- Emergency evacuation support
"""

import cv2
import numpy as np
import time
import uuid
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict
import os

from .camera_model_base import DatacenterCameraModelBase
from config import DatacenterConfig, DatacenterEventTypes
from logger import audit_logger, performance_logger

class CommonAreaMonitor(DatacenterCameraModelBase):
    """
    Monitor for datacenter common areas implementing Phase 1 use cases:
    - People counting and occupancy monitoring
    - Loitering detection for extended stays
    - Visitor flow analysis
    - Reception area management
    - Emergency evacuation support
    """

    def __init__(self, camera_id: int, datacenter_id: int, zones: Optional[Dict] = None, 
                 rules: Optional[List] = None, settings: Optional[Dict] = None, 
                 db=None, db_writer=None, frames_base_dir: str = 'frames'):
        """
        Initialize the common area monitor
        
        Args:
            camera_id: Camera identifier
            datacenter_id: Datacenter identifier
            zones: Zone definitions (common_zones, waiting_areas, reception_areas)
            rules: Monitoring rules for common area events
            settings: Camera-specific settings
            db: Database instance
            db_writer: Database writer
            frames_base_dir: Base directory for frame storage
        """
        
        super().__init__(camera_id, datacenter_id, zones, rules, settings, 
                        db, db_writer, frames_base_dir)
        
        self.logger.info("Initializing CommonAreaMonitor")
        
        # Common area specific configuration
        self.loitering_threshold = settings.get('loitering_threshold', DatacenterConfig.LOITERING_THRESHOLD)
        self.movement_threshold = settings.get('movement_threshold', DatacenterConfig.MOVEMENT_THRESHOLD)
        self.occupancy_warning_ratio = settings.get('occupancy_warning_ratio', 0.8)  # 80% of limit
        self.flow_analysis_enabled = settings.get('flow_analysis_enabled', True)
        
        # Get zone configurations
        self.common_zones = self._get_zones_by_type('common_zone')
        self.waiting_areas = self._get_zones_by_type('waiting_area')
        self.reception_areas = self._get_zones_by_type('reception_area')
        
        # Loitering detection state
        self.person_stationary_times = {}  # Track how long each person has been stationary
        self.person_last_positions = {}    # Track last known positions
        self.loitering_warnings = {}       # Track loitering warnings to avoid spam
        
        # Occupancy monitoring
        self.zone_occupancy_history = defaultdict(list)  # Historical occupancy data
        self.peak_occupancy_times = {}     # Track peak occupancy periods
        self.occupancy_warnings = {}       # Track occupancy warnings
        
        # Flow analysis
        self.entry_points = {}             # Track where people enter zones
        self.exit_points = {}              # Track where people exit zones
        self.flow_patterns = {}            # Store movement patterns
        self.dwell_times = {}              # Track how long people spend in areas
        
        # Heat map data for crowd analysis
        self.heat_map_data = {}            # Position frequency data
        self.heat_map_resolution = 50      # Grid resolution for heat map
        
        # Emergency evacuation support
        self.evacuation_mode = False       # Emergency evacuation mode
        self.evacuation_routes = {}        # Defined evacuation routes
        self.crowd_density_threshold = 0.8 # Threshold for crowd density alerts
        
        # Statistics for common area monitoring
        self.area_stats = {
            'total_visitors': 0,
            'current_occupancy': 0,
            'peak_occupancy': 0,
            'loitering_incidents': 0,
            'occupancy_violations': 0,
            'average_dwell_time': 0,
            'flow_efficiency': 0
        }
        
        # Zone color mapping
        self.zone_colors.update({
            'common_zone': (0, 255, 0),      # Green
            'waiting_area': (255, 165, 0),   # Orange  
            'reception_area': (0, 255, 255)  # Cyan
        })
        
        self.logger.info(f"Common area monitor initialized")
        self.logger.info(f"Monitoring zones: {len(self.common_zones)} common, {len(self.waiting_areas)} waiting, {len(self.reception_areas)} reception")
        self.logger.info(f"Loitering threshold: {self.loitering_threshold}s, Movement threshold: {self.movement_threshold}m")
    
    def _get_zones_by_type(self, zone_type: str) -> List[Dict]:
        """Extract zones of specific type from zone configuration"""
        zones_of_type = []
        
        # Handle both database format and config file format
        if zone_type in self.zones:
            zones_of_type = self.zones[zone_type]
        else:
            # Search through all zone types
            for zt, zone_list in self.zones.items():
                if isinstance(zone_list, list):
                    for zone in zone_list:
                        if zone.get('type') == zone_type or zone.get('zone_type') == zone_type:
                            zones_of_type.append(zone)
        
        return zones_of_type
    
    def _process_frame_impl(self, frame: np.ndarray, timestamp: float, 
                           detection_result: Any, ppe_result: Any = None) -> Tuple[np.ndarray, Dict]:
        """
        Process frame for common area monitoring
        
        Args:
            frame: Video frame
            timestamp: Frame timestamp
            detection_result: Person detection results from YOLO
            ppe_result: PPE detection results (optional for common areas)
            
        Returns:
            Tuple of (annotated_frame, processing_results)
        """
        
        # Extract people detections
        people_detections, ppe_detections = self.detect_people_and_ppe(frame, detection_result, ppe_result)
        
        # Update object tracking
        tracked_people = self.update_object_tracking(people_detections)
        
        # Store current people count
        self.current_people_count = len(tracked_people)
        
        # Update position tracking for loitering detection
        self._update_position_tracking(tracked_people, timestamp)
        
        # Analyze zone occupancy
        occupancy_analysis = self._analyze_zone_occupancy(tracked_people, timestamp)
        
        # Detect loitering
        loitering_events = self._detect_loitering(tracked_people, timestamp)
        
        # Check occupancy limits
        occupancy_events = self._check_occupancy_limits(tracked_people, timestamp)
        
        # Analyze crowd flow patterns
        flow_analysis = self._analyze_crowd_flow(tracked_people, timestamp)
        
        # Update heat map data
        self._update_heat_map_data(tracked_people, timestamp)
        
        # Check for emergency evacuation needs
        evacuation_analysis = self._check_evacuation_needs(tracked_people, timestamp)
        
        # Annotate frame with all detections and analysis
        annotated_frame = self._annotate_frame(frame, tracked_people, occupancy_analysis, flow_analysis)
        
        # Handle all detected events
        all_events = loitering_events + occupancy_events
        for event in all_events:
            self._handle_common_area_event(event, annotated_frame, timestamp)
        
        # Update statistics
        self._update_area_statistics(tracked_people, occupancy_analysis, flow_analysis)
        
        # Prepare processing results
        processing_results = {
            'people_count': len(tracked_people),
            'occupancy_analysis': occupancy_analysis,
            'flow_analysis': flow_analysis,
            'loitering_detections': len(loitering_events),
            'evacuation_analysis': evacuation_analysis,
            'events': all_events,
            'area_stats': self.area_stats.copy()
        }
        
        return annotated_frame, processing_results
    
    def _update_position_tracking(self, tracked_people: List[Dict], timestamp: float):
        """
        Update position tracking for loitering detection
        
        Args:
            tracked_people: List of tracked person detections
            timestamp: Current timestamp
        """
        current_track_ids = set()
        
        for person in tracked_people:
            track_id = person['track_id']
            current_position = person['center']
            current_track_ids.add(track_id)
            
            # Initialize tracking for new person
            if track_id not in self.person_last_positions:
                self.person_last_positions[track_id] = current_position
                self.person_stationary_times[track_id] = timestamp
                continue
            
            # Calculate movement distance
            last_position = self.person_last_positions[track_id]
            movement_distance = np.sqrt(
                (current_position[0] - last_position[0])**2 + 
                (current_position[1] - last_position[1])**2
            )
            
            # Check if person has moved significantly
            if movement_distance > self.movement_threshold:
                # Person has moved, reset stationary time
                self.person_stationary_times[track_id] = timestamp
                self.person_last_positions[track_id] = current_position
            
            # Update last known position
            self.person_last_positions[track_id] = current_position
        
        # Clean up tracking for people who are no longer visible
        track_ids_to_remove = []
        for track_id in self.person_last_positions:
            if track_id not in current_track_ids:
                track_ids_to_remove.append(track_id)
        
        for track_id in track_ids_to_remove:
            self.person_last_positions.pop(track_id, None)
            self.person_stationary_times.pop(track_id, None)
            self.loitering_warnings.pop(track_id, None)
    
    def _analyze_zone_occupancy(self, tracked_people: List[Dict], timestamp: float) -> Dict:
        """
        Analyze occupancy across all zones
        
        Args:
            tracked_people: List of tracked person detections
            timestamp: Current timestamp
            
        Returns:
            Dictionary with occupancy analysis
        """
        occupancy_analysis = {
            'zone_occupancy': {},
            'total_occupancy': len(tracked_people),
            'warnings': [],
            'trends': {}
        }
        
        all_zones = self.common_zones + self.waiting_areas + self.reception_areas
        
        for zone in all_zones:
            zone_name = zone.get('name', 'Common Area')
            zone_type = zone.get('type', zone.get('zone_type', 'common_zone'))
            occupancy_limit = zone.get('occupancy_limit', DatacenterConfig.get_occupancy_limit(zone_type))
            
            # Count people in this zone
            people_in_zone = []
            for person in tracked_people:
                if self._is_point_in_zone(person['center'], zone):
                    people_in_zone.append(person)
            
            current_occupancy = len(people_in_zone)
            occupancy_analysis['zone_occupancy'][zone_name] = {
                'current': current_occupancy,
                'limit': occupancy_limit,
                'utilization': current_occupancy / max(occupancy_limit, 1),
                'people': people_in_zone,
                'zone_type': zone_type
            }
            
            # Track occupancy history
            self.zone_occupancy_history[zone_name].append({
                'timestamp': timestamp,
                'occupancy': current_occupancy
            })
            
            # Keep only recent history (last 10 minutes)
            self.zone_occupancy_history[zone_name] = [
                entry for entry in self.zone_occupancy_history[zone_name]
                if timestamp - entry['timestamp'] <= 600
            ]
            
            # Check for warnings
            if current_occupancy >= occupancy_limit * self.occupancy_warning_ratio:
                occupancy_analysis['warnings'].append({
                    'zone': zone_name,
                    'type': 'approaching_limit',
                    'current': current_occupancy,
                    'limit': occupancy_limit
                })
            
            # Calculate trends
            if len(self.zone_occupancy_history[zone_name]) >= 2:
                recent_occupancy = [entry['occupancy'] for entry in self.zone_occupancy_history[zone_name][-5:]]
                trend = 'increasing' if recent_occupancy[-1] > recent_occupancy[0] else 'decreasing'
                occupancy_analysis['trends'][zone_name] = trend
        
        return occupancy_analysis
    
    def _detect_loitering(self, tracked_people: List[Dict], timestamp: float) -> List[Dict]:
        """
        Detect loitering (people staying in areas too long)
        
        Args:
            tracked_people: List of tracked person detections
            timestamp: Current timestamp
            
        Returns:
            List of loitering event dictionaries
        """
        loitering_events = []
        
        for person in tracked_people:
            track_id = person['track_id']
            
            # Check if person has been stationary too long
            if track_id in self.person_stationary_times:
                stationary_duration = timestamp - self.person_stationary_times[track_id]
                
                if stationary_duration >= self.loitering_threshold:
                    # Check if we've already warned about this person
                    warning_key = f"loitering_{track_id}"
                    
                    if warning_key not in self.loitering_warnings or \
                       (timestamp - self.loitering_warnings[warning_key]) >= self.event_cooldown:
                        
                        # Find which zone the person is in
                        zone_name = "Common Area"
                        zone_type = "common_zone"
                        
                        for zone in self.common_zones + self.waiting_areas + self.reception_areas:
                            if self._is_point_in_zone(person['center'], zone):
                                zone_name = zone.get('name', 'Common Area')
                                zone_type = zone.get('type', zone.get('zone_type', 'common_zone'))
                                break
                        
                        loitering_event = {
                            'type': DatacenterEventTypes.LOITERING,
                            'severity': 'medium',
                            'zone_name': zone_name,
                            'zone_type': zone_type,
                            'person': person,
                            'stationary_duration': stationary_duration,
                            'timestamp': timestamp,
                            'details': {
                                'loitering_threshold': self.loitering_threshold,
                                'movement_threshold': self.movement_threshold,
                                'detection_method': 'position_tracking'
                            }
                        }
                        
                        loitering_events.append(loitering_event)
                        self.loitering_warnings[warning_key] = timestamp
                        self.area_stats['loitering_incidents'] += 1
                        
                        self.logger.warning(f"Loitering detected in {zone_name}: Person {track_id} stationary for {stationary_duration:.1f}s")
        
        return loitering_events
    
    def _check_occupancy_limits(self, tracked_people: List[Dict], timestamp: float) -> List[Dict]:
        """
        Check for occupancy limit violations
        
        Args:
            tracked_people: List of tracked person detections
            timestamp: Current timestamp
            
        Returns:
            List of occupancy violation events
        """
        occupancy_events = []
        
        all_zones = self.common_zones + self.waiting_areas + self.reception_areas
        
        for zone in all_zones:
            zone_name = zone.get('name', 'Common Area')
            zone_type = zone.get('type', zone.get('zone_type', 'common_zone'))
            occupancy_limit = zone.get('occupancy_limit', DatacenterConfig.get_occupancy_limit(zone_type))
            
            # Count people in this zone
            people_in_zone = []
            for person in tracked_people:
                if self._is_point_in_zone(person['center'], zone):
                    people_in_zone.append(person)
            
            if len(people_in_zone) > occupancy_limit:
                event_key = f"occupancy_{zone_name}"
                
                if event_key not in self.occupancy_warnings or \
                   (timestamp - self.occupancy_warnings[event_key]) >= self.event_cooldown:
                    
                    occupancy_event = {
                        'type': DatacenterEventTypes.PEOPLE_COUNTING,
                        'severity': 'medium',
                        'zone_name': zone_name,
                        'zone_type': zone_type,
                        'people_count': len(people_in_zone),
                        'occupancy_limit': occupancy_limit,
                        'people': people_in_zone,
                        'timestamp': timestamp,
                        'details': {
                            'violation_type': 'occupancy_exceeded',
                            'excess_count': len(people_in_zone) - occupancy_limit,
                            'utilization': len(people_in_zone) / occupancy_limit
                        }
                    }
                    
                    occupancy_events.append(occupancy_event)
                    self.occupancy_warnings[event_key] = timestamp
                    self.area_stats['occupancy_violations'] += 1
                    
                    self.logger.warning(f"Occupancy limit exceeded in {zone_name}: {len(people_in_zone)}/{occupancy_limit}")
        
        return occupancy_events
    
    def _analyze_crowd_flow(self, tracked_people: List[Dict], timestamp: float) -> Dict:
        """
        Analyze crowd flow patterns and efficiency
        
        Args:
            tracked_people: List of tracked person detections
            timestamp: Current timestamp
            
        Returns:
            Dictionary with flow analysis
        """
        flow_analysis = {
            'active_tracks': len(tracked_people),
            'movement_patterns': [],
            'bottlenecks': [],
            'flow_efficiency': 0,
            'crowd_density': 0
        }
        
        if not self.flow_analysis_enabled:
            return flow_analysis
        
        # Analyze movement patterns
        for person in tracked_people:
            track_id = person['track_id']
            
            if track_id in self.tracked_objects:
                positions = self.tracked_objects[track_id].get('positions', [])
                
                if len(positions) >= 2:
                    # Calculate movement vector
                    start_pos = positions[0]
                    end_pos = positions[-1]
                    
                    movement_vector = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
                    movement_distance = np.sqrt(movement_vector[0]**2 + movement_vector[1]**2)
                    
                    # Calculate movement speed
                    time_tracked = timestamp - self.tracked_objects[track_id].get('first_seen', timestamp)
                    speed = movement_distance / max(time_tracked, 1)
                    
                    flow_analysis['movement_patterns'].append({
                        'track_id': track_id,
                        'movement_vector': movement_vector,
                        'distance': movement_distance,
                        'speed': speed,
                        'direction': self._calculate_movement_direction(movement_vector)
                    })
        
        # Calculate crowd density
        if tracked_people:
            # Calculate area covered by people
            positions = [person['center'] for person in tracked_people]
            if len(positions) >= 2:
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                
                area_width = max(x_coords) - min(x_coords)
                area_height = max(y_coords) - min(y_coords)
                covered_area = max(area_width * area_height, 1)
                
                flow_analysis['crowd_density'] = len(tracked_people) / covered_area * 10000  # People per 10k pixels
        
        # Calculate flow efficiency (higher is better)
        if flow_analysis['movement_patterns']:
            total_speed = sum(pattern['speed'] for pattern in flow_analysis['movement_patterns'])
            average_speed = total_speed / len(flow_analysis['movement_patterns'])
            flow_analysis['flow_efficiency'] = min(average_speed / 10, 1.0)  # Normalize to 0-1
        
        return flow_analysis
    
    def _calculate_movement_direction(self, movement_vector: Tuple[float, float]) -> str:
        """Calculate primary movement direction"""
        dx, dy = movement_vector
        
        if abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'down' if dy > 0 else 'up'
    
    def _update_heat_map_data(self, tracked_people: List[Dict], timestamp: float):
        """
        Update heat map data for position frequency analysis
        
        Args:
            tracked_people: List of tracked person detections
            timestamp: Current timestamp
        """
        for person in tracked_people:
            # Convert position to grid coordinates
            x, y = person['center']
            grid_x = int(x // self.heat_map_resolution)
            grid_y = int(y // self.heat_map_resolution)
            
            grid_key = (grid_x, grid_y)
            
            if grid_key not in self.heat_map_data:
                self.heat_map_data[grid_key] = {'count': 0, 'last_update': timestamp}
            
            self.heat_map_data[grid_key]['count'] += 1
            self.heat_map_data[grid_key]['last_update'] = timestamp
        
        # Clean up old heat map data (older than 1 hour)
        keys_to_remove = []
        for key, data in self.heat_map_data.items():
            if timestamp - data['last_update'] > 3600:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.heat_map_data[key]
    
    def _check_evacuation_needs(self, tracked_people: List[Dict], timestamp: float) -> Dict:
        """
        Check if evacuation assistance is needed based on crowd density
        
        Args:
            tracked_people: List of tracked person detections
            timestamp: Current timestamp
            
        Returns:
            Dictionary with evacuation analysis
        """
        evacuation_analysis = {
            'evacuation_needed': False,
            'crowd_density_alert': False,
            'blocked_exits': [],
            'recommended_actions': []
        }
        
        # Check crowd density
        if len(tracked_people) > 0:
            # Calculate crowd density in each zone
            for zone in self.common_zones + self.waiting_areas + self.reception_areas:
                zone_name = zone.get('name', 'Area')
                people_in_zone = [p for p in tracked_people if self._is_point_in_zone(p['center'], zone)]
                
                if len(people_in_zone) > 0:
                    # Calculate zone area (simplified)
                    coordinates = zone.get('coordinates', [])
                    if len(coordinates) >= 3:
                        # Simple area calculation (not accurate for complex polygons)
                        zone_area = self._calculate_polygon_area(coordinates)
                        density = len(people_in_zone) / max(zone_area, 1)
                        
                        if density > self.crowd_density_threshold:
                            evacuation_analysis['crowd_density_alert'] = True
                            evacuation_analysis['recommended_actions'].append(
                                f"High crowd density in {zone_name}: {len(people_in_zone)} people"
                            )
        
        return evacuation_analysis
    
    def _calculate_polygon_area(self, coordinates: List[List[float]]) -> float:
        """Calculate approximate area of polygon"""
        if len(coordinates) < 3:
            return 0
        
        # Shoelace formula
        area = 0
        n = len(coordinates)
        
        for i in range(n):
            j = (i + 1) % n
            area += coordinates[i][0] * coordinates[j][1]
            area -= coordinates[j][0] * coordinates[i][1]
        
        return abs(area) / 2
    
    def _update_area_statistics(self, tracked_people: List[Dict], 
                               occupancy_analysis: Dict, flow_analysis: Dict):
        """Update area statistics"""
        
        self.area_stats['current_occupancy'] = len(tracked_people)
        self.area_stats['peak_occupancy'] = max(self.area_stats['peak_occupancy'], len(tracked_people))
        self.area_stats['flow_efficiency'] = flow_analysis.get('flow_efficiency', 0)
        
        # Update visitor count (simplified - should track unique entries)
        self.area_stats['total_visitors'] = len(self.tracked_objects)
        
        # Calculate average dwell time
        if self.tracked_objects:
            current_time = time.time()
            total_dwell_time = sum(
                current_time - obj.get('first_seen', current_time)
                for obj in self.tracked_objects.values()
            )
            self.area_stats['average_dwell_time'] = total_dwell_time / len(self.tracked_objects)
    
    def _handle_common_area_event(self, event: Dict, frame: np.ndarray, timestamp: float):
        """Handle detected common area events"""
        try:
            event_type = event['type']
            severity = event['severity']
            zone_name = event.get('zone_name', 'Common Area')
            
            # Log event for audit
            audit_logger.log_event_detection(
                event_type=event_type,
                camera_id=str(self.camera_id),
                datacenter_id=str(self.datacenter_id),
                severity=severity,
                detection_data=event.get('details', {})
            )
            
            # Generate event ID
            event_id = str(uuid.uuid4())
            
            # Save event based on severity
            if severity in ['medium', 'high']:
                self._save_event_media(event_type, event, frame, timestamp, zone_name, event_id)
            
            self.logger.info(f"Common area event handled: {event_type} in {zone_name}")
            
        except Exception as e:
            self.logger.error(f"Error handling common area event: {e}")
    
    def _annotate_frame(self, frame: np.ndarray, tracked_people: List[Dict], 
                       occupancy_analysis: Dict, flow_analysis: Dict) -> np.ndarray:
        """Annotate frame with common area monitoring information"""
        
        annotated_frame = frame.copy()
        
        # Draw zones
        for zone in self.common_zones:
            self._draw_zone(annotated_frame, zone, self.zone_colors['common_zone'], 
                          alpha=0.2, label=zone.get('name', 'Common'))
        
        for zone in self.waiting_areas:
            self._draw_zone(annotated_frame, zone, self.zone_colors['waiting_area'], 
                          alpha=0.2, label=zone.get('name', 'Waiting'))
        
        for zone in self.reception_areas:
            self._draw_zone(annotated_frame, zone, self.zone_colors['reception_area'], 
                          alpha=0.2, label=zone.get('name', 'Reception'))
        
        # Draw people with loitering indicators
        for person in tracked_people:
            bbox = person['bbox']
            track_id = person['track_id']
            
            # Check if person is loitering
            color = (0, 255, 0)  # Green default
            label = f"ID:{track_id}"
            
            if track_id in self.person_stationary_times:
                stationary_duration = time.time() - self.person_stationary_times[track_id]
                
                if stationary_duration >= self.loitering_threshold:
                    color = (0, 0, 255)  # Red for loitering
                    label += f" (LOITERING {stationary_duration:.0f}s)"
                elif stationary_duration >= self.loitering_threshold * 0.5:
                    color = (0, 165, 255)  # Orange for warning
                    label += f" ({stationary_duration:.0f}s)"
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
            
            # Draw label
            cv2.putText(annotated_frame, label,
                       (int(bbox[0]), int(bbox[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw occupancy information
        y_offset = 30
        for zone_name, occupancy_info in occupancy_analysis.get('zone_occupancy', {}).items():
            current = occupancy_info['current']
            limit = occupancy_info['limit']
            utilization = occupancy_info['utilization']
            
            color = (255, 255, 255)
            if utilization >= 1.0:
                color = (0, 0, 255)  # Red for over limit
            elif utilization >= 0.8:
                color = (0, 165, 255)  # Orange for warning
            
            occupancy_text = f"{zone_name}: {current}/{limit} ({utilization:.1%})"
            cv2.putText(annotated_frame, occupancy_text,
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        # Draw flow efficiency
        flow_efficiency = flow_analysis.get('flow_efficiency', 0)
        crowd_density = flow_analysis.get('crowd_density', 0)
        
        flow_text = f"Flow Efficiency: {flow_efficiency:.1%} | Crowd Density: {crowd_density:.1f}"
        cv2.putText(annotated_frame, flow_text,
                   (10, annotated_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw statistics
        stats_text = f"People: {len(tracked_people)} | Peak: {self.area_stats['peak_occupancy']} | Loitering: {self.area_stats['loitering_incidents']}"
        cv2.putText(annotated_frame, stats_text,
                   (10, annotated_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw movement vectors for flow analysis
        for pattern in flow_analysis.get('movement_patterns', []):
            track_id = pattern['track_id']
            # Find the person with this track_id
            person = next((p for p in tracked_people if p['track_id'] == track_id), None)
            if person:
                start_point = person['center']
                movement_vector = pattern['movement_vector']
                
                # Scale down the vector for visualization
                end_point = (
                    int(start_point[0] + movement_vector[0] * 0.1),
                    int(start_point[1] + movement_vector[1] * 0.1)
                )
                
                # Draw movement vector
                cv2.arrowedLine(annotated_frame, 
                               (int(start_point[0]), int(start_point[1])), 
                               end_point, (255, 0, 255), 2)
        
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
    
    def _save_frame_with_detections(self, event_id: str, frame: np.ndarray, timestamp: float) -> Optional[str]:
        """Save annotated frame to disk"""
        try:
            # Generate filename
            timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S_%f')
            frame_filename = f"cam{self.camera_id}_{timestamp_str}_{event_id[:8]}.jpg"
            frame_path = os.path.join(self.camera_output_dir, frame_filename)
            
            # Save frame
            cv2.imwrite(frame_path, frame)
            
            # Queue for cloud storage if available
            if self.db_writer:
                from db_writer import ProcessedFrame
                frame_info = ProcessedFrame(
                    camera_id=self.camera_id,
                    event_id=event_id,
                    timestamp=datetime.fromtimestamp(timestamp),
                    local_path=frame_path,
                    frame_path=""  # Will be set by db_writer during upload
                )
                self.db_writer.queue_frame(frame_info)
            
            return frame_path
            
        except Exception as e:
            self.logger.error(f"Error saving frame: {e}")
            return None
    
    def _trigger_video_recording(self, event_id: str, event_data: Dict, timestamp: float):
        """Trigger video recording for event"""
        try:
            # Check cooldown to prevent multiple recordings
            current_time = time.time()
            if current_time - self.last_recording_time < self.recording_cooldown:
                return
            
            self.last_recording_time = current_time
            
            # Create video recording thread
            import threading
            video_thread = threading.Thread(
                target=self._save_event_video,
                args=(event_id, event_data, timestamp),
                daemon=True
            )
            video_thread.start()
            
        except Exception as e:
            self.logger.error(f"Error triggering video recording: {e}")
    
    def _save_event_video(self, event_id: str, event_data: Dict, timestamp: float):
        """Save event video from frame buffer"""
        try:
            # Get frames from buffer
            with self.buffer_lock:
                frames = list(self.frame_buffer)
                timestamps = list(self.timestamp_buffer)
            
            if not frames:
                return
            
            # Generate video filename
            timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S')
            video_filename = f"cam{self.camera_id}_{timestamp_str}_{event_id[:8]}.{self.video_extension}"
            video_path = os.path.join(self.video_output_dir, video_filename)
            
            # Create video writer
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
            video_writer = cv2.VideoWriter(video_path, fourcc, self.video_fps, (width, height))
            
            # Write frames
            for frame in frames:
                video_writer.write(frame)
            
            video_writer.release()
            
            # Queue video metadata for database
            if self.db_writer:
                video_metadata = {
                    'event_id': event_id,
                    'video_path': video_path,
                    'duration': len(frames) / self.video_fps,
                    'camera_id': self.camera_id,
                    'timestamp': timestamp
                }
                self.db_writer.queue_video_metadata(video_metadata)
            
            self.logger.info(f"Video saved: {video_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving video: {e}")
    
    def get_heat_map_data(self) -> Dict:
        """Get current heat map data for visualization"""
        return self.heat_map_data.copy()
    
    def get_occupancy_trends(self) -> Dict:
        """Get occupancy trends for analytics"""
        return {
            zone_name: list(history) 
            for zone_name, history in self.zone_occupancy_history.items()
        }
    
    def set_evacuation_mode(self, enabled: bool):
        """Enable/disable evacuation mode"""
        self.evacuation_mode = enabled
        if enabled:
            self.logger.warning("Evacuation mode enabled - enhanced monitoring active")
        else:
            self.logger.info("Evacuation mode disabled")
    
    def get_flow_statistics(self) -> Dict:
        """Get flow analysis statistics"""
        return {
            'current_people': len(self.tracked_objects),
            'flow_efficiency': self.area_stats['flow_efficiency'],
            'average_dwell_time': self.area_stats['average_dwell_time'],
            'heat_map_points': len(self.heat_map_data),
            'active_movement_patterns': len([
                obj for obj in self.tracked_objects.values()
                if len(obj.get('positions', [])) >= 2
            ])
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            super().cleanup()
            
            # Clear common area specific data
            self.person_stationary_times.clear()
            self.person_last_positions.clear()
            self.loitering_warnings.clear()
            self.occupancy_warnings.clear()
            self.zone_occupancy_history.clear()
            self.heat_map_data.clear()
            
            self.logger.info("Common area monitor cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

# Export the monitor class
__all__ = ['CommonAreaMonitor']