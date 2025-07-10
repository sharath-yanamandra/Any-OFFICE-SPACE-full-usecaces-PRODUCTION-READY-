#!/usr/bin/env python3
"""
Script 15: video_processor.py
File Path: src/video_processor.py

Datacenter Monitoring System - Main Video Processing Orchestrator

This module orchestrates the entire video processing pipeline:
1. Coordinates camera managers, model managers, and camera models
2. Manages batch processing for GPU efficiency
3. Routes detection results to appropriate camera models
4. Handles multi-camera coordination and cross-camera logic
5. Manages database writing and cloud storage
6. Provides system monitoring and health checks
"""

import os
import sys
import threading
import time
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple
from functools import partial

# Import core components
from database import DatacenterDatabase
from logger import setup_datacenter_logger, audit_logger, performance_logger
from config import DatacenterConfig, DatacenterEventTypes, DatacenterCameraTypes, DATACENTER_CAMERA_MODEL_MAPPING
from camera_manager import DatacenterCameraManager
from model_manager import DatacenterModelManager
from db_writer import DatacenterDatabaseWriter
from storage_handler import DatacenterStorageHandler

# Import camera models
try:
    from camera_models import (
        DatacenterEntryMonitor,
        ServerRoomMonitor,
        CorridorMonitor,
        PerimeterMonitor,
        CriticalZoneMonitor,
        CommonAreaMonitor
    )
    from camera_models.multi_camera_coordinator import MultiCameraCoordinator
    camera_models_available = True
except ImportError as e:
    camera_models_available = False
    print(f"Warning: Some camera models not available: {e}")

# Camera model mapping
CAMERA_MODEL_CLASSES = {
    DatacenterCameraTypes.ENTRY_MONITOR: DatacenterEntryMonitor,
    DatacenterCameraTypes.SERVER_ROOM: ServerRoomMonitor,
    DatacenterCameraTypes.CORRIDOR: CorridorMonitor,
    DatacenterCameraTypes.PERIMETER: PerimeterMonitor,
    DatacenterCameraTypes.CRITICAL_ZONE: CriticalZoneMonitor,
    DatacenterCameraTypes.COMMON_AREA: CommonAreaMonitor
}

class DatacenterVideoProcessor:
    """
    Main video processing orchestrator for datacenter monitoring system.
    Coordinates all components and manages the processing pipeline.
    """
    
    def __init__(self, datacenter_id: Optional[int] = None):
        """
        Initialize the datacenter video processor
        
        Args:
            datacenter_id: Optional datacenter ID for filtering cameras
        """
        self.logger = setup_datacenter_logger(
            'datacenter_video_processor', 
            'datacenter_video_processor.log',
            datacenter_id=str(datacenter_id) if datacenter_id else None
        )
        self.logger.info("Initializing DatacenterVideoProcessor")
        
        # Store datacenter filter
        self.datacenter_id = datacenter_id
        if self.datacenter_id:
            self.logger.info(f"Filtering for datacenter ID: {self.datacenter_id}")
        
        # Initialize core components
        self.database = DatacenterDatabase()
        self.camera_manager = DatacenterCameraManager()
        self.model_manager = DatacenterModelManager()
        self.db_writer = DatacenterDatabaseWriter()
        self.storage_handler = DatacenterStorageHandler()
        
        # Camera configuration and models
        self.camera_models = {}
        self.camera_metadata = {}
        self.camera_feeds = {}
        
        # Multi-camera coordination
        self.datacenter_coordinators = {}  # {datacenter_id: MultiCameraCoordinator}
        
        # Batch processing configuration
        self.batch_size = DatacenterConfig.BATCH_SIZE
        self.batch_timeout = DatacenterConfig.BATCH_TIMEOUT
        self.max_parallel_cameras = DatacenterConfig.MAX_PARALLEL_CAMERAS
        
        # Processing threads and state
        self.batch_processing_thread = None
        self.batch_processing_running = False
        self.processing_executor = None
        
        # System monitoring
        self.system_stats = {
            'total_frames_processed': 0,
            'total_events_detected': 0,
            'cameras_active': 0,
            'average_fps': 0,
            'uptime_seconds': 0,
            'start_time': time.time()
        }
        
        # Load camera configurations and initialize models
        self._load_camera_configurations()
        self._initialize_camera_models()
        self._initialize_multi_camera_coordinators()
        
        # Set up result routing
        self._setup_result_routing()
        
        self.logger.info("DatacenterVideoProcessor initialization complete")
    
    def _load_camera_configurations(self):
        """Load camera configurations from database"""
        try:
            self.logger.info("Loading camera configurations from database")
            
            # Build query based on datacenter filter
            if self.datacenter_id:
                query = """
                    SELECT 
                        c.camera_id, c.datacenter_id, c.name, c.stream_url, c.camera_type,
                        c.location_details, c.status, c.metadata,
                        d.name as datacenter_name, d.location as datacenter_location
                    FROM cameras c
                    JOIN datacenters d ON c.datacenter_id = d.datacenter_id
                    WHERE c.status = 'active' AND d.status = 'active' AND c.datacenter_id = %s
                    ORDER BY c.camera_id
                """
                cameras = self.database.execute_query(query, (self.datacenter_id,))
            else:
                query = """
                    SELECT 
                        c.camera_id, c.datacenter_id, c.name, c.stream_url, c.camera_type,
                        c.location_details, c.status, c.metadata,
                        d.name as datacenter_name, d.location as datacenter_location
                    FROM cameras c
                    JOIN datacenters d ON c.datacenter_id = d.datacenter_id
                    WHERE c.status = 'active' AND d.status = 'active'
                    ORDER BY c.datacenter_id, c.camera_id
                """
                cameras = self.database.execute_query(query)
            
            if not cameras:
                self.logger.warning("No active cameras found in database")
                return
            
            # Process camera configurations
            for camera in cameras:
                camera_id = camera['camera_id']
                datacenter_id = camera['datacenter_id']
                
                # Parse metadata
                metadata = camera['metadata']
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                elif not metadata:
                    metadata = {}
                
                # Determine activity level
                activity_level = metadata.get('activity_level', 'medium')
                
                # Store camera feed info
                self.camera_feeds[camera_id] = (camera['stream_url'], activity_level)
                
                # Store camera metadata
                self.camera_metadata[camera_id] = {
                    'camera_id': camera_id,
                    'datacenter_id': datacenter_id,
                    'name': camera['name'],
                    'stream_url': camera['stream_url'],
                    'camera_type': camera['camera_type'],
                    'location_details': camera['location_details'],
                    'activity_level': activity_level,
                    'datacenter_name': camera['datacenter_name'],
                    'datacenter_location': camera['datacenter_location'],
                    'frames_processed': 0,
                    'events_detected': 0,
                    'last_processed': None,
                    'metadata': metadata
                }
                
                self.logger.info(f"Loaded camera {camera_id} ({camera['name']}) - Type: {camera['camera_type']}")
            
            self.logger.info(f"Loaded {len(self.camera_feeds)} camera configurations")
            
        except Exception as e:
            self.logger.error(f"Error loading camera configurations: {str(e)}", exc_info=True)
    
    def _initialize_camera_models(self):
        """Initialize camera models based on camera types"""
        try:
            self.logger.info("Initializing camera models")
            
            if not camera_models_available:
                self.logger.warning("Camera models not available, using basic processing")
                return
            
            for camera_id, camera_info in self.camera_metadata.items():
                camera_type = camera_info['camera_type']
                datacenter_id = camera_info['datacenter_id']
                
                # Get zones and rules for this camera
                zones = self._get_camera_zones(camera_id)
                rules = self._get_camera_rules(camera_id)
                
                # Get camera settings
                settings = camera_info.get('metadata', {})
                
                # Initialize appropriate camera model
                if camera_type in CAMERA_MODEL_CLASSES:
                    model_class = CAMERA_MODEL_CLASSES[camera_type]
                    
                    try:
                        camera_model = model_class(
                            camera_id=camera_id,
                            datacenter_id=datacenter_id,
                            zones=zones,
                            rules=rules,
                            settings=settings,
                            db=self.database,
                            db_writer=self.db_writer,
                            frames_base_dir=DatacenterConfig.FRAMES_OUTPUT_DIR
                        )
                        
                        self.camera_models[camera_id] = camera_model
                        self.logger.info(f"Initialized {camera_type} model for camera {camera_id}")
                        
                    except Exception as e:
                        self.logger.error(f"Error initializing model for camera {camera_id}: {str(e)}")
                        
                else:
                    self.logger.warning(f"Unknown camera type '{camera_type}' for camera {camera_id}")
            
            self.logger.info(f"Initialized {len(self.camera_models)} camera models")
            
        except Exception as e:
            self.logger.error(f"Error initializing camera models: {str(e)}", exc_info=True)
    
    def _get_camera_zones(self, camera_id: int) -> Dict:
        """Get zone definitions for a camera"""
        try:
            query = """
                SELECT zone_id, name, zone_type, polygon_coordinates, security_level, 
                       access_requirements, monitoring_rules, metadata
                FROM zones
                WHERE camera_id = %s
                ORDER BY zone_type, name
            """
            
            zones_data = self.database.execute_query(query, (camera_id,))
            
            # Organize zones by type
            zones = {}
            for zone_data in zones_data:
                zone_type = zone_data['zone_type']
                
                if zone_type not in zones:
                    zones[zone_type] = []
                
                # Parse coordinates
                coordinates = zone_data['polygon_coordinates']
                if isinstance(coordinates, str):
                    try:
                        coordinates = json.loads(coordinates)
                    except json.JSONDecodeError:
                        coordinates = []
                
                # Parse metadata
                metadata = zone_data['metadata']
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                elif not metadata:
                    metadata = {}
                
                zone = {
                    'zone_id': zone_data['zone_id'],
                    'name': zone_data['name'],
                    'type': zone_type,
                    'coordinates': coordinates,
                    'security_level': zone_data['security_level'],
                    'access_requirements': zone_data['access_requirements'],
                    'monitoring_rules': zone_data['monitoring_rules'],
                    **metadata
                }
                
                zones[zone_type].append(zone)
            
            return zones
            
        except Exception as e:
            self.logger.error(f"Error getting zones for camera {camera_id}: {str(e)}")
            return {}
    
    def _get_camera_rules(self, camera_id: int) -> List[Dict]:
        """Get monitoring rules for a camera"""
        try:
            query = """
                SELECT rule_id, name, description, event_type, severity, 
                       parameters, schedule, notification_settings, enabled
                FROM rules
                WHERE camera_id = %s AND enabled = TRUE
                ORDER BY severity DESC, rule_id
            """
            
            rules_data = self.database.execute_query(query, (camera_id,))
            
            rules = []
            for rule_data in rules_data:
                # Parse parameters
                parameters = rule_data['parameters']
                if isinstance(parameters, str):
                    try:
                        parameters = json.loads(parameters)
                    except json.JSONDecodeError:
                        parameters = {}
                elif not parameters:
                    parameters = {}
                
                rule = {
                    'rule_id': rule_data['rule_id'],
                    'name': rule_data['name'],
                    'description': rule_data['description'],
                    'event_type': rule_data['event_type'],
                    'severity': rule_data['severity'],
                    'parameters': parameters,
                    'schedule': rule_data['schedule'],
                    'notification_settings': rule_data['notification_settings'],
                    'enabled': rule_data['enabled']
                }
                
                rules.append(rule)
            
            return rules
            
        except Exception as e:
            self.logger.error(f"Error getting rules for camera {camera_id}: {str(e)}")
            return []
    
    def _initialize_multi_camera_coordinators(self):
        """Initialize multi-camera coordinators for each datacenter"""
        try:
            # Group cameras by datacenter
            datacenter_cameras = {}
            for camera_id, camera_info in self.camera_metadata.items():
                datacenter_id = camera_info['datacenter_id']
                if datacenter_id not in datacenter_cameras:
                    datacenter_cameras[datacenter_id] = []
                datacenter_cameras[datacenter_id].append(camera_id)
            
            # Create coordinators for datacenters with multiple cameras
            for datacenter_id, camera_ids in datacenter_cameras.items():
                if len(camera_ids) > 1:
                    try:
                        coordinator = MultiCameraCoordinator(
                            datacenter_id=datacenter_id,
                            camera_ids=camera_ids,
                            camera_models=self.camera_models
                        )
                        
                        self.datacenter_coordinators[datacenter_id] = coordinator
                        self.logger.info(f"Created coordinator for datacenter {datacenter_id} with {len(camera_ids)} cameras")
                        
                    except Exception as e:
                        self.logger.error(f"Error creating coordinator for datacenter {datacenter_id}: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error initializing multi-camera coordinators: {str(e)}", exc_info=True)
    
    def _setup_result_routing(self):
        """Setup result routing from batch processing to camera models"""
        try:
            self.logger.info("Setting up result routing")
            
            # Set result callback for camera manager
            self.camera_manager.set_result_callback(self._process_camera_result)
            
            self.logger.info("Result routing setup complete")
            
        except Exception as e:
            self.logger.error(f"Error setting up result routing: {str(e)}", exc_info=True)
    
    def _process_camera_result(self, camera_id: int, frame, result, timestamp, enhanced_metadata=None):
        """Process detection result for a specific camera"""
        try:
            # Update camera metadata
            if camera_id in self.camera_metadata:
                self.camera_metadata[camera_id]['last_processed'] = timestamp
                self.camera_metadata[camera_id]['frames_processed'] += 1
            
            # Update system statistics
            self.system_stats['total_frames_processed'] += 1
            
            # Process with camera model if available
            if camera_id in self.camera_models:
                camera_model = self.camera_models[camera_id]
                
                # Extract detection results
                detection_result = enhanced_metadata.get('detection_result') if enhanced_metadata else result
                ppe_result = enhanced_metadata.get('ppe_result') if enhanced_metadata else None
                
                # Process frame through camera model
                annotated_frame, processing_results = camera_model.process_frame(
                    frame, timestamp, detection_result, ppe_result
                )
                
                # Handle cross-camera coordination
                if enhanced_metadata and enhanced_metadata.get('cross_camera_coordination'):
                    self._handle_cross_camera_coordination(camera_id, processing_results, timestamp)
                
                # Update event statistics
                events = processing_results.get('events', [])
                if events:
                    self.system_stats['total_events_detected'] += len(events)
                    self.camera_metadata[camera_id]['events_detected'] += len(events)
                
                # Log processing performance
                if self.camera_metadata[camera_id]['frames_processed'] % 100 == 0:
                    self._log_processing_performance(camera_id)
                
            else:
                self.logger.warning(f"No camera model available for camera {camera_id}")
                
        except Exception as e:
            self.logger.error(f"Error processing result for camera {camera_id}: {str(e)}", exc_info=True)
    
    def _handle_cross_camera_coordination(self, camera_id: int, processing_results: Dict, timestamp: float):
        """Handle cross-camera coordination logic"""
        try:
            datacenter_id = self.camera_metadata[camera_id]['datacenter_id']
            
            if datacenter_id in self.datacenter_coordinators:
                coordinator = self.datacenter_coordinators[datacenter_id]
                
                # Update coordinator with camera results
                people_count = processing_results.get('people_count', 0)
                events = processing_results.get('events', [])
                
                coordinator.update_camera_data(camera_id, people_count, events, timestamp)
                
                # Get coordination results
                coordination_result = coordinator.get_coordination_result()
                
                if coordination_result.get('trigger_cross_camera_event'):
                    self._handle_cross_camera_event(coordination_result, timestamp)
                    
        except Exception as e:
            self.logger.error(f"Error in cross-camera coordination: {str(e)}", exc_info=True)
    
    def _handle_cross_camera_event(self, coordination_result: Dict, timestamp: float):
        """Handle events that span multiple cameras"""
        try:
            event_type = coordination_result.get('event_type')
            affected_cameras = coordination_result.get('cameras', [])
            
            # Log cross-camera event
            audit_logger.log_event_detection(
                event_type=f"cross_camera_{event_type}",
                camera_id=','.join(map(str, affected_cameras)),
                datacenter_id=str(coordination_result.get('datacenter_id')),
                severity=coordination_result.get('severity', 'medium'),
                detection_data=coordination_result
            )
            
            self.logger.info(f"Cross-camera event detected: {event_type} across cameras {affected_cameras}")
            
        except Exception as e:
            self.logger.error(f"Error handling cross-camera event: {str(e)}", exc_info=True)
    
    def _log_processing_performance(self, camera_id: int):
        """Log processing performance metrics"""
        try:
            camera_info = self.camera_metadata[camera_id]
            frames_processed = camera_info['frames_processed']
            
            if frames_processed > 0:
                # Calculate performance metrics
                current_time = time.time()
                uptime = current_time - self.system_stats['start_time']
                fps = frames_processed / uptime if uptime > 0 else 0
                
                # Log performance
                performance_logger.log_processing_stats(
                    camera_id=str(camera_id),
                    fps=fps,
                    batch_size=self.batch_size,
                    processing_time=1.0 / fps if fps > 0 else 0,
                    queue_size=len(self.camera_manager.get_camera_queue_sizes().get(camera_id, []))
                )
                
        except Exception as e:
            self.logger.error(f"Error logging performance for camera {camera_id}: {str(e)}")
    
    async def start_monitoring(self) -> bool:
        """Start the monitoring system"""
        try:
            self.logger.info("Starting datacenter monitoring system")
            
            if not self.camera_feeds:
                self.logger.error("No camera feeds configured")
                return False
            
            # Start camera manager
            cameras_started = self.camera_manager.start_cameras(self.camera_feeds)
            if cameras_started == 0:
                self.logger.error("Failed to start any cameras")
                return False
            
            # Start batch processing
            success = await self._start_batch_processing()
            if not success:
                self.logger.error("Failed to start batch processing")
                return False
            
            # Update system stats
            self.system_stats['cameras_active'] = cameras_started
            
            self.logger.info(f"Monitoring started successfully for {cameras_started} cameras")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring: {str(e)}", exc_info=True)
            return False
    
    async def _start_batch_processing(self) -> bool:
        """Start batch processing of camera frames"""
        try:
            if self.batch_processing_running:
                self.logger.warning("Batch processing already running")
                return True
            
            self.logger.info("Starting batch processing")
            
            # Create thread pool for parallel processing
            self.processing_executor = ThreadPoolExecutor(
                max_workers=self.max_parallel_cameras,
                thread_name_prefix="CameraProcessor"
            )
            
            # Start batch processing thread
            self.batch_processing_running = True
            self.batch_processing_thread = threading.Thread(
                target=self._batch_processing_worker,
                daemon=True,
                name="BatchProcessor"
            )
            self.batch_processing_thread.start()
            
            self.logger.info("Batch processing started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting batch processing: {str(e)}", exc_info=True)
            return False
    
    def _batch_processing_worker(self):
        """Main batch processing worker thread"""
        try:
            self.logger.info("Batch processing worker started")
            
            # Get detection model
            detection_model = self.model_manager.get_model('detection')
            if not detection_model:
                self.logger.error("Detection model not available")
                return
            
            # Get PPE model if enabled
            ppe_model = None
            if DatacenterConfig.PPE_DETECTION_ENABLED:
                ppe_model = self.model_manager.get_model('ppe_detection')
            
            frames_processed = 0
            batches_processed = 0
            
            while self.batch_processing_running:
                try:
                    # Get batch of frames
                    batch_data = self.camera_manager.get_batch(timeout=self.batch_timeout)
                    
                    if not batch_data or not batch_data[0]:
                        time.sleep(0.01)
                        continue
                    
                    frames, metadata = batch_data
                    batch_size = len(frames)
                    
                    if batch_size == 0:
                        continue
                    
                    # Process batch
                    self._process_batch(detection_model, ppe_model, frames, metadata)
                    
                    # Update statistics
                    frames_processed += batch_size
                    batches_processed += 1
                    
                    # Log progress periodically
                    if batches_processed % 10 == 0:
                        elapsed = time.time() - self.system_stats['start_time']
                        fps = frames_processed / elapsed if elapsed > 0 else 0
                        self.logger.info(f"Processed {batches_processed} batches, {frames_processed} frames, {fps:.1f} FPS")
                    
                except Exception as e:
                    self.logger.error(f"Error in batch processing: {str(e)}", exc_info=True)
                    time.sleep(0.1)
            
            self.logger.info("Batch processing worker stopped")
            
        except Exception as e:
            self.logger.error(f"Fatal error in batch processing worker: {str(e)}", exc_info=True)
        finally:
            # Release models
            if detection_model:
                self.model_manager.release_model(detection_model)
            if ppe_model:
                self.model_manager.release_model(ppe_model)
    
    def _process_batch(self, detection_model, ppe_model, frames, metadata):
        """Process a batch of frames through AI models"""
        try:
            # Run person detection
            detection_results = detection_model.model(frames, conf=DatacenterConfig.PERSON_DETECTION_CONFIDENCE)
            
            # Run PPE detection if enabled
            ppe_results = None
            if ppe_model and DatacenterConfig.PPE_DETECTION_ENABLED:
                ppe_results = ppe_model.model(frames, conf=DatacenterConfig.PPE_CONFIDENCE_THRESHOLD)
            
            # Process results for each frame
            for i, (frame, frame_metadata) in enumerate(zip(frames, metadata)):
                camera_id = frame_metadata['camera_id']
                timestamp = frame_metadata['timestamp']
                
                # Prepare enhanced metadata
                enhanced_metadata = {
                    'detection_result': detection_results[i],
                    'ppe_result': ppe_results[i] if ppe_results else None,
                    'batch_index': i,
                    'batch_size': len(frames),
                    'cross_camera_coordination': camera_id in self.datacenter_coordinators
                }
                
                # Route to camera
                self.camera_manager.route_result_to_camera(
                    camera_id=camera_id,
                    frame=frame,
                    result=detection_results[i],
                    timestamp=timestamp,
                    enhanced_metadata=enhanced_metadata
                )
                
        except Exception as e:
            self.logger.error(f"Error processing batch: {str(e)}", exc_info=True)
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        try:
            self.logger.info("Stopping datacenter monitoring system")
            
            # Stop batch processing
            self.batch_processing_running = False
            if self.batch_processing_thread and self.batch_processing_thread.is_alive():
                self.batch_processing_thread.join(timeout=5.0)
            
            # Shutdown thread pool
            if self.processing_executor:
                self.processing_executor.shutdown(wait=True)
            
            # Stop camera manager
            self.camera_manager.stop_all_cameras()
            
            # Cleanup camera models
            for camera_model in self.camera_models.values():
                if hasattr(camera_model, 'cleanup'):
                    camera_model.cleanup()
            
            # Stop storage and database components
            self.storage_handler.stop_upload_thread()
            self.db_writer.shutdown()
            
            self.logger.info("Monitoring system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {str(e)}", exc_info=True)
    
    def get_system_stats(self) -> Dict:
        """Get current system statistics"""
        try:
            current_time = time.time()
            uptime = current_time - self.system_stats['start_time']
            
            # Calculate average FPS
            avg_fps = self.system_stats['total_frames_processed'] / uptime if uptime > 0 else 0
            
            # Get camera statistics
            camera_stats = {}
            for camera_id, camera_info in self.camera_metadata.items():
                camera_stats[camera_id] = {
                    'frames_processed': camera_info['frames_processed'],
                    'events_detected': camera_info['events_detected'],
                    'last_processed': camera_info['last_processed'],
                    'camera_type': camera_info['camera_type'],
                    'datacenter_name': camera_info['datacenter_name']
                }
            
            return {
                'system': {
                    'uptime_seconds': uptime,
                    'total_frames_processed': self.system_stats['total_frames_processed'],
                    'total_events_detected': self.system_stats['total_events_detected'],
                    'cameras_active': self.system_stats['cameras_active'],
                    'average_fps': avg_fps,
                    'batch_processing_active': self.batch_processing_running
                },
                'cameras': camera_stats,
                'coordinators': {
                    dc_id: coordinator.get_stats() 
                    for dc_id, coordinator in self.datacenter_coordinators.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system stats: {str(e)}")
            return {}
    
    def get_health_status(self) -> Dict:
        """Get system health status"""
        try:
            health = {
                'status': 'healthy',
                'components': {},
                'issues': []
            }
            
            # Check camera manager
            camera_health = self.camera_manager.get_health_status()
            health['components']['camera_manager'] = camera_health
            
            # Check model manager
            model_health = self.model_manager.get_health_status()
            health['components']['model_manager'] = model_health
            
            # Check database
            try:
                self.database.execute_query("SELECT 1")
                health['components']['database'] = 'healthy'
            except Exception as e:
                health['components']['database'] = 'unhealthy'
                health['issues'].append(f"Database: {str(e)}")
            
            # Check batch processing
            if not self.batch_processing_running:
                health['components']['batch_processing'] = 'stopped'
                health['issues'].append("Batch processing not running")
            else:
                health['components']['batch_processing'] = 'running'
            
            # Determine overall status
            if health['issues']:
                health['status'] = 'degraded' if len(health['issues']) < 3 else 'unhealthy'
            
            return health
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'components': {},
                'issues': [f"Health check failed: {str(e)}"]
            }

# Export the main class
__all__ = ['DatacenterVideoProcessor']