#!/usr/bin/env python3
"""
Script 8: camera_manager.py
File Path: src/camera_manager.py

Datacenter Monitoring System - Camera Management

This module handles:
1. RTSP camera stream management for datacenter monitoring
2. Multi-camera frame batching for efficient GPU processing
3. Activity-based FPS control for different zone types
4. Camera health monitoring and automatic reconnection
5. Result routing back to individual camera processors
6. Frame queuing and batch collection for datacenter use cases
"""

import cv2
import time
import threading
import queue
import numpy as np
import os
from typing import Dict, List, Tuple, Any, Optional, Callable
from collections import deque
from datetime import datetime

from logger import setup_datacenter_logger
from config import DatacenterConfig

class DatacenterCameraFrame:
    """Data class for a single camera frame with datacenter-specific metadata"""
    
    def __init__(self, camera_id: str, frame: np.ndarray, timestamp: float, frame_number: int,
                 camera_type: str = 'unknown', datacenter_id: Optional[str] = None):
        self.camera_id = camera_id
        self.frame = frame
        self.timestamp = timestamp
        self.frame_number = frame_number
        self.camera_type = camera_type
        self.datacenter_id = datacenter_id
        self.processed_result = None
        self.detection_metadata = {}
        
    def set_processed_result(self, result: Any, metadata: Optional[Dict] = None):
        """Set the processed result for this frame"""
        self.processed_result = result
        if metadata:
            self.detection_metadata.update(metadata)

class DatacenterCameraReader:
    """Manages a single datacenter camera stream with enhanced monitoring capabilities"""
    
    def __init__(self, camera_id: str, stream_url: str, camera_type: str,
                 frame_queue: queue.Queue, result_callback: Optional[Callable] = None,
                 logger=None, target_fps: Optional[int] = None, 
                 datacenter_id: Optional[str] = None):
        
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.camera_type = camera_type
        self.datacenter_id = datacenter_id
        self.frame_queue = frame_queue
        self.result_callback = result_callback
        self.logger = logger or setup_datacenter_logger(
            f'camera_reader_{camera_id}', 
            f'camera_reader_{camera_id}.log',
            camera_id=camera_id,
            datacenter_id=datacenter_id
        )
        
        # Camera status and configuration
        self.running = False
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = DatacenterConfig.MAX_RETRIES
        self.reconnect_delay = 5  # seconds
        
        # Frame processing configuration
        self.frame_count = 0
        self.frames_captured = 0
        self.frames_dropped = 0
        self.last_frame_time = 0
        self.fps = 0
        
        # Activity-based FPS control for datacenter monitoring
        self.target_fps = target_fps or self._get_default_fps_for_camera_type()
        
        # Thread management
        self.reader_thread = None
        self.result_thread = None
        
        # Result processing queue
        self.result_queue = queue.Queue(maxsize=DatacenterConfig.MAX_QUEUE_SIZE)
        
        # Streaming queue for real-time monitoring
        self.streaming_queue = queue.Queue(maxsize=20)
        
        # Camera health monitoring
        self.health_stats = {
            'last_successful_frame': 0,
            'connection_uptime': 0,
            'total_reconnections': 0,
            'average_fps': 0,
            'frame_drop_rate': 0
        }
        
        self.logger.info(f"Datacenter camera reader initialized for {camera_type} camera {camera_id}")
    
    def _get_default_fps_for_camera_type(self) -> int:
        """Get default FPS based on camera type for datacenter monitoring"""
        fps_mapping = {
            'dc_entry_monitor': DatacenterConfig.ACTIVITY_LEVEL_HIGH,     # 10 FPS - High activity
            'dc_critical_zone': DatacenterConfig.ACTIVITY_LEVEL_HIGH,    # 10 FPS - Critical areas
            'dc_server_room': DatacenterConfig.ACTIVITY_LEVEL_MEDIUM,    # 4 FPS - Server rooms
            'dc_perimeter': DatacenterConfig.ACTIVITY_LEVEL_MEDIUM,      # 4 FPS - Perimeter
            'dc_corridor': DatacenterConfig.ACTIVITY_LEVEL_LOW,          # 2 FPS - Corridors
            'dc_common_area': DatacenterConfig.ACTIVITY_LEVEL_LOW        # 2 FPS - Common areas
        }
        return fps_mapping.get(self.camera_type, DatacenterConfig.ACTIVITY_LEVEL_MEDIUM)
    
    def start(self) -> bool:
        """Start the camera reader with result processor"""
        if self.running:
            self.logger.warning(f"Camera {self.camera_id} reader already running")
            return False
            
        self.running = True
        
        # Start camera reader thread
        self.reader_thread = threading.Thread(
            target=self._reader_worker,
            daemon=True,
            name=f"datacenter_camera_reader_{self.camera_id}"
        )
        self.reader_thread.start()
        
        # Start result processor thread
        self.result_thread = threading.Thread(
            target=self._result_processor_worker,
            daemon=True,
            name=f"datacenter_result_processor_{self.camera_id}"
        )
        self.result_thread.start()
        
        self.logger.info(f"Started datacenter camera reader for {self.camera_type} camera {self.camera_id} at {self.target_fps} FPS")
        return True
    
    def stop(self):
        """Stop the camera reader and result processor"""
        self.logger.info(f"Stopping datacenter camera reader for {self.camera_id}")
        self.running = False
        
        # Wait for threads to finish
        for thread, name in [(self.reader_thread, "reader"), (self.result_thread, "result processor")]:
            if thread and thread.is_alive():
                thread.join(timeout=3.0)
                if thread.is_alive():
                    self.logger.warning(f"Camera {self.camera_id} {name} thread did not terminate gracefully")
    
    def _reader_worker(self):
        """Enhanced camera reader worker for datacenter monitoring"""
        self.logger.info(f"Datacenter camera reader worker started for {self.camera_id}")
        
        # Performance tracking
        frames_processed = 0
        frames_read = 0
        frames_dropped = 0
        last_stats_time = time.time()
        connection_start_time = 0
        
        while self.running:
            try:
                # Attempt camera connection
                connection_start = time.time()
                if not self._connect_to_datacenter_camera():
                    time.sleep(self.reconnect_delay)
                    continue
                
                connection_start_time = time.time()
                self.health_stats['total_reconnections'] += 1
                
                # Frame reading loop
                frame_interval = 1.0 / max(1, self.target_fps)
                last_frame_time = time.time()
                
                while self.running and self.is_connected:
                    current_time = time.time()
                    
                    # Periodic statistics logging
                    if current_time - last_stats_time >= 60:
                        self._log_camera_stats(frames_read, frames_processed, frames_dropped)
                        frames_read = frames_processed = frames_dropped = 0
                        last_stats_time = current_time
                    
                    # FPS control
                    time_since_last = current_time - last_frame_time
                    if time_since_last < frame_interval:
                        time.sleep(max(0, frame_interval - time_since_last))
                        continue
                    
                    # Read frame
                    ret, frame = self._read_frame_with_retry()
                    if not ret:
                        self.is_connected = False
                        frames_dropped += 1
                        break
                    
                    frames_read += 1
                    self.frame_count += 1
                    self.frames_captured += 1
                    frame_timestamp = time.time()
                    
                    # Update FPS calculation
                    self.fps = 1.0 / (frame_timestamp - last_frame_time)
                    last_frame_time = frame_timestamp
                    self.last_frame_time = frame_timestamp
                    self.health_stats['last_successful_frame'] = frame_timestamp
                    
                    # Create datacenter frame object
                    datacenter_frame = DatacenterCameraFrame(
                        camera_id=self.camera_id,
                        frame=frame,
                        timestamp=frame_timestamp,
                        frame_number=self.frame_count,
                        camera_type=self.camera_type,
                        datacenter_id=self.datacenter_id
                    )
                    
                    # Queue frame for processing
                    if self.frame_queue.full():
                        self.logger.warning(f"Frame queue full for camera {self.camera_id}, dropping frame")
                        frames_dropped += 1
                        self.frames_dropped += 1
                    else:
                        self.frame_queue.put(datacenter_frame)
                        frames_processed += 1
                    
                    # Add to streaming queue (replace oldest if full)
                    self._add_to_streaming_queue(frame)
                
                # Update connection uptime
                if connection_start_time > 0:
                    self.health_stats['connection_uptime'] += time.time() - connection_start_time
                
                # Connection lost
                if self.running:
                    self.logger.warning(f"Lost connection to datacenter camera {self.camera_id}")
                    self._release_camera()
                
            except Exception as e:
                self.logger.error(f"Error in datacenter camera reader for {self.camera_id}: {str(e)}", exc_info=True)
                self._release_camera()
                time.sleep(self.reconnect_delay)
        
        self._release_camera()
        self.logger.info(f"Datacenter camera reader worker stopped for {self.camera_id}")
    
    def _connect_to_datacenter_camera(self) -> bool:
        """Connect to datacenter camera with enhanced error handling"""
        try:
            connection_start = time.time()
            self.logger.info(f"Connecting to datacenter camera {self.camera_id} ({self.camera_type}) at {self.stream_url}")
            
            # Set OpenCV options for RTSP
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            
            # Open camera connection
            self.cap = cv2.VideoCapture(self.stream_url)
            
            # Set camera properties for datacenter monitoring
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            connection_time = time.time() - connection_start
            
            if not self.cap.isOpened():
                self.reconnect_attempts += 1
                self.logger.warning(f"Failed to connect to datacenter camera {self.camera_id} after {connection_time:.3f}s")
                
                if self.reconnect_attempts >= self.max_reconnect_attempts:
                    self.logger.error(f"Max reconnection attempts reached for datacenter camera {self.camera_id}")
                    self.running = False
                
                return False
            
            # Test frame read
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                self.logger.warning(f"Could not read test frame from datacenter camera {self.camera_id}")
                return False
            
            # Successfully connected
            self.is_connected = True
            self.reconnect_attempts = 0
            
            # Log camera information
            try:
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                source_fps = self.cap.get(cv2.CAP_PROP_FPS)
                
                self.logger.info(f"Connected to datacenter camera {self.camera_id} ({self.camera_type}) - "
                               f"Resolution: {width}x{height}, Source FPS: {source_fps:.2f}, "
                               f"Target FPS: {self.target_fps}, Connection time: {connection_time:.3f}s")
            except:
                self.logger.info(f"Connected to datacenter camera {self.camera_id} in {connection_time:.3f}s")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to datacenter camera {self.camera_id}: {str(e)}", exc_info=True)
            self.reconnect_attempts += 1
            return False
    
    def _read_frame_with_retry(self, max_retries: int = 3) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame with retry logic for datacenter monitoring"""
        for attempt in range(max_retries):
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return True, frame
                else:
                    if attempt < max_retries - 1:
                        time.sleep(0.1)  # Brief pause before retry
            except Exception as e:
                self.logger.warning(f"Frame read attempt {attempt + 1} failed for camera {self.camera_id}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.1)
        
        return False, None
    
    def _add_to_streaming_queue(self, frame: np.ndarray):
        """Add frame to streaming queue for real-time monitoring"""
        try:
            if self.streaming_queue.full():
                # Remove oldest frame
                try:
                    self.streaming_queue.get_nowait()
                    self.streaming_queue.task_done()
                except queue.Empty:
                    pass
            
            self.streaming_queue.put_nowait(frame.copy())
        except Exception as e:
            self.logger.debug(f"Error adding frame to streaming queue: {e}")
    
    def _log_camera_stats(self, frames_read: int, frames_processed: int, frames_dropped: int):
        """Log camera performance statistics"""
        self.logger.info(f"Datacenter camera {self.camera_id} ({self.camera_type}) stats: "
                        f"read {frames_read}, processed {frames_processed}, dropped {frames_dropped}, "
                        f"FPS: {self.fps:.2f}, target: {self.target_fps}")
        
        # Update health stats
        if frames_read > 0:
            self.health_stats['frame_drop_rate'] = frames_dropped / frames_read
        self.health_stats['average_fps'] = self.fps
    
    def _result_processor_worker(self):
        """Process results from GPU detection for datacenter events"""
        self.logger.info(f"Result processor started for datacenter camera {self.camera_id}")
        
        results_processed = 0
        errors = 0
        last_stats_time = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Periodic stats
                if current_time - last_stats_time >= 60:
                    self.logger.info(f"Result processor stats for camera {self.camera_id}: "
                                   f"processed {results_processed}, errors {errors}")
                    results_processed = errors = 0
                    last_stats_time = current_time
                
                # Get result from queue
                try:
                    result_data = self.result_queue.get(timeout=0.5)
                    self.result_queue.task_done()
                except queue.Empty:
                    continue
                
                # Process result
                if self.result_callback:
                    try:
                        # Handle different result formats
                        if len(result_data) == 4:
                            frame, result, timestamp, metadata = result_data
                        else:
                            frame, result, timestamp = result_data
                            metadata = {}
                        
                        # Add datacenter-specific metadata
                        enhanced_metadata = {
                            'camera_type': self.camera_type,
                            'datacenter_id': self.datacenter_id,
                            'processing_timestamp': current_time,
                            **metadata
                        }
                        
                        self.result_callback(self.camera_id, frame, result, timestamp, enhanced_metadata)
                        results_processed += 1
                        
                    except Exception as e:
                        errors += 1
                        self.logger.error(f"Error in result callback for camera {self.camera_id}: {str(e)}")
                
            except Exception as e:
                self.logger.error(f"Error in result processor for camera {self.camera_id}: {str(e)}")
                time.sleep(0.1)
        
        self.logger.info(f"Result processor stopped for datacenter camera {self.camera_id}")
    
    def queue_result(self, frame: np.ndarray, result: Any, timestamp: float, 
                    metadata: Optional[Dict] = None) -> bool:
        """Queue a processed result for this datacenter camera"""
        try:
            if self.result_queue.full():
                self.logger.warning(f"Result queue full for camera {self.camera_id}")
                return False
            
            result_data = (frame, result, timestamp, metadata) if metadata else (frame, result, timestamp)
            self.result_queue.put(result_data)
            return True
            
        except Exception as e:
            self.logger.error(f"Error queueing result for camera {self.camera_id}: {str(e)}")
            return False
    
    def _release_camera(self):
        """Release camera resources"""
        try:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            self.is_connected = False
        except Exception as e:
            self.logger.error(f"Error releasing camera {self.camera_id}: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive camera status for datacenter monitoring"""
        current_time = time.time()
        
        status = {
            'camera_id': self.camera_id,
            'camera_type': self.camera_type,
            'datacenter_id': self.datacenter_id,
            'connected': self.is_connected,
            'frames_captured': self.frames_captured,
            'frames_dropped': self.frames_dropped,
            'last_frame_time': self.last_frame_time,
            'current_fps': self.fps,
            'target_fps': self.target_fps,
            'reconnect_attempts': self.reconnect_attempts,
            'health_stats': self.health_stats.copy()
        }
        
        # Add connection status
        if self.last_frame_time > 0:
            time_since_last_frame = current_time - self.last_frame_time
            if time_since_last_frame > 30:
                status['status'] = 'stalled'
            elif time_since_last_frame > 10:
                status['status'] = 'slow'
            else:
                status['status'] = 'active'
        else:
            status['status'] = 'disconnected'
        
        return status
    
    def update_target_fps(self, new_fps: int):
        """Update target FPS for dynamic adjustment"""
        old_fps = self.target_fps
        self.target_fps = new_fps
        self.logger.info(f"Updated target FPS for camera {self.camera_id} from {old_fps} to {new_fps}")

class DatacenterCameraManager:
    """Enhanced camera manager for datacenter monitoring with multi-camera coordination"""
    
    def __init__(self):
        self.logger = setup_datacenter_logger('datacenter_camera_manager', 'datacenter_camera_manager.log')
        self.logger.info("Initializing Datacenter Camera Manager")
        
        # Log configuration
        self.logger.info("Datacenter monitoring configuration:")
        self.logger.info(f"- Max retries: {DatacenterConfig.MAX_RETRIES}")
        self.logger.info(f"- Batch size: {DatacenterConfig.BATCH_SIZE}")
        self.logger.info(f"- Activity levels - High: {DatacenterConfig.ACTIVITY_LEVEL_HIGH}, "
                        f"Medium: {DatacenterConfig.ACTIVITY_LEVEL_MEDIUM}, Low: {DatacenterConfig.ACTIVITY_LEVEL_LOW}")
        
        # Camera management
        self.camera_readers: Dict[str, DatacenterCameraReader] = {}
        self.camera_queues: Dict[str, queue.Queue] = {}
        
        # Batch processing for GPU efficiency
        self.batch_queue = queue.Queue(maxsize=30)
        self.batch_collection_running = False
        self.batch_collection_threads = []
        self.num_batch_threads = DatacenterConfig.NUM_BATCH_THREADS
        
        # Camera metadata and health monitoring
        self.camera_metadata = {}
        self.datacenter_mappings = {}  # camera_id -> datacenter_id
        
        # Result routing
        self.result_callback = None
        
        self.logger.info(f"Datacenter Camera Manager initialized with {self.num_batch_threads} batch threads")
    
    def set_result_callback(self, callback_fn: Callable):
        """Set callback function for processing results"""
        self.result_callback = callback_fn
        
        # Update existing camera readers
        for camera_reader in self.camera_readers.values():
            camera_reader.result_callback = callback_fn
        
        self.logger.info("Set result callback function for datacenter monitoring")
    
    def start_datacenter_camera(self, camera_id: str, stream_url: str, camera_type: str,
                               activity_level: str = 'medium', datacenter_id: Optional[str] = None) -> bool:
        """Start a datacenter camera with specified monitoring type"""
        
        if camera_id in self.camera_readers:
            self.logger.warning(f"Datacenter camera {camera_id} already started")
            return False
        
        # Determine FPS based on activity level
        target_fps = DatacenterConfig.get_activity_fps(activity_level)
        
        # Create dedicated queue for this camera
        camera_queue = queue.Queue(maxsize=DatacenterConfig.MAX_QUEUE_SIZE)
        self.camera_queues[camera_id] = camera_queue
        
        # Create and start camera reader
        reader = DatacenterCameraReader(
            camera_id=camera_id,
            stream_url=stream_url,
            camera_type=camera_type,
            frame_queue=camera_queue,
            result_callback=self.result_callback,
            logger=self.logger,
            target_fps=target_fps,
            datacenter_id=datacenter_id
        )
        
        success = reader.start()
        
        if success:
            self.camera_readers[camera_id] = reader
            self.datacenter_mappings[camera_id] = datacenter_id
            
            # Store metadata
            self.camera_metadata[camera_id] = {
                'camera_id': camera_id,
                'camera_type': camera_type,
                'datacenter_id': datacenter_id,
                'stream_url': stream_url,
                'activity_level': activity_level,
                'target_fps': target_fps,
                'started_at': time.time()
            }
            
            self.logger.info(f"Started datacenter camera {camera_id} ({camera_type}) at {target_fps} FPS")
        
        return success
    
    def start_datacenter_cameras(self, camera_sources: Dict[str, Tuple[str, str, str]]) -> int:
        """
        Start multiple datacenter cameras
        
        Args:
            camera_sources: Dict mapping camera_id to (stream_url, camera_type, activity_level)
        """
        self.logger.info(f"Starting {len(camera_sources)} datacenter cameras")
        start_time = time.time()
        
        successful_starts = 0
        
        for camera_id, (stream_url, camera_type, activity_level) in camera_sources.items():
            # Extract datacenter_id if available
            datacenter_id = self.camera_metadata.get(camera_id, {}).get('datacenter_id')
            
            success = self.start_datacenter_camera(camera_id, stream_url, camera_type, activity_level, datacenter_id)
            if success:
                successful_starts += 1
        
        # Start batch collection if cameras started successfully
        if successful_starts > 0 and not self.batch_collection_running:
            self.start_batch_collection()
        
        total_time = time.time() - start_time
        self.logger.info(f"Datacenter camera startup complete: {successful_starts}/{len(camera_sources)} "
                        f"cameras started in {total_time:.3f}s")
        
        return successful_starts
    
    def stop_datacenter_camera(self, camera_id: str) -> bool:
        """Stop a specific datacenter camera"""
        if camera_id not in self.camera_readers:
            self.logger.warning(f"Datacenter camera {camera_id} not found")
            return False
        
        reader = self.camera_readers[camera_id]
        reader.stop()
        
        # Cleanup
        del self.camera_readers[camera_id]
        if camera_id in self.camera_queues:
            del self.camera_queues[camera_id]
        if camera_id in self.camera_metadata:
            del self.camera_metadata[camera_id]
        if camera_id in self.datacenter_mappings:
            del self.datacenter_mappings[camera_id]
        
        self.logger.info(f"Stopped datacenter camera {camera_id}")
        return True
    
    def stop_all_cameras(self):
        """Stop all datacenter cameras"""
        self.logger.info(f"Stopping all datacenter cameras ({len(self.camera_readers)})")
        
        camera_ids = list(self.camera_readers.keys())
        for camera_id in camera_ids:
            self.stop_datacenter_camera(camera_id)
        
        self.stop_batch_collection()
        self.logger.info("All datacenter cameras stopped")
    
    def start_batch_collection(self):
        """Start batch collection threads for efficient GPU processing"""
        if self.batch_collection_running:
            return False
        
        self.batch_collection_running = True
        self.batch_collection_threads = []
        
        for i in range(self.num_batch_threads):
            thread = threading.Thread(
                target=self._batch_collection_worker,
                args=(i,),
                daemon=True,
                name=f"datacenter_batch_collection_{i}"
            )
            thread.start()
            self.batch_collection_threads.append(thread)
        
        self.logger.info(f"Started {self.num_batch_threads} batch collection threads for datacenter monitoring")
        return True
    
    def stop_batch_collection(self):
        """Stop batch collection threads"""
        self.logger.info("Stopping datacenter batch collection threads")
        self.batch_collection_running = False
        
        for thread in self.batch_collection_threads:
            if thread and thread.is_alive():
                thread.join(timeout=3.0)
                if thread.is_alive():
                    self.logger.warning(f"Batch collection thread {thread.name} did not terminate gracefully")
        
        self.logger.info("Datacenter batch collection stopped")
    
    def _batch_collection_worker(self, worker_id: int):
        """Batch collection worker for datacenter monitoring"""
        self.logger.info(f"Datacenter batch collection worker {worker_id} started")
        
        pending_frames = []
        last_frame_time = time.time()
        frames_collected = 0
        batches_created = 0
        
        while self.batch_collection_running:
            try:
                current_time = time.time()
                
                # Check for batch completion conditions
                timeout_occurred = (current_time - last_frame_time) >= DatacenterConfig.BATCH_TIMEOUT
                batch_ready = len(pending_frames) >= DatacenterConfig.BATCH_SIZE
                
                if batch_ready or (pending_frames and timeout_occurred):
                    # Create batch
                    batch_size = min(DatacenterConfig.BATCH_SIZE, len(pending_frames))
                    batch_frames = pending_frames[:batch_size]
                    pending_frames = pending_frames[batch_size:]
                    
                    if batch_frames:
                        frames = [frame.frame for frame in batch_frames]
                        metadata = [{
                            'camera_id': frame.camera_id,
                            'camera_type': frame.camera_type,
                            'datacenter_id': frame.datacenter_id,
                            'timestamp': frame.timestamp,
                            'frame_number': frame.frame_number
                        } for frame in batch_frames]
                        
                        # Add to batch queue
                        if not self.batch_queue.full():
                            self.batch_queue.put((frames, metadata))
                            batches_created += 1
                            last_frame_time = current_time
                        
                        camera_types = [f['camera_type'] for f in metadata]
                        self.logger.debug(f"Worker {worker_id} created batch: {len(frames)} frames, "
                                        f"types: {set(camera_types)}")
                
                # Collect frames from camera queues
                frames_added = self._collect_frames_from_cameras(worker_id, pending_frames)
                frames_collected += frames_added
                
                if frames_added == 0:
                    time.sleep(0.01)  # Brief pause if no frames collected
                
            except Exception as e:
                self.logger.error(f"Error in datacenter batch collection worker {worker_id}: {str(e)}")
                time.sleep(0.1)
        
        self.logger.info(f"Datacenter batch collection worker {worker_id} stopped - "
                        f"collected {frames_collected} frames, created {batches_created} batches")
    
    def _collect_frames_from_cameras(self, worker_id: int, pending_frames: List) -> int:
        """Collect frames from camera queues for batch processing"""
        frames_added = 0
        
        # Distribute cameras across workers
        camera_ids = list(self.camera_queues.keys())
        if not camera_ids:
            return 0
        
        # Round-robin assignment
        my_cameras = [cid for i, cid in enumerate(camera_ids) if i % self.num_batch_threads == worker_id]
        
        for camera_id in my_cameras:
            try:
                if camera_id in self.camera_queues and not self.camera_queues[camera_id].empty():
                    frame = self.camera_queues[camera_id].get(block=False)
                    pending_frames.append(frame)
                    self.camera_queues[camera_id].task_done()
                    frames_added += 1
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.warning(f"Error collecting frame from camera {camera_id}: {e}")
        
        return frames_added
    
    def get_batch(self, timeout: float = 1.0) -> Tuple[Optional[List], Optional[List]]:
        """Get a batch of frames for GPU processing"""
        try:
            start_time = time.time()
            batch = self.batch_queue.get(timeout=timeout)
            wait_time = time.time() - start_time
            
            frames, metadata = batch
            if frames and metadata:
                camera_types = [meta['camera_type'] for meta in metadata]
                camera_count = len(set([meta['camera_id'] for meta in metadata]))
                
                self.logger.debug(f"Retrieved batch: {len(frames)} frames from {camera_count} cameras, "
                                f"types: {set(camera_types)}, wait: {wait_time:.3f}s")
            
            self.batch_queue.task_done()
            return batch
            
        except queue.Empty:
            self.logger.debug(f"No batch available after {timeout:.3f}s timeout")
            return None, None
    
    def route_result_to_camera(self, camera_id: str, frame: np.ndarray, result: Any,
                              timestamp: float, enhanced_metadata: Optional[Dict] = None) -> bool:
        """Route processed result back to specific datacenter camera"""
        if camera_id not in self.camera_readers:
            self.logger.warning(f"Cannot route result: Datacenter camera {camera_id} not found")
            return False
        
        return self.camera_readers[camera_id].queue_result(frame, result, timestamp, enhanced_metadata)
    
    def get_datacenter_camera_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all datacenter cameras"""
        stats = {}
        
        for camera_id, reader in self.camera_readers.items():
            camera_stats = reader.get_status()
            
            # Add metadata
            if camera_id in self.camera_metadata:
                camera_stats.update(self.camera_metadata[camera_id])
            
            stats[camera_id] = camera_stats
        
        return stats
    
    def update_camera_activity_level(self, camera_id: str, activity_level: str) -> bool:
        """Update activity level and FPS for a datacenter camera"""
        if camera_id not in self.camera_readers:
            self.logger.warning(f"Cannot update activity: Datacenter camera {camera_id} not found")
            return False
        
        # Calculate new FPS
        new_fps = DatacenterConfig.get_activity_fps(activity_level)
        
        # Update camera reader
        reader = self.camera_readers[camera_id]
        old_fps = reader.target_fps
        reader.update_target_fps(new_fps)
        
        # Update metadata
        if camera_id in self.camera_metadata:
            self.camera_metadata[camera_id]['activity_level'] = activity_level
            self.camera_metadata[camera_id]['target_fps'] = new_fps
        
        self.logger.info(f"Updated datacenter camera {camera_id} activity to {activity_level} (FPS: {old_fps} → {new_fps})")
        return True
    
    def get_datacenter_summary(self) -> Dict[str, Any]:
        """Get summary statistics for datacenter monitoring"""
        stats = self.get_datacenter_camera_stats()
        
        summary = {
            'total_cameras': len(stats),
            'active_cameras': sum(1 for s in stats.values() if s.get('status') == 'active'),
            'connected_cameras': sum(1 for s in stats.values() if s.get('connected', False)),
            'camera_types': {},
            'datacenters': {},
            'total_frames_processed': sum(s.get('frames_captured', 0) for s in stats.values()),
            'total_frames_dropped': sum(s.get('frames_dropped', 0) for s in stats.values()),
            'average_fps': 0,
            'batch_queue_size': self.batch_queue.qsize()
        }
        
        # Analyze by camera type
        for camera_stats in stats.values():
            camera_type = camera_stats.get('camera_type', 'unknown')
            if camera_type not in summary['camera_types']:
                summary['camera_types'][camera_type] = {'count': 0, 'active': 0}
            
            summary['camera_types'][camera_type]['count'] += 1
            if camera_stats.get('status') == 'active':
                summary['camera_types'][camera_type]['active'] += 1
        
        # Analyze by datacenter
        for camera_stats in stats.values():
            datacenter_id = camera_stats.get('datacenter_id', 'unknown')
            if datacenter_id not in summary['datacenters']:
                summary['datacenters'][datacenter_id] = {'cameras': 0, 'active': 0}
            
            summary['datacenters'][datacenter_id]['cameras'] += 1
            if camera_stats.get('status') == 'active':
                summary['datacenters'][datacenter_id]['active'] += 1
        
        # Calculate average FPS
        active_cameras = [s for s in stats.values() if s.get('connected', False)]
        if active_cameras:
            summary['average_fps'] = sum(s.get('current_fps', 0) for s in active_cameras) / len(active_cameras)
        
        return summary
    
    def get_camera_health_report(self) -> Dict[str, Any]:
        """Generate health report for all datacenter cameras"""
        stats = self.get_datacenter_camera_stats()
        current_time = time.time()
        
        health_report = {
            'timestamp': current_time,
            'healthy_cameras': [],
            'warning_cameras': [],
            'critical_cameras': [],
            'disconnected_cameras': [],
            'summary': {
                'total': len(stats),
                'healthy': 0,
                'warning': 0,
                'critical': 0,
                'disconnected': 0
            }
        }
        
        for camera_id, camera_stats in stats.items():
            health_status = self._assess_camera_health(camera_stats, current_time)
            
            camera_health = {
                'camera_id': camera_id,
                'camera_type': camera_stats.get('camera_type'),
                'datacenter_id': camera_stats.get('datacenter_id'),
                'status': health_status,
                'current_fps': camera_stats.get('current_fps', 0),
                'target_fps': camera_stats.get('target_fps', 0),
                'frames_dropped': camera_stats.get('frames_dropped', 0),
                'last_frame_age': current_time - camera_stats.get('last_frame_time', 0)
            }
            
            health_report[f'{health_status}_cameras'].append(camera_health)
            health_report['summary'][health_status] += 1
        
        return health_report
    
    def _assess_camera_health(self, camera_stats: Dict, current_time: float) -> str:
        """Assess individual camera health status"""
        if not camera_stats.get('connected', False):
            return 'disconnected'
        
        last_frame_time = camera_stats.get('last_frame_time', 0)
        time_since_last_frame = current_time - last_frame_time
        
        current_fps = camera_stats.get('current_fps', 0)
        target_fps = camera_stats.get('target_fps', 1)
        fps_ratio = current_fps / target_fps if target_fps > 0 else 0
        
        frame_drop_rate = camera_stats.get('health_stats', {}).get('frame_drop_rate', 0)
        
        # Critical conditions
        if time_since_last_frame > 60 or fps_ratio < 0.3 or frame_drop_rate > 0.5:
            return 'critical'
        
        # Warning conditions
        if time_since_last_frame > 15 or fps_ratio < 0.7 or frame_drop_rate > 0.2:
            return 'warning'
        
        return 'healthy'


# Testing functionality
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Datacenter Camera Manager")
    parser.add_argument("--test-cameras", nargs='+', 
                       help="List of test camera URLs",
                       default=["rtsp://demo:demo@demo.com/stream1"])
    parser.add_argument("--duration", type=int, default=30,
                       help="Test duration in seconds")
    
    args = parser.parse_args()
    
    # Create camera manager
    manager = DatacenterCameraManager()
    
    # Test camera sources
    camera_sources = {}
    for i, url in enumerate(args.test_cameras):
        camera_id = f"test_camera_{i+1}"
        camera_type = 'dc_entry_monitor' if i % 2 == 0 else 'dc_server_room'
        activity_level = 'high' if i % 2 == 0 else 'medium'
        camera_sources[camera_id] = (url, camera_type, activity_level)
    
    try:
        print(f"Testing with {len(camera_sources)} datacenter cameras for {args.duration} seconds")
        
        # Start cameras
        started = manager.start_datacenter_cameras(camera_sources)
        print(f"Started {started} cameras")
        
        if started > 0:
            # Monitor for specified duration
            for i in range(args.duration):
                time.sleep(1)
                
                if i % 10 == 0:  # Every 10 seconds
                    summary = manager.get_datacenter_summary()
                    print(f"Time {i}s: {summary['active_cameras']}/{summary['total_cameras']} cameras active, "
                          f"avg FPS: {summary['average_fps']:.1f}, queue: {summary['batch_queue_size']}")
                    
                    # Test batch retrieval
                    frames, metadata = manager.get_batch(timeout=1.0)
                    if frames:
                        print(f"  Retrieved batch with {len(frames)} frames from "
                              f"{len(set(m['camera_id'] for m in metadata))} cameras")
            
            # Final health report
            health_report = manager.get_camera_health_report()
            print(f"\nFinal Health Report:")
            print(f"  Healthy: {health_report['summary']['healthy']}")
            print(f"  Warning: {health_report['summary']['warning']}")
            print(f"  Critical: {health_report['summary']['critical']}")
            print(f"  Disconnected: {health_report['summary']['disconnected']}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        print("Stopping all cameras...")
        manager.stop_all_cameras()
        print("Test complete")#!/usr/bin/env python3
