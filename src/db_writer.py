#!/usr/bin/env python3
"""
Script 13: db_writer.py
File Path: src/db_writer.py

Datacenter Monitoring System - Database Writing and Batch Operations

This module handles:
1. Batch writing of events, frames, and metadata to database
2. Cloud storage upload coordination
3. Audit trail logging for compliance
4. Performance optimization for high-volume writes
5. Error handling and retry mechanisms
6. Data validation and sanitization
"""

import json
import time
import numpy as np
import mysql.connector
from typing import Dict, List, Any, Optional
from queue import Queue, Empty
from datetime import datetime
from threading import Thread
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import os
import uuid

from config import DatacenterConfig
from database import DatacenterDatabase
from logger import setup_datacenter_logger, audit_logger, performance_logger

try:
    from google.cloud import storage
    from google.oauth2 import service_account
    gcp_available = True
except ImportError:
    gcp_available = False

@dataclass
class DatacenterProcessedFrame:
    """Data class for processed frame with datacenter context"""
    camera_id: int
    datacenter_id: int
    event_id: str
    timestamp: datetime
    local_path: str
    frame_path: str  # Will be set after cloud upload
    event_type: str = "general"
    severity: str = "low"
    zone_name: str = "unknown"

@dataclass
class DatacenterEvent:
    """Data class for datacenter security events"""
    event_id: str
    camera_id: int
    datacenter_id: int
    event_type: str
    severity: str
    timestamp: datetime
    zone_name: Optional[str] = None
    detection_data: Optional[Dict] = None
    snapshot_url: Optional[str] = None
    video_clip_url: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass
class DatacenterVideoClip:
    """Data class for datacenter video clips"""
    event_id: str
    camera_id: int
    datacenter_id: int
    local_path: str
    cloud_path: str
    duration: float
    timestamp: datetime
    event_type: str
    metadata: Dict

class DatacenterDatabaseWriter:
    """Database writer optimized for datacenter monitoring workloads"""
    
    def __init__(self):
        self.logger = setup_datacenter_logger('datacenter_db_writer', 'datacenter_db_writer.log')
        self.logger.info("Initializing DatacenterDatabaseWriter")
        
        # Batch configuration
        self.batch_size = DatacenterConfig.DB_WRITER_BATCH_SIZE
        self.max_queue_size = 2000  # Prevent memory issues
        
        # Processing queues
        self.frame_queue = Queue(maxsize=self.max_queue_size)
        self.event_queue = Queue(maxsize=self.max_queue_size)
        self.video_queue = Queue(maxsize=self.max_queue_size)
        self.access_log_queue = Queue(maxsize=self.max_queue_size)
        
        # Database connection
        self.db = DatacenterDatabase()
        
        # Cloud storage client
        self.storage_client = None
        self.bucket = None
        self._init_cloud_storage()
        
        # Worker threads
        self.running = True
        self.worker_threads = []
        self.max_retries = 3
        self.retry_delay = 2
        
        # Thread pool for parallel uploads
        self.upload_executor = ThreadPoolExecutor(max_workers=self.batch_size)
        
        # Performance tracking
        self.stats = {
            'frames_processed': 0,
            'events_processed': 0,
            'videos_processed': 0,
            'batch_operations': 0,
            'upload_errors': 0,
            'database_errors': 0
        }
        
        # Start worker threads
        self._start_worker_threads()
        
        self.logger.info("DatacenterDatabaseWriter initialized successfully")
    
    def _init_cloud_storage(self):
        """Initialize Google Cloud Storage client"""
        if not gcp_available:
            self.logger.warning("Google Cloud Storage not available")
            return
        
        try:
            credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            if credentials_path and os.path.exists(credentials_path):
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                self.storage_client = storage.Client(
                    credentials=credentials,
                    project=DatacenterConfig.GCP_PROJECT
                )
            else:
                self.storage_client = storage.Client()
            
            if DatacenterConfig.BUCKET_NAME:
                self.bucket = self.storage_client.bucket(DatacenterConfig.BUCKET_NAME)
                self.logger.info(f"Connected to GCS bucket: {DatacenterConfig.BUCKET_NAME}")
            else:
                self.logger.warning("No bucket name configured")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize cloud storage: {str(e)}")
    
    def _start_worker_threads(self):
        """Start worker threads for different data types"""
        
        # Frame processing thread
        frame_thread = Thread(
            target=self._process_frame_queue,
            daemon=True,
            name="frame_processor"
        )
        frame_thread.start()
        self.worker_threads.append(frame_thread)
        
        # Event processing thread
        event_thread = Thread(
            target=self._process_event_queue,
            daemon=True,
            name="event_processor"
        )
        event_thread.start()
        self.worker_threads.append(event_thread)
        
        # Video processing thread
        video_thread = Thread(
            target=self._process_video_queue,
            daemon=True,
            name="video_processor"
        )
        video_thread.start()
        self.worker_threads.append(video_thread)
        
        # Access log processing thread
        access_thread = Thread(
            target=self._process_access_log_queue,
            daemon=True,
            name="access_log_processor"
        )
        access_thread.start()
        self.worker_threads.append(access_thread)
        
        self.logger.info(f"Started {len(self.worker_threads)} worker threads")
    
    def queue_frame(self, frame_data: DatacenterProcessedFrame):
        """Queue a frame for processing"""
        try:
            if self.frame_queue.full():
                self.logger.warning("Frame queue full, dropping frame")
                return False
            
            self.frame_queue.put(frame_data)
            return True
            
        except Exception as e:
            self.logger.error(f"Error queueing frame: {str(e)}")
            return False
    
    def queue_event(self, event_data: DatacenterEvent):
        """Queue an event for processing"""
        try:
            if self.event_queue.full():
                self.logger.warning("Event queue full, dropping event")
                return False
            
            self.event_queue.put(event_data)
            return True
            
        except Exception as e:
            self.logger.error(f"Error queueing event: {str(e)}")
            return False
    
    def queue_video(self, video_data: DatacenterVideoClip):
        """Queue a video clip for processing"""
        try:
            if self.video_queue.full():
                self.logger.warning("Video queue full, dropping video")
                return False
            
            self.video_queue.put(video_data)
            return True
            
        except Exception as e:
            self.logger.error(f"Error queueing video: {str(e)}")
            return False
    
    def queue_access_log(self, access_data: Dict):
        """Queue an access log entry"""
        try:
            if self.access_log_queue.full():
                self.logger.warning("Access log queue full, dropping entry")
                return False
            
            self.access_log_queue.put(access_data)
            return True
            
        except Exception as e:
            self.logger.error(f"Error queueing access log: {str(e)}")
            return False
    
    def _process_frame_queue(self):
        """Process frame queue in batches"""
        self.logger.info("Frame processor started")
        
        while self.running:
            try:
                frames_batch = []
                
                # Collect frames for batch processing
                try:
                    # Get first frame with timeout
                    frame = self.frame_queue.get(timeout=1.0)
                    frames_batch.append(frame)
                    
                    # Collect more frames without blocking
                    while len(frames_batch) < self.batch_size:
                        try:
                            frame = self.frame_queue.get_nowait()
                            frames_batch.append(frame)
                        except Empty:
                            break
                            
                except Empty:
                    continue
                
                if frames_batch:
                    success = self._process_frames_batch(frames_batch)
                    if success:
                        self.stats['frames_processed'] += len(frames_batch)
                        self.stats['batch_operations'] += 1
                        
                        # Mark tasks as done
                        for _ in frames_batch:
                            self.frame_queue.task_done()
                    else:
                        self.stats['database_errors'] += 1
                        
            except Exception as e:
                self.logger.error(f"Error in frame processor: {str(e)}", exc_info=True)
                time.sleep(1)
        
        self.logger.info("Frame processor stopped")
    
    def _process_frames_batch(self, frames_batch: List[DatacenterProcessedFrame]) -> bool:
        """Process a batch of frames"""
        try:
            # Upload frames to cloud storage in parallel
            upload_tasks = []
            
            for frame in frames_batch:
                future = self.upload_executor.submit(
                    self._upload_frame_to_cloud,
                    frame.local_path,
                    frame.camera_id,
                    frame.datacenter_id,
                    frame.event_type,
                    frame.timestamp
                )
                upload_tasks.append((frame, future))
            
            # Wait for uploads and update frame paths
            successful_frames = []
            for frame, future in upload_tasks:
                try:
                    cloud_path = future.result(timeout=30)
                    if cloud_path:
                        frame.frame_path = cloud_path
                        successful_frames.append(frame)
                        
                        # Clean up local file
                        try:
                            os.remove(frame.local_path)
                        except Exception as e:
                            self.logger.warning(f"Failed to delete local file: {e}")
                    else:
                        self.stats['upload_errors'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Frame upload failed: {str(e)}")
                    self.stats['upload_errors'] += 1
            
            # Update database with successful uploads
            if successful_frames:
                self._update_events_with_snapshots(successful_frames)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing frames batch: {str(e)}", exc_info=True)
            return False
    
    def _upload_frame_to_cloud(self, local_path: str, camera_id: int, datacenter_id: int, 
                              event_type: str, timestamp: datetime) -> Optional[str]:
        """Upload a single frame to cloud storage"""
        try:
            if not self.bucket or not os.path.exists(local_path):
                return None
            
            # Generate cloud path
            date_str = timestamp.strftime('%Y/%m/%d')
            filename = os.path.basename(local_path)
            cloud_path = f"datacenters/{datacenter_id}/cameras/{camera_id}/events/{event_type}/{date_str}/{filename}"
            
            # Upload file
            blob = self.bucket.blob(cloud_path)
            blob.upload_from_filename(local_path)
            
            # Set metadata
            blob.metadata = {
                'camera_id': str(camera_id),
                'datacenter_id': str(datacenter_id),
                'event_type': event_type,
                'timestamp': timestamp.isoformat(),
                'upload_time': datetime.utcnow().isoformat()
            }
            blob.patch()
            
            return cloud_path
            
        except Exception as e:
            self.logger.error(f"Failed to upload frame to cloud: {str(e)}")
            return None
    
    def _update_events_with_snapshots(self, frames: List[DatacenterProcessedFrame]):
        """Update events with snapshot URLs"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                for frame in frames:
                    try:
                        # Update event with snapshot URL
                        cursor.execute("""
                            UPDATE events 
                            SET snapshot_url = %s
                            WHERE event_id = %s
                        """, (frame.frame_path, frame.event_id))
                        
                        # Log audit trail
                        audit_logger.log_system_event(
                            component='db_writer',
                            event='snapshot_updated',
                            status='success',
                            details={
                                'event_id': frame.event_id,
                                'camera_id': frame.camera_id,
                                'snapshot_url': frame.frame_path
                            }
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Failed to update event {frame.event_id}: {e}")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error updating events with snapshots: {str(e)}")
    
    def _process_event_queue(self):
        """Process event queue in batches"""
        self.logger.info("Event processor started")
        
        while self.running:
            try:
                events_batch = []
                
                # Collect events for batch processing
                try:
                    event = self.event_queue.get(timeout=1.0)
                    events_batch.append(event)
                    
                    while len(events_batch) < self.batch_size:
                        try:
                            event = self.event_queue.get_nowait()
                            events_batch.append(event)
                        except Empty:
                            break
                            
                except Empty:
                    continue
                
                if events_batch:
                    success = self._process_events_batch(events_batch)
                    if success:
                        self.stats['events_processed'] += len(events_batch)
                        self.stats['batch_operations'] += 1
                        
                        for _ in events_batch:
                            self.event_queue.task_done()
                    else:
                        self.stats['database_errors'] += 1
                        
            except Exception as e:
                self.logger.error(f"Error in event processor: {str(e)}", exc_info=True)
                time.sleep(1)
        
        self.logger.info("Event processor stopped")
    
    def _process_events_batch(self, events_batch: List[DatacenterEvent]) -> bool:
        """Process a batch of events"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                for event in events_batch:
                    try:
                        # Find matching rule
                        cursor.execute("""
                            SELECT rule_id, severity FROM rules 
                            WHERE camera_id = %s AND event_type = %s AND enabled = TRUE
                            LIMIT 1
                        """, (event.camera_id, event.event_type))
                        
                        rule_result = cursor.fetchone()
                        if not rule_result:
                            self.logger.warning(f"No rule found for {event.event_type} on camera {event.camera_id}")
                            continue
                        
                        rule_id, rule_severity = rule_result
                        
                        # Find zone if specified
                        zone_id = None
                        if event.zone_name:
                            cursor.execute("""
                                SELECT zone_id FROM zones 
                                WHERE camera_id = %s AND name = %s 
                                LIMIT 1
                            """, (event.camera_id, event.zone_name))
                            
                            zone_result = cursor.fetchone()
                            if zone_result:
                                zone_id = zone_result[0]
                        
                        # Insert event
                        cursor.execute("""
                            INSERT INTO events (
                                event_id, rule_id, camera_id, zone_id, timestamp, 
                                event_type, severity, detection_data, snapshot_url, 
                                video_clip_url, status
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                            )
                        """, (
                            event.event_id,
                            rule_id,
                            event.camera_id,
                            zone_id,
                            event.timestamp,
                            event.event_type,
                            event.severity,
                            json.dumps(event.detection_data) if event.detection_data else None,
                            event.snapshot_url,
                            event.video_clip_url,
                            'new'
                        ))
                        
                        # Log audit trail
                        audit_logger.log_event_detection(
                            event_type=event.event_type,
                            camera_id=str(event.camera_id),
                            datacenter_id=str(event.datacenter_id),
                            severity=event.severity,
                            detection_data=event.detection_data or {}
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process event {event.event_id}: {e}")
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Error processing events batch: {str(e)}", exc_info=True)
            return False
    
    def _process_video_queue(self):
        """Process video queue"""
        self.logger.info("Video processor started")
        
        while self.running:
            try:
                try:
                    video = self.video_queue.get(timeout=1.0)
                    
                    success = self._process_video(video)
                    if success:
                        self.stats['videos_processed'] += 1
                    else:
                        self.stats['upload_errors'] += 1
                    
                    self.video_queue.task_done()
                    
                except Empty:
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error in video processor: {str(e)}", exc_info=True)
                time.sleep(1)
        
        self.logger.info("Video processor stopped")
    
    def _process_video(self, video: DatacenterVideoClip) -> bool:
        """Process a single video clip"""
        try:
            # Upload to cloud storage
            cloud_path = self._upload_video_to_cloud(video)
            if not cloud_path:
                return False
            
            # Update database
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE events 
                    SET video_clip_url = %s,
                        detection_data = JSON_SET(
                            COALESCE(detection_data, '{}'),
                            '$.video_duration', %s,
                            '$.video_path', %s
                        )
                    WHERE event_id = %s
                """, (cloud_path, video.duration, video.local_path, video.event_id))
                
                conn.commit()
            
            # Clean up local file
            try:
                os.remove(video.local_path)
            except Exception as e:
                self.logger.warning(f"Failed to delete video file: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            return False
    
    def _upload_video_to_cloud(self, video: DatacenterVideoClip) -> Optional[str]:
        """Upload video to cloud storage"""
        try:
            if not self.bucket or not os.path.exists(video.local_path):
                return None
            
            # Generate cloud path
            date_str = video.timestamp.strftime('%Y/%m/%d')
            filename = os.path.basename(video.local_path)
            cloud_path = f"datacenters/{video.datacenter_id}/cameras/{video.camera_id}/videos/{video.event_type}/{date_str}/{filename}"
            
            # Upload file
            blob = self.bucket.blob(cloud_path)
            blob.upload_from_filename(video.local_path)
            
            # Set metadata
            blob.metadata = {
                'camera_id': str(video.camera_id),
                'datacenter_id': str(video.datacenter_id),
                'event_type': video.event_type,
                'duration': str(video.duration),
                'timestamp': video.timestamp.isoformat(),
                'upload_time': datetime.utcnow().isoformat()
            }
            blob.patch()
            
            return cloud_path
            
        except Exception as e:
            self.logger.error(f"Failed to upload video to cloud: {str(e)}")
            return None
    
    def _process_access_log_queue(self):
        """Process access log queue"""
        self.logger.info("Access log processor started")
        
        while self.running:
            try:
                access_logs = []
                
                # Collect access logs
                try:
                    log = self.access_log_queue.get(timeout=1.0)
                    access_logs.append(log)
                    
                    while len(access_logs) < self.batch_size * 2:  # Larger batch for logs
                        try:
                            log = self.access_log_queue.get_nowait()
                            access_logs.append(log)
                        except Empty:
                            break
                            
                except Empty:
                    continue
                
                if access_logs:
                    self._process_access_logs_batch(access_logs)
                    
                    for _ in access_logs:
                        self.access_log_queue.task_done()
                        
            except Exception as e:
                self.logger.error(f"Error in access log processor: {str(e)}", exc_info=True)
                time.sleep(1)
        
        self.logger.info("Access log processor stopped")
    
    def _process_access_logs_batch(self, access_logs: List[Dict]):
        """Process batch of access logs"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                for log in access_logs:
                    try:
                        cursor.execute("""
                            INSERT INTO access_logs (
                                datacenter_id, camera_id, zone_id, person_id, 
                                access_type, timestamp, detection_confidence, 
                                additional_data
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s
                            )
                        """, (
                            log.get('datacenter_id'),
                            log.get('camera_id'),
                            log.get('zone_id'),
                            log.get('person_id'),
                            log.get('access_type'),
                            log.get('timestamp'),
                            log.get('detection_confidence'),
                            json.dumps(log.get('additional_data', {}))
                        ))
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process access log: {e}")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error processing access logs batch: {str(e)}")
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            'queue_sizes': {
                'frames': self.frame_queue.qsize(),
                'events': self.event_queue.qsize(),
                'videos': self.video_queue.qsize(),
                'access_logs': self.access_log_queue.qsize()
            },
            'processing_stats': self.stats.copy(),
            'worker_threads': len([t for t in self.worker_threads if t.is_alive()])
        }
    
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down DatacenterDatabaseWriter")
        
        self.running = False
        
        # Wait for worker threads to finish
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=30)
                if thread.is_alive():
                    self.logger.warning(f"Thread {thread.name} did not terminate gracefully")
        
        # Shutdown thread pool
        self.upload_executor.shutdown(wait=True)
        
        # Log final stats
        final_stats = self.get_stats()
        self.logger.info(f"Final processing stats: {final_stats}")
        
        self.logger.info("DatacenterDatabaseWriter shutdown complete")

# Export main classes
__all__ = [
    'DatacenterDatabaseWriter',
    'DatacenterProcessedFrame', 
    'DatacenterEvent',
    'DatacenterVideoClip'
]