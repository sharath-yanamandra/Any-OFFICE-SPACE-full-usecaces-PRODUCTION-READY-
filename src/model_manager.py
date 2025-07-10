#!/usr/bin/env python3
"""
Script 9: model_manager.py
File Path: src/model_manager.py

Datacenter Monitoring System - AI Model Management

This module handles:
1. GPU-accelerated AI model loading and management
2. Person detection and PPE detection model coordination
3. Model instance pooling for efficient GPU utilization
4. Memory management and performance optimization
5. Model inference batching for datacenter monitoring
6. Enhanced model capabilities for datacenter use cases
"""

import torch
import threading
import time
import gc
from typing import Dict, List, Optional, Any, Tuple
from ultralytics import YOLO
import numpy as np
import os

from logger import setup_datacenter_logger, performance_logger
from config import DatacenterConfig

class DatacenterModelInstance:
    """Enhanced model instance for datacenter monitoring with additional capabilities"""
    
    def __init__(self, model: YOLO, model_type: str, model_path: str):
        self.model = model
        self.model_type = model_type
        self.model_path = model_path
        self.in_use = False
        self.lock = threading.Lock()
        self.creation_time = time.time()
        self.inference_count = 0
        self.total_inference_time = 0
        self.last_used = time.time()
        
        # Model-specific configurations
        self.confidence_threshold = self._get_confidence_threshold()
        self.supported_classes = self._get_supported_classes()
        
    def _get_confidence_threshold(self) -> float:
        """Get confidence threshold based on model type"""
        thresholds = {
            'person_detection': DatacenterConfig.PERSON_DETECTION_CONFIDENCE,
            'ppe_detection': DatacenterConfig.PPE_CONFIDENCE_THRESHOLD,
            'general_detection': DatacenterConfig.GENERAL_DETECTION_CONFIDENCE
        }
        return thresholds.get(self.model_type, 0.5)
    
    def _get_supported_classes(self) -> List[str]:
        """Get supported classes for this model type"""
        if self.model_type == 'person_detection':
            return ['person']
        elif self.model_type == 'ppe_detection':
            return ['hard_hat', 'safety_vest', 'safety_glasses', 'gloves', 'boots']
        else:
            return list(self.model.names.values()) if hasattr(self.model, 'names') else []
    
    def update_usage_stats(self, inference_time: float):
        """Update model usage statistics"""
        self.inference_count += 1
        self.total_inference_time += inference_time
        self.last_used = time.time()
    
    def get_average_inference_time(self) -> float:
        """Get average inference time for this model instance"""
        if self.inference_count == 0:
            return 0.0
        return self.total_inference_time / self.inference_count
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        uptime = time.time() - self.creation_time
        time_since_last_use = time.time() - self.last_used
        
        return {
            'model_type': self.model_type,
            'inference_count': self.inference_count,
            'average_inference_time': self.get_average_inference_time(),
            'total_inference_time': self.total_inference_time,
            'uptime_seconds': uptime,
            'time_since_last_use': time_since_last_use,
            'currently_in_use': self.in_use,
            'confidence_threshold': self.confidence_threshold,
            'supported_classes': len(self.supported_classes)
        }

class DatacenterModelManager:
    """Enhanced model manager for datacenter monitoring with GPU optimization"""
    
    def __init__(self, memory_threshold: float = 0.85):
        self.logger = setup_datacenter_logger('datacenter_model_manager', 'datacenter_model_manager.log')
        self.logger.info("Initializing Datacenter Model Manager")
        
        # GPU and memory management
        self.memory_threshold = memory_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.device == 'cuda':
            self.total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
            self.logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"Total GPU memory: {self.total_gpu_memory / 1e9:.2f} GB")
        else:
            self.logger.warning("No CUDA GPU detected, falling back to CPU")
            self.total_gpu_memory = 0
        
        # Model instance pools
        self.model_instances: Dict[str, List[DatacenterModelInstance]] = {
            'person_detection': [],
            'ppe_detection': [],
            'general_detection': []
        }
        
        # Model locks for thread safety
        self.model_locks = {
            'person_detection': threading.Lock(),
            'ppe_detection': threading.Lock(),
            'general_detection': threading.Lock()
        }
        
        # Performance monitoring
        self.performance_stats = {
            'total_inferences': 0,
            'total_inference_time': 0,
            'gpu_memory_peak': 0,
            'model_load_time': 0
        }
        
        # Initialize models
        self._initialize_datacenter_models()
        
        self.logger.info("Datacenter Model Manager initialization complete")
    
    def _initialize_datacenter_models(self):
        """Initialize AI models for datacenter monitoring"""
        start_time = time.time()
        
        try:
            if not torch.cuda.is_available():
                self.logger.error("CUDA not available - datacenter monitoring requires GPU acceleration")
                raise RuntimeError("GPU acceleration required for datacenter monitoring")
            
            with torch.cuda.device(0):
                # Initialize person detection model
                self._load_person_detection_model()
                
                # Initialize PPE detection model if enabled
                if DatacenterConfig.PPE_DETECTION_ENABLED:
                    self._load_ppe_detection_model()
                
                # Initialize general detection model
                self._load_general_detection_model()
            
            load_time = time.time() - start_time
            self.performance_stats['model_load_time'] = load_time
            
            # Log memory usage after model loading
            self._log_gpu_memory_usage("after model initialization")
            
            self.logger.info(f"All datacenter models loaded successfully in {load_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize datacenter models: {str(e)}", exc_info=True)
            raise
    
    def _load_person_detection_model(self):
        """Load person detection model optimized for datacenter monitoring"""
        try:
            self.logger.info("Loading person detection model for datacenter monitoring")
            
            # Load YOLO model for person detection
            person_model = YOLO(DatacenterConfig.DETECTION_MODEL_PATH)
            person_model.to(self.device)
            person_model.model.eval()
            
            # Create model instance
            instance = DatacenterModelInstance(person_model, 'person_detection', DatacenterConfig.DETECTION_MODEL_PATH)
            self.model_instances['person_detection'].append(instance)
            
            # Verify model capabilities
            device = next(person_model.model.parameters()).device
            self.logger.info(f"Person detection model loaded on {device}")
            self.logger.info(f"Model classes: {len(person_model.names)} total, focusing on 'person' class")
            
        except Exception as e:
            self.logger.error(f"Failed to load person detection model: {str(e)}")
            raise
    
    def _load_ppe_detection_model(self):
        """Load PPE detection model for safety compliance monitoring"""
        try:
            self.logger.info("Loading PPE detection model for safety monitoring")
            
            # Load PPE-specific model if available, otherwise use general model
            model_path = getattr(DatacenterConfig, 'PPE_DETECTION_MODEL_PATH', DatacenterConfig.DETECTION_MODEL_PATH)
            
            ppe_model = YOLO(model_path)
            ppe_model.to(self.device)
            ppe_model.model.eval()
            
            # Create model instance
            instance = DatacenterModelInstance(ppe_model, 'ppe_detection', model_path)
            self.model_instances['ppe_detection'].append(instance)
            
            device = next(ppe_model.model.parameters()).device
            self.logger.info(f"PPE detection model loaded on {device}")
            self.logger.info(f"PPE classes supported: {instance.supported_classes}")
            
        except Exception as e:
            self.logger.error(f"Failed to load PPE detection model: {str(e)}")
            # PPE detection is optional, so don't raise
            self.logger.warning("PPE detection will be disabled")
    
    def _load_general_detection_model(self):
        """Load general object detection model for comprehensive monitoring"""
        try:
            self.logger.info("Loading general detection model")
            
            general_model = YOLO(DatacenterConfig.DETECTION_MODEL_PATH)
            general_model.to(self.device)
            general_model.model.eval()
            
            # Create model instance
            instance = DatacenterModelInstance(general_model, 'general_detection', DatacenterConfig.DETECTION_MODEL_PATH)
            self.model_instances['general_detection'].append(instance)
            
            device = next(general_model.model.parameters()).device
            self.logger.info(f"General detection model loaded on {device}")
            self.logger.info(f"General detection classes: {len(general_model.names)}")
            
        except Exception as e:
            self.logger.error(f"Failed to load general detection model: {str(e)}")
            raise
    
    def get_model_instance(self, model_type: str, timeout: float = 30.0) -> Optional[DatacenterModelInstance]:
        """Get an available model instance with timeout"""
        if model_type not in self.model_instances:
            self.logger.error(f"Unknown model type: {model_type}")
            return None
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.model_locks[model_type]:
                for instance in self.model_instances[model_type]:
                    with instance.lock:
                        if not instance.in_use:
                            instance.in_use = True
                            
                            # Log GPU stats when model is acquired
                            self._log_gpu_memory_usage(f"acquired {model_type} model")
                            
                            # Verify model is on correct device
                            device = next(instance.model.parameters()).device
                            if str(device) != self.device:
                                self.logger.warning(f"Model device mismatch: expected {self.device}, got {device}")
                            
                            self.logger.debug(f"Acquired {model_type} model instance")
                            return instance
            
            # If no instance available, wait briefly
            time.sleep(0.1)
        
        self.logger.warning(f"Timeout waiting for {model_type} model instance")
        return None
    
    def release_model_instance(self, instance: DatacenterModelInstance):
        """Release a model instance back to the pool with memory management"""
        try:
            with instance.lock:
                # Clear any cached computations
                if hasattr(instance.model, 'predictor') and instance.model.predictor:
                    # Clear predictor cache if it exists
                    instance.model.predictor.seen = 0
                
                # Check memory usage and clean up if needed
                current_usage = self._get_memory_usage()
                if current_usage > self.memory_threshold:
                    self.logger.warning(f"High GPU memory usage ({current_usage:.2%}), performing cleanup")
                    self._cleanup_gpu_memory()
                
                # Mark as available
                instance.in_use = False
                
                self.logger.debug(f"Released {instance.model_type} model instance")
                
        except Exception as e:
            self.logger.error(f"Error releasing model instance: {str(e)}", exc_info=True)
    
    def run_person_detection(self, frames: List[np.ndarray], confidence: Optional[float] = None) -> List[Any]:
        """Run person detection on batch of frames"""
        model_instance = self.get_model_instance('person_detection')
        if not model_instance:
            self.logger.error("No person detection model available")
            return []
        
        try:
            start_time = time.time()
            
            # Set confidence threshold
            conf_threshold = confidence or model_instance.confidence_threshold
            
            # Run inference
            results = model_instance.model(frames, conf=conf_threshold, classes=[0])  # Only person class
            
            inference_time = time.time() - start_time
            
            # Update statistics
            model_instance.update_usage_stats(inference_time)
            self.performance_stats['total_inferences'] += 1
            self.performance_stats['total_inference_time'] += inference_time
            
            # Log performance metrics
            performance_logger.log_detection_stats(
                model_type='person_detection',
                inference_time=inference_time,
                detections_count=sum(len(r.boxes) if r.boxes else 0 for r in results),
                confidence_avg=conf_threshold
            )
            
            self.logger.debug(f"Person detection completed: {len(frames)} frames in {inference_time:.3f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in person detection: {str(e)}", exc_info=True)
            return []
        finally:
            self.release_model_instance(model_instance)
    
    def run_ppe_detection(self, frames: List[np.ndarray], confidence: Optional[float] = None) -> List[Any]:
        """Run PPE detection on batch of frames"""
        if not DatacenterConfig.PPE_DETECTION_ENABLED:
            self.logger.debug("PPE detection disabled")
            return []
        
        model_instance = self.get_model_instance('ppe_detection')
        if not model_instance:
            self.logger.warning("No PPE detection model available")
            return []
        
        try:
            start_time = time.time()
            
            # Set confidence threshold
            conf_threshold = confidence or model_instance.confidence_threshold
            
            # Run inference for PPE classes
            results = model_instance.model(frames, conf=conf_threshold)
            
            inference_time = time.time() - start_time
            
            # Update statistics
            model_instance.update_usage_stats(inference_time)
            
            # Log performance metrics
            performance_logger.log_detection_stats(
                model_type='ppe_detection',
                inference_time=inference_time,
                detections_count=sum(len(r.boxes) if r.boxes else 0 for r in results),
                confidence_avg=conf_threshold
            )
            
            self.logger.debug(f"PPE detection completed: {len(frames)} frames in {inference_time:.3f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in PPE detection: {str(e)}", exc_info=True)
            return []
        finally:
            self.release_model_instance(model_instance)
    
    def run_combined_detection(self, frames: List[np.ndarray]) -> Tuple[List[Any], List[Any]]:
        """Run both person and PPE detection efficiently"""
        # Run person detection
        person_results = self.run_person_detection(frames)
        
        # Run PPE detection only if enabled and persons detected
        ppe_results = []
        if DatacenterConfig.PPE_DETECTION_ENABLED:
            # Check if any frames have person detections
            has_persons = any(len(r.boxes) > 0 if r.boxes else False for r in person_results)
            
            if has_persons:
                ppe_results = self.run_ppe_detection(frames)
            else:
                self.logger.debug("No persons detected, skipping PPE detection")
                ppe_results = [None] * len(frames)
        
        return person_results, ppe_results
    
    def detect_camera_tampering(self, current_frame: np.ndarray, reference_frame: np.ndarray) -> Dict[str, Any]:
        """Detect camera tampering using frame comparison"""
        try:
            if not DatacenterConfig.TAMPER_DETECTION_ENABLED:
                return {'tamper_detected': False, 'reason': 'tamper_detection_disabled'}
            
            # Convert to grayscale for comparison
            gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY) if len(current_frame.shape) == 3 else current_frame
            gray_reference = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY) if len(reference_frame.shape) == 3 else reference_frame
            
            # Resize frames to same size if needed
            if gray_current.shape != gray_reference.shape:
                gray_reference = cv2.resize(gray_reference, (gray_current.shape[1], gray_current.shape[0]))
            
            # Calculate frame difference
            diff = cv2.absdiff(gray_current, gray_reference)
            
            # Calculate difference metrics
            mean_diff = np.mean(diff)
            max_diff = np.max(diff)
            diff_ratio = mean_diff / 255.0
            
            # Check for tampering
            tamper_detected = False
            reason = "normal"
            
            if diff_ratio > DatacenterConfig.FRAME_DIFF_THRESHOLD:
                tamper_detected = True
                reason = "significant_frame_difference"
            elif max_diff < 10:  # Very low difference might indicate obstruction
                tamper_detected = True
                reason = "possible_obstruction"
            
            return {
                'tamper_detected': tamper_detected,
                'reason': reason,
                'difference_ratio': diff_ratio,
                'mean_difference': mean_diff,
                'max_difference': max_diff
            }
            
        except Exception as e:
            self.logger.error(f"Error in camera tampering detection: {str(e)}")
            return {'tamper_detected': False, 'reason': 'detection_error', 'error': str(e)}
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage ratio"""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            allocated = torch.cuda.memory_allocated()
            return allocated / self.total_gpu_memory
        except Exception:
            return 0.0
    
    def _cleanup_gpu_memory(self):
        """Cleanup GPU memory"""
        try:
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info("GPU memory cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during GPU memory cleanup: {str(e)}")
    
    def _log_gpu_memory_usage(self, context: str = ""):
        """Log current GPU memory usage"""
        if not torch.cuda.is_available():
            return
        
        try:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            
            allocated_gb = allocated / 1e9
            reserved_gb = reserved / 1e9
            usage_percent = (allocated / self.total_gpu_memory) * 100
            
            # Update peak memory usage
            self.performance_stats['gpu_memory_peak'] = max(
                self.performance_stats['gpu_memory_peak'], 
                allocated
            )
            
            context_str = f" ({context})" if context else ""
            self.logger.debug(f"GPU Memory{context_str}: {allocated_gb:.2f}GB allocated, "
                            f"{reserved_gb:.2f}GB reserved, {usage_percent:.1f}% used")
            
            # Log performance metrics
            performance_logger.log_system_resources(
                cpu_usage=0,  # Would need psutil for this
                memory_usage=0,  # Would need psutil for this
                gpu_usage=usage_percent,
                disk_usage=0  # Would need psutil for this
            )
            
        except Exception as e:
            self.logger.error(f"Error logging GPU memory usage: {str(e)}")
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get comprehensive model performance statistics"""
        stats = {
            'device': self.device,
            'gpu_available': torch.cuda.is_available(),
            'total_gpu_memory_gb': self.total_gpu_memory / 1e9 if self.total_gpu_memory > 0 else 0,
            'current_gpu_usage': self._get_memory_usage(),
            'performance_stats': self.performance_stats.copy(),
            'model_instances': {}
        }
        
        # Add per-model statistics
        for model_type, instances in self.model_instances.items():
            if instances:
                instance_stats = [instance.get_usage_stats() for instance in instances]
                
                stats['model_instances'][model_type] = {
                    'instance_count': len(instances),
                    'total_inferences': sum(s['inference_count'] for s in instance_stats),
                    'average_inference_time': np.mean([s['average_inference_time'] for s in instance_stats if s['inference_count'] > 0]),
                    'instances_in_use': sum(1 for s in instance_stats if s['currently_in_use']),
                    'confidence_threshold': instances[0].confidence_threshold,
                    'supported_classes': len(instances[0].supported_classes)
                }
        
        # Add derived metrics
        if stats['performance_stats']['total_inferences'] > 0:
            stats['overall_average_inference_time'] = (
                stats['performance_stats']['total_inference_time'] / 
                stats['performance_stats']['total_inferences']
            )
        else:
            stats['overall_average_inference_time'] = 0
        
        return stats
    
    def get_model_health_status(self) -> Dict[str, str]:
        """Get health status of all model types"""
        health_status = {}
        
        for model_type, instances in self.model_instances.items():
            if not instances:
                health_status[model_type] = 'not_loaded'
                continue
            
            # Check if any instance is available
            available_instances = sum(1 for instance in instances if not instance.in_use)
            
            if available_instances == 0:
                health_status[model_type] = 'overloaded'
            elif available_instances == len(instances):
                health_status[model_type] = 'idle'
            else:
                health_status[model_type] = 'active'
        
        return health_status
    
    def optimize_memory_usage(self):
        """Optimize GPU memory usage"""
        try:
            self.logger.info("Optimizing GPU memory usage")
            
            current_usage = self._get_memory_usage()
            self.logger.info(f"Current GPU memory usage: {current_usage:.2%}")
            
            if current_usage > self.memory_threshold:
                self.logger.warning("GPU memory usage above threshold, performing optimization")
                
                # Clear unused model instances
                self._cleanup_unused_instances()
                
                # Force garbage collection
                self._cleanup_gpu_memory()
                
                # Log results
                new_usage = self._get_memory_usage()
                saved_memory = (current_usage - new_usage) * self.total_gpu_memory / 1e9
                
                self.logger.info(f"Memory optimization complete: {new_usage:.2%} usage "
                               f"(saved {saved_memory:.2f}GB)")
            else:
                self.logger.info("GPU memory usage within acceptable limits")
                
        except Exception as e:
            self.logger.error(f"Error during memory optimization: {str(e)}")
    
    def _cleanup_unused_instances(self):
        """Clean up unused model instances that haven't been used recently"""
        current_time = time.time()
        cleanup_threshold = 300  # 5 minutes
        
        for model_type, instances in self.model_instances.items():
            instances_to_remove = []
            
            for i, instance in enumerate(instances):
                with instance.lock:
                    if (not instance.in_use and 
                        current_time - instance.last_used > cleanup_threshold and
                        len(instances) > 1):  # Keep at least one instance
                        
                        instances_to_remove.append(i)
                        self.logger.info(f"Marking unused {model_type} instance for cleanup")
            
            # Remove instances (in reverse order to maintain indices)
            for i in reversed(instances_to_remove):
                try:
                    del instances[i]
                    self.logger.info(f"Cleaned up unused {model_type} instance")
                except Exception as e:
                    self.logger.error(f"Error cleaning up instance: {str(e)}")
    
    def reload_model(self, model_type: str) -> bool:
        """Reload a specific model type"""
        try:
            self.logger.info(f"Reloading {model_type} model")
            
            # Clear existing instances
            with self.model_locks[model_type]:
                # Wait for all instances to be released
                max_wait = 30  # seconds
                start_wait = time.time()
                
                while any(instance.in_use for instance in self.model_instances[model_type]):
                    if time.time() - start_wait > max_wait:
                        self.logger.error(f"Timeout waiting for {model_type} instances to be released")
                        return False
                    time.sleep(0.1)
                
                # Clear instances
                self.model_instances[model_type].clear()
                
                # Reload model
                if model_type == 'person_detection':
                    self._load_person_detection_model()
                elif model_type == 'ppe_detection':
                    self._load_ppe_detection_model()
                elif model_type == 'general_detection':
                    self._load_general_detection_model()
                else:
                    self.logger.error(f"Unknown model type for reload: {model_type}")
                    return False
            
            self.logger.info(f"Successfully reloaded {model_type} model")
            return True
            
        except Exception as e:
            self.logger.error(f"Error reloading {model_type} model: {str(e)}", exc_info=True)
            return False
    
    def shutdown(self):
        """Shutdown model manager and cleanup resources"""
        self.logger.info("Shutting down Datacenter Model Manager")
        
        try:
            # Wait for all models to be released
            for model_type in self.model_instances:
                with self.model_locks[model_type]:
                    for instance in self.model_instances[model_type]:
                        with instance.lock:
                            if instance.in_use:
                                self.logger.warning(f"Model instance {model_type} still in use during shutdown")
            
            # Clear all instances
            for model_type in self.model_instances:
                self.model_instances[model_type].clear()
            
            # Final memory cleanup
            self._cleanup_gpu_memory()
            
            # Log final statistics
            final_stats = self.get_model_statistics()
            self.logger.info(f"Final model statistics: {final_stats['performance_stats']}")
            
            self.logger.info("Datacenter Model Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during model manager shutdown: {str(e)}")

# Convenience functions for backwards compatibility
def get_model_instance(model_type: str, manager: Optional[DatacenterModelManager] = None) -> Optional[DatacenterModelInstance]:
    """Get model instance - backwards compatibility function"""
    if manager is None:
        # This would require a global manager instance
        raise ValueError("Model manager instance required")
    return manager.get_model_instance(model_type)

def release_model_instance(instance: DatacenterModelInstance, manager: Optional[DatacenterModelManager] = None):
    """Release model instance - backwards compatibility function"""
    if manager is None:
        raise ValueError("Model manager instance required")
    manager.release_model_instance(instance)

# Testing functionality
if __name__ == "__main__":
    import cv2
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Datacenter Model Manager")
    parser.add_argument("--test-image", type=str, help="Path to test image")
    parser.add_argument("--iterations", type=int, default=10, help="Number of test iterations")
    
    args = parser.parse_args()
    
    # Initialize model manager
    print("Initializing Datacenter Model Manager...")
    manager = DatacenterModelManager()
    
    try:
        # Load test image or create dummy data
        if args.test_image and os.path.exists(args.test_image):
            test_frame = cv2.imread(args.test_image)
            print(f"Loaded test image: {test_frame.shape}")
        else:
            # Create dummy frame
            test_frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            print("Using dummy test frame")
        
        # Test person detection
        print(f"\nTesting person detection ({args.iterations} iterations)...")
        start_time = time.time()
        
        for i in range(args.iterations):
            results = manager.run_person_detection([test_frame])
            if i == 0:
                detections = len(results[0].boxes) if results and results[0].boxes else 0
                print(f"First iteration: {detections} detections")
        
        person_time = time.time() - start_time
        print(f"Person detection: {person_time:.3f}s total, {person_time/args.iterations:.3f}s average")
        
        # Test PPE detection if enabled
        if DatacenterConfig.PPE_DETECTION_ENABLED:
            print(f"\nTesting PPE detection ({args.iterations} iterations)...")
            start_time = time.time()
            
            for i in range(args.iterations):
                results = manager.run_ppe_detection([test_frame])
                if i == 0:
                    detections = len(results[0].boxes) if results and results[0].boxes else 0
                    print(f"First iteration: {detections} PPE detections")
            
            ppe_time = time.time() - start_time
            print(f"PPE detection: {ppe_time:.3f}s total, {ppe_time/args.iterations:.3f}s average")
        
        # Test combined detection
        print(f"\nTesting combined detection ({args.iterations} iterations)...")
        start_time = time.time()
        
        for i in range(args.iterations):
            person_results, ppe_results = manager.run_combined_detection([test_frame])
        
        combined_time = time.time() - start_time
        print(f"Combined detection: {combined_time:.3f}s total, {combined_time/args.iterations:.3f}s average")
        
        # Print model statistics
        print("\nModel Statistics:")
        stats = manager.get_model_statistics()
        for key, value in stats.items():
            if key != 'model_instances':
                print(f"  {key}: {value}")
        
        print("\nModel Instance Statistics:")
        for model_type, instance_stats in stats['model_instances'].items():
            print(f"  {model_type}:")
            for key, value in instance_stats.items():
                print(f"    {key}: {value}")
        
        # Test memory optimization
        print("\nTesting memory optimization...")
        manager.optimize_memory_usage()
        
        # Health status
        health = manager.get_model_health_status()
        print(f"\nModel Health Status: {health}")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutting down model manager...")
        manager.shutdown()
        print("Test complete")