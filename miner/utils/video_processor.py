import asyncio
import time
from typing import AsyncGenerator, Optional, Tuple
import cv2
import numpy as np
import supervision as sv
import torch
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

class VideoProcessor:
    """Handles video processing with frame streaming and timeout management."""
    
    def __init__(
        self,
        device: str = "cpu",
        cuda_timeout: float = 900.0,  # 15 minutes for CUDA
        mps_timeout: float = 1800.0,  # 30 minutes for MPS
        cpu_timeout: float = 10800.0,  # 3 hours for CPU
    ):
        self.device = device
        # Set timeout based on device
        if device == "cuda":
            self.processing_timeout = cuda_timeout
        elif device == "mps":
            self.processing_timeout = mps_timeout
        else:  # cpu or any other device
            self.processing_timeout = cpu_timeout
        
        # Initialize thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
            
        logger.info(f"Video processor initialized with {device} device, timeout: {self.processing_timeout:.1f}s")
    
    async def stream_frames(
        self,
        video_path: str
    ) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """
        Stream video frames asynchronously with timeout protection.
        Process ALL frames regardless of compute device.
        
        Args:
            video_path: Path to the video file
            
        Yields:
            Tuple[int, np.ndarray]: Frame number and frame data
        """
        start_time = time.time()
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            frame_count = 0
            while True:
                elapsed_time = time.time() - start_time
                if elapsed_time > self.processing_timeout:
                    logger.warning(
                        f"Video processing timeout reached after {elapsed_time:.1f}s "
                        f"on {self.device} device ({frame_count} frames processed)"
                    )
                    break
                
                # Use run_in_executor to prevent blocking the event loop
                ret, frame = await asyncio.get_event_loop().run_in_executor(
                    None, cap.read
                )
                
                if not ret:
                    logger.info(f"Completed processing {frame_count} frames in {elapsed_time:.1f}s on {self.device} device")
                    break
                
                yield frame_count, frame
                frame_count += 1
                
                # Small delay to prevent CPU hogging while still processing all frames
                await asyncio.sleep(0)
        
        finally:
            cap.release()
    
    async def stream_sampled_frames(
        self,
        video_path: str,
        frame_indices: list,
        batch_size: int = 4
    ) -> AsyncGenerator[Tuple[int, np.ndarray], None]:
        """
        Stream specific frames efficiently with batching for RTX 4090.
        
        Args:
            video_path: Path to the video file
            frame_indices: List of frame indices to extract
            batch_size: Number of frames to process in parallel
            
        Yields:
            Tuple[int, np.ndarray]: Frame number and frame data
        """
        start_time = time.time()
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Sort frame indices for efficient seeking
        sorted_indices = sorted(frame_indices)
        
        try:
            for i, target_frame in enumerate(sorted_indices):
                elapsed_time = time.time() - start_time
                if elapsed_time > self.processing_timeout:
                    logger.warning(f"Video processing timeout reached after {elapsed_time:.1f}s")
                    break
                
                # Seek to the target frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"Could not read frame {target_frame}")
                    continue
                
                yield target_frame, frame
                
                # Add small delay to prevent overwhelming the GPU
                if self.device == "cuda" and i % batch_size == 0:
                    await asyncio.sleep(0.001)  # 1ms delay every batch
        
        finally:
            cap.release()
    
    @staticmethod
    def get_video_info(video_path: str) -> sv.VideoInfo:
        """Get video information using supervision."""
        return sv.VideoInfo.from_video_path(video_path)
    
    @staticmethod
    async def ensure_video_readable(video_path: str, timeout: float = 5.0) -> bool:
        """
        Check if video is readable within timeout period.
        
        Args:
            video_path: Path to video file
            timeout: Maximum time to wait for video check
            
        Returns:
            bool: True if video is readable
        """
        try:
            async def _check_video():
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    return False
                ret, _ = cap.read()
                cap.release()
                return ret
            
            return await asyncio.wait_for(_check_video(), timeout)
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout while checking video readability: {video_path}")
            return False
        except Exception as e:
            logger.error(f"Error checking video readability: {str(e)}")
            return False 