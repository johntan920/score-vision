import asyncio
import time
from typing import AsyncGenerator, Optional, Tuple
import cv2
import numpy as np
import supervision as sv
from loguru import logger
import subprocess
# import pynvcodec as nvc
import torch
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
        batch_size: int = 4,
        sample_rate: int = 5,
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
        start_index = 0  # Start from the first frame

        # cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        width, height = 640, 360  # Desired output resolution

        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-f", "rawvideo",      # Raw output
            "-pix_fmt", "bgr24",   # OpenCV-compatible pixel format
            "-vf", f"scale={width}:{height}",  # Resize (remove if not needed)
            "pipe:1"               # Send to stdout
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
        frame_size = width * height * 3  # Bytes per frame (BGR)

        flag = True
        
        try:
            while True:
                if not flag:
                    break

                frame_batch = []

                # if start_index >= total_frames:
                #     break
                # for target_frame in range(batch_size):
                #     elapsed_time = time.time() - start_time
                #     if elapsed_time > self.processing_timeout:
                #         logger.warning(f"Video processing timeout reached after {elapsed_time:.1f}s")
                #         break
                    
                #     # Seek to the target frame
                #     ret, frame = cap.read()

                #     frame = cv2.resize(frame, (640, 360))

                #     frame_batch.append(frame)
                    
                #     if not ret:
                #         logger.warning(f"Could not read frame {target_frame}")
                #         break

                for i in range(batch_size * sample_rate):
                    raw_frame = process.stdout.read(frame_size)
                    if not raw_frame:
                        flag = False
                        break  # End of video
                    
                    if (start_index + i + 1) % sample_rate == 0:
                        frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
                        frame_batch.append(frame)

                yield start_index, frame_batch
                start_index += batch_size * sample_rate
                
                # # Add small delay to prevent overwhelming the GPU
                # if self.device == "cuda" and i % batch_size == 0:
                #     await asyncio.sleep(0.001)  # 1ms delay every batch

            process.stdout.close()
            process.wait()
        
        finally:
            print('OK')
            # cap.release()

    async def stream_sampled_frames_gpu(self, video_path: str, batch_size: int = 4):
        

        gpu_id = 0
        nvdec = nvc.PyNvCodec.PyNvDecoder(video_path, gpu_id)

        batch = []
        idx = 0
        while True:
            t1 = time.time()
            success, frame = nvdec.DecodeSingleFrame()
            if not success:
                if batch:
                    yield idx - len(batch), batch
                break

            tensor = torch.as_tensor(frame)
            batch.append(tensor)

            if len(batch) == batch_size:
                yield idx - batch_size + 1, batch
                batch = []

            idx += 1
            t2 = time.time()
            print('===Batching time===', t2 - t1)

        if batch:
            yield idx - len(batch), batch

    async def stream_sampled_frames_ffmpeg(self,video_path, batch_size=15):
        probe = ffmpeg.probe(video_path)
        w = int(probe['streams'][0]['width'])
        h = int(probe['streams'][0]['height'])

        process = (
            ffmpeg.input(video_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True)
        )

        batch = []
        frame_number = 0
        while True:
            t1 = time.time()
            in_bytes = process.stdout.read(w * h * 3)
            if not in_bytes:
                if batch:
                    yield frame_number - len(batch), batch
                break
            frame = np.frombuffer(in_bytes, np.uint8).reshape([h, w, 3])
            batch.append(torch.from_numpy(frame).permute(2, 0, 1).cuda())
            if len(batch) == batch_size:
                yield frame_number - batch_size + 1, batch
                batch = []
            frame_number += 1
            t2 = time.time()
            print('===Batching time===', t2 - t1)
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