#!/usr/bin/env python3
"""
RTX 4090 Performance Benchmark Script
Tests the optimized soccer video processing performance
"""

import asyncio
import time
import tempfile
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from miner.utils.model_manager import ModelManager
from miner.endpoints.soccer import process_soccer_video
from miner.configs.rtx4090_config import RTX4090Config


async def benchmark_processing(video_path: str, iterations: int = 3):
    """Benchmark the optimized video processing"""
    
    print("🚀 RTX 4090 Soccer Video Processing Benchmark")
    print("=" * 50)
    
    # Initialize model manager
    print("📦 Loading models...")
    model_manager = ModelManager(device="cuda")  # Force CUDA for RTX 4090
    model_manager.load_all_models()
    
    print(f"✅ Models loaded on device: {model_manager.device}")
    print(f"📊 Processing configuration:")
    print(f"   - Max frames: {RTX4090Config.MAX_FRAMES}")
    print(f"   - Inference size: {RTX4090Config.INFERENCE_SIZE}")
    print(f"   - Half precision: {RTX4090Config.USE_HALF_PRECISION}")
    print(f"   - Batch size: {RTX4090Config.BATCH_SIZE}")
    print()
    
    processing_times = []
    
    for i in range(iterations):
        print(f"🎬 Processing iteration {i+1}/{iterations}...")
        
        start_time = time.time()
        try:
            result = await process_soccer_video(video_path, model_manager)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            frames_processed = len(result.get("frames", []))
            fps = frames_processed / processing_time if processing_time > 0 else 0
            
            print(f"   ✅ Completed in {processing_time:.2f}s")
            print(f"   📈 Processed {frames_processed} frames ({fps:.2f} fps)")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return
        
        print()
    
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        
        print("📊 BENCHMARK RESULTS")
        print("=" * 30)
        print(f"Average time: {avg_time:.2f}s")
        print(f"Minimum time: {min_time:.2f}s")
        print(f"Maximum time: {max_time:.2f}s")
        print()
        
        if avg_time <= 2.0:
            print("🎉 TARGET ACHIEVED: Processing time ≤ 2s!")
        elif avg_time <= 5.0:
            print("✅ GOOD: Processing time ≤ 5s")
        elif avg_time <= 10.0:
            print("⚠️  MODERATE: Processing time ≤ 10s")
        else:
            print("❌ NEEDS OPTIMIZATION: Processing time > 10s")
        
        speedup = 40.0 / avg_time  # Assuming original was 40s
        print(f"🚀 Speedup: {speedup:.1f}x faster than original")


def create_test_video(duration_seconds: int = 10) -> str:
    """Create a test video for benchmarking"""
    import cv2
    import numpy as np
    
    # Create temporary video file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file.close()
    
    # Video properties
    width, height = 1920, 1080
    fps = 30
    total_frames = duration_seconds * fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))
    
    print(f"🎬 Creating test video: {temp_file.name}")
    print(f"   Duration: {duration_seconds}s, Resolution: {width}x{height}, FPS: {fps}")
    
    for frame_num in range(total_frames):
        # Create a simple test frame (soccer field-like)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (34, 139, 34)  # Green background
        
        # Add some moving elements (simulating players)
        for i in range(5):
            x = int((frame_num * 5 + i * 100) % width)
            y = int(height // 2 + 50 * np.sin(frame_num * 0.1 + i))
            cv2.circle(frame, (x, y), 20, (255, 255, 255), -1)
        
        # Add field lines
        cv2.rectangle(frame, (100, 100), (width-100, height-100), (255, 255, 255), 3)
        cv2.circle(frame, (width//2, height//2), 100, (255, 255, 255), 3)
        
        writer.write(frame)
    
    writer.release()
    print(f"✅ Test video created: {temp_file.name}")
    
    return temp_file.name


async def main():
    """Main benchmark function"""
    
    # Check if a video file is provided
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return
    else:
        # Create a test video
        video_path = create_test_video(duration_seconds=30)
    
    try:
        await benchmark_processing(video_path, iterations=3)
    finally:
        # Clean up test video if we created it
        if len(sys.argv) <= 1:
            try:
                os.unlink(video_path)
                print(f"🧹 Cleaned up test video: {video_path}")
            except:
                pass


if __name__ == "__main__":
    print("RTX 4090 Soccer Processing Benchmark")
    print("Usage: python benchmark_rtx4090.py [video_file]")
    print("If no video file is provided, a test video will be created.")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()