"""
RTX 4090 Optimization Configuration
Optimizes video processing for maximum speed on RTX 4090 with 24GB VRAM
"""

class RTX4090Config:
    """Configuration optimized for RTX 4090 24GB VRAM"""
    
    # Inference settings
    INFERENCE_SIZE = 640  # Reduced from 1280 for speed
    BATCH_SIZE = 8        # Process multiple frames at once
    MAX_FRAMES = 30       # Process 30 frames instead of 3
    
    # Model settings
    USE_HALF_PRECISION = True     # FP16 for faster inference
    CONFIDENCE_THRESHOLD = 0.3    # Lower threshold for speed
    IOU_THRESHOLD = 0.7          # Higher IoU to reduce false positives
    
    # CUDA optimizations
    ENABLE_CUDNN_BENCHMARK = True
    ENABLE_TF32 = True
    
    # Memory management
    GPU_MEMORY_FRACTION = 0.8    # Use 80% of 24GB = ~19GB
    CLEAR_CACHE_INTERVAL = 100   # Clear cache every 100 frames
    
    # Processing optimizations
    PARALLEL_WORKERS = 4         # Parallel processing threads
    FRAME_SKIP_THRESHOLD = 0.95  # Skip frames with low confidence
    
    # Video processing
    VIDEO_BUFFER_SIZE = 16       # Number of frames to buffer
    USE_GPU_DECODE = False       # OpenCV doesn't support GPU decode well
    
    @classmethod
    def get_yolo_kwargs(cls) -> dict:
        """Get optimized YOLO inference parameters"""
        return {
            'imgsz': cls.INFERENCE_SIZE,
            'half': cls.USE_HALF_PRECISION,
            'verbose': False,
            'conf': cls.CONFIDENCE_THRESHOLD,
            'iou': cls.IOU_THRESHOLD,
        }
    
    @classmethod
    def setup_cuda_optimizations(cls):
        """Setup CUDA optimizations for RTX 4090"""
        import torch
        if torch.cuda.is_available():
            # Enable optimizations
            torch.backends.cudnn.benchmark = cls.ENABLE_CUDNN_BENCHMARK
            torch.backends.cuda.matmul.allow_tf32 = cls.ENABLE_TF32
            
            # Set memory fraction
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(cls.GPU_MEMORY_FRACTION)
            
            # Enable async operations
            torch.backends.cudnn.allow_tf32 = True
            
            print(f"CUDA optimizations enabled for RTX 4090:")
            print(f"  - cuDNN benchmark: {cls.ENABLE_CUDNN_BENCHMARK}")
            print(f"  - TensorFloat-32: {cls.ENABLE_TF32}")
            print(f"  - Memory fraction: {cls.GPU_MEMORY_FRACTION}")