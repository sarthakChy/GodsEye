import os
import cv2
import imageio_ffmpeg
import numpy as np
from typing import Iterator, Tuple
import logging

def _even(n: int) -> int:
    return int(n) // 2 * 2

def _get_vf_args(W: int, H: int) -> str:
    return f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black"

class VideoLoader:
    """Streams video frames using imageio_ffmpeg for AV1/WebM support.
    
    Frames are yielded one at a time to keep memory bounded.
    """
    def __init__(self, path: str, sample_fps: float = 2.0, max_frames: int = 600, resize_max_dim: int = 1280):
        self.path = path
        self.sample_fps = sample_fps
        self.max_frames = max_frames
        self.resize_max_dim = resize_max_dim
        self.metadata = self._probe()
        
    def _probe(self) -> dict:
        logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
        gen = imageio_ffmpeg.read_frames(self.path, pix_fmt="bgr24")
        try:
            meta = next(gen)
        except StopIteration:
            raise ValueError(f"Could not read metadata from {self.path}")
        finally:
            gen.close()
        
        w, h = meta["size"]
        fps = float(meta.get("fps") or 25.0)
        duration = float(meta.get("duration") or 0.0)
        
        # Calculate target dimensions
        r = min(self.resize_max_dim / w, self.resize_max_dim / h)
        if r < 1.0:
            target_w = _even(w * r)
            target_h = _even(h * r)
        else:
            target_w = _even(w)
            target_h = _even(h)
            
        return {
            "w": w, "h": h, 
            "target_w": target_w, "target_h": target_h,
            "fps": fps, "duration": duration
        }
        
    def stream_frames(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        """Yields (frame_idx, timestamp_sec, frame_array_bgr)."""
        W = self.metadata["target_w"]
        H = self.metadata["target_h"]
        
        vf = _get_vf_args(W, H)
        if self.sample_fps > 0:
            vf += f",fps={self.sample_fps:.4f}"
            
        op = ["-vf", vf]
        
        gen = imageio_ffmpeg.read_frames(self.path, pix_fmt="bgr24", output_params=op)
        next(gen) # Skip meta
        
        frame_idx = 0
        actual_fps = self.sample_fps if self.sample_fps > 0 else self.metadata["fps"]
        
        for buf in gen:
            if frame_idx >= self.max_frames:
                break
            
            frame_array = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            timestamp_sec = frame_idx / actual_fps
            
            yield frame_idx, timestamp_sec, frame_array
            frame_idx += 1
            
    def get_frame(self, timestamp_sec: float) -> np.ndarray:
        """Seek to a precise timestamp and extract a single frame with identical preprocessing."""
        W = self.metadata["target_w"]
        H = self.metadata["target_h"]
        vf = _get_vf_args(W, H)
        
        # We don't need fps resampling for a single frame
        op = ["-vf", vf, "-vframes", "1"]
        ip = ["-ss", str(timestamp_sec)]
        
        gen = imageio_ffmpeg.read_frames(self.path, pix_fmt="bgr24", input_params=ip, output_params=op)
        next(gen) # Skip meta
        
        try:
            buf = next(gen)
            frame_array = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            return frame_array
        except StopIteration:
            return np.zeros((H, W, 3), dtype=np.uint8)
