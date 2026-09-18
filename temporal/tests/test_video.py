import pytest
import os
import numpy as np
import cv2
import tempfile
from temporal.video import VideoLoader

@pytest.fixture
def sample_video():
    """Create a temporary video file for testing."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    
    # Create a 1-second video at 30 fps, size 640x480
    fps = 30
    w, h = 640, 480
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for i in range(fps):
        # Frame is just a specific color to verify extraction if needed
        frame = np.full((h, w, 3), (i, i, i), dtype=np.uint8)
        out.write(frame)
    out.release()
    
    yield path
    os.remove(path)

def test_video_loader_metadata(sample_video):
    loader = VideoLoader(sample_video, resize_max_dim=320)
    meta = loader.metadata
    assert meta["w"] == 640
    assert meta["h"] == 480
    assert meta["target_w"] == 320
    assert meta["target_h"] == 240
    assert meta["fps"] == 30.0

def test_video_loader_streaming(sample_video):
    # Sample at 2 fps. Over 1 second, we should get 2 frames.
    loader = VideoLoader(sample_video, sample_fps=2.0, resize_max_dim=320)
    
    frames = list(loader.stream_frames())
    
    assert len(frames) == 2
    
    # Check dimensions and timestamps
    idx1, ts1, frame1 = frames[0]
    idx2, ts2, frame2 = frames[1]
    
    assert idx1 == 0
    assert ts1 == 0.0
    assert frame1.shape == (240, 320, 3)
    
    assert idx2 == 1
    assert ts2 == 0.5
    assert frame2.shape == (240, 320, 3)
