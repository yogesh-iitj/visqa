"""
Video reading and writing utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm


class VideoReader:
    """Reads frames from a video file with optional striding."""

    def __init__(self, path: str, stride: int = 1, max_frames: Optional[int] = None):
        self.path = path
        self.stride = stride
        self.max_frames = max_frames

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    def read_all(self, show_progress: bool = False) -> List[np.ndarray]:
        """Read all frames (with stride) as a list of RGB numpy arrays."""
        cap = cv2.VideoCapture(self.path)
        frames = []
        frame_idx = 0
        it = tqdm(total=self.total_frames // self.stride, desc="Reading video") if show_progress else None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.stride == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if it:
                    it.update(1)
                if self.max_frames and len(frames) >= self.max_frames:
                    break
            frame_idx += 1

        cap.release()
        if it:
            it.close()
        return frames

    def read_frame(self, frame_idx: int) -> np.ndarray:
        """Read a single frame by index."""
        cap = cv2.VideoCapture(self.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Cannot read frame {frame_idx}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return self.total_frames // self.stride


class VideoWriter:
    """Writes frames to an output video file."""

    def __init__(
        self,
        path: str,
        fps: float = 30.0,
        width: int = 1920,
        height: int = 1080,
        codec: str = "mp4v",
    ):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    def write(self, frame: np.ndarray):
        """Write an RGB frame."""
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._writer.write(bgr)

    def release(self):
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


def extract_frames_to_dir(video_path: str, output_dir: str, stride: int = 1):
    """Extract video frames to a directory as JPEG images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = VideoReader(video_path, stride=stride)
    frames = reader.read_all(show_progress=True)

    for i, frame in enumerate(frames):
        out_path = output_dir / f"{i:06d}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    return frames, reader.fps
