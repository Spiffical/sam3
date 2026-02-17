
from __future__ import annotations
import shutil
import tempfile
import cv2
import gradio as gr
import numpy as np
from pathlib import Path
from typing import List, Tuple

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

def _safe_sort_key(path: Path) -> Tuple[int, str]:
    stem = path.stem
    if stem.isdigit():
        return (0, f"{int(stem):09d}")
    return (1, stem)


def _load_frames(resource_path: str) -> List[np.ndarray]:
    path = Path(resource_path)
    frames: List[np.ndarray] = []

    if path.is_file():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise gr.Error(f"Couldn't open video: {path}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        img_files = [p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if not img_files:
            raise gr.Error("No JPEG frames found in the provided folder.")
        img_files.sort(key=_safe_sort_key)
        for img_path in img_files:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not frames:
        raise gr.Error("No frames could be decoded from this input.")

    return frames


def _copy_frame_uploads(files: List) -> str:
    tmp_dir = Path(tempfile.mkdtemp(prefix="sam3_frames_"))
    for fh in files:
        src = Path(fh.name)
        orig_name = getattr(fh, "orig_name", None) or src.name
        dst = tmp_dir / Path(orig_name).name
        shutil.copy(src, dst)
    return str(tmp_dir)
