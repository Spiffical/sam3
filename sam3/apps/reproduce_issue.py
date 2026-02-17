import os
import json
import torch
import cv2
import sys
from sam3.apps.interactive_video.backend import PredictorBackend

# Mocking Gradio Error to avoid imports failure if specific to UI
class MockGradioError(Exception):
    pass

import gradio as gr
gr.Error = MockGradioError

def main():
    video_path = os.path.abspath("assets/videos/chinacreekclipped.mp4")
    annotation_path = os.path.abspath("assets/annotations/chinacreekclipped_prompts.json")
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    print(f"Testing with video: {video_path}")
    
    # Initialize Backend
    # Assumes running on GPU 0
    backend = PredictorBackend(gpu_ids=[0])
    
    # Simulate start_session logic from ui.py
    # Load frames to get size
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return
        
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to read first frame")
        return

    h, w = frame.shape[:2]
    # ui.py uses image_size=1024 by default (from dropdown value)
    image_size = 1008 
    print(f"Video size: {w}x{h}, processing size: {image_size}")
    
    try:
        session_id = backend.start_session(
            resource_path=video_path,
            image_size=image_size,
            offload_video_to_cpu=True
        )
        print(f"Session started: {session_id}")
        
        # Load annotations
        if not os.path.exists(annotation_path):
             print("No annotations found.")
             return

        with open(annotation_path, "r") as f:
            annotations = json.load(f)
            
        print(f"Loading {len(annotations)} annotations...")
        
        frame_size = (w, h)
        
        for item in annotations:
            frame_idx = item.get("frame_idx", 0)
            obj_id = item.get("obj_id", 1)
            points_raw = item.get("points", []) 
            
            # format points
            current_points = []
            for p in points_raw:
                x, y, label = p
                current_points.append((x, y, int(label)))
            
            print(f"Adding prompt: Frame {frame_idx}, Obj {obj_id}, Points {current_points}")
            
            # This should trigger the error
            backend.add_point_prompt(
                session_id,
                frame_idx,
                obj_id,
                current_points,
                frame_size
            )
            print(f"Successfully added prompt for obj {obj_id}")

    except Exception as e:
        print(f"\n[REPRODUCTION CAUGHT EXCEPTION]")
        print(f"Type: {type(e)}")
        print(f"Message: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
