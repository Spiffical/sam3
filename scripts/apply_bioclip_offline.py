import sys
import os
import json
import torch
import cv2
import numpy as np
import threading
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sam3.apps.interactive_video.backend import PredictorBackend
from sam3.apps.interactive_video.bioclip_utils import predict_hierarchical
from sam3.apps.interactive_video.video_loader import _load_frames

def get_bbox_from_mask(mask_logit):
    # mask_logit: HxW np array (float)
    # Binary mask
    mask = mask_logit > 0.0
    if not np.any(mask):
        return None
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return [x_min, y_min, x_max - x_min, y_max - y_min] # xywh

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run BioCLIP on video with SAM3 propagation")
    parser.add_argument("--video", type=str, default="assets/videos/chinacreekclipped.mp4", help="Path to input video")
    parser.add_argument("--prompts", type=str, default="assets/annotations/chinacreekclipped_prompts.json", help="Path to input prompts JSON")
    parser.add_argument("--output", type=str, default="assets/annotations/chinacreekclipped_prompts_bioclip_hierarchical.json", help="Path to output JSON")
    parser.add_argument("--save-debug-crops", action="store_true", help="Save the best crop for each object to assets/debug_crops/")
    parser.add_argument("--region", type=str, default=None, help="Comma-separated list of allowed regions (e.g. Pacific,Canada)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    video_rel_path = args.video
    prompts_rel_path = args.prompts
    output_rel_path = args.output
    allowed_regions = args.region.split(',') if args.region else None

    cwd = os.getcwd()
    video_path = os.path.abspath(video_rel_path)
    prompts_path = os.path.abspath(prompts_rel_path)
    output_path = os.path.abspath(output_rel_path)
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
    if not os.path.exists(prompts_path):
        print(f"Error: Prompts not found at {prompts_path}")
        return

    # 1. Load Video Frames
    print(f"Loading video frames from {video_path}...")
    frames = _load_frames(video_path)
    if not frames:
        print("Failed to load frames.")
        return
        
    H_frame, W_frame = frames[0].shape[:2]
    print(f"Loaded {len(frames)} frames. Size: {W_frame}x{H_frame}")

    # 2. Init Backend
    print("Initializing SAM3 Backend...")
    backend = PredictorBackend(gpu_ids=[0])
    
    try:
        print("Starting SAM3 session...")
        session_id = backend.start_session(
            resource_path=video_path,
            image_size=1008,
            offload_video_to_cpu=True
        )
        print(f"Session started: {session_id}")
        
        # 3. Load Prompts
        print(f"Loading annotations from {prompts_path}...")
        with open(prompts_path, 'r') as f:
            prompts_list = json.load(f)
            
        # Group by Object ID
        objects = {} 
        for p in prompts_list:
            objects.setdefault(p["obj_id"], []).append(p)
        
        print(f"Found {len(objects)} unique objects.")
        
        # 4. Phase 1: Add Prompts for ALL objects
        print("Phase 1: Adding prompts...")
        for obj_id, entries in objects.items():
            # Add ONLY the first prompt per object to avoid conflicts or complications?
            # Or add all? Usually tracking assumes 1 initial prompt is enough, but SAM2 allows multi-frame prompts.
            # Let's add all prompt *entries* provided in JSON.
            # But the JSON currently has 1 entry per object (based on previous run).
            # If multiple, we add them all.
            
            for entry in entries:
                frame_idx = entry["frame_idx"]
                points = entry["points"] # [[x, y, label], ...]
                fmt_points = [(p[0], p[1], p[2]) for p in points]
                
                backend.add_point_prompt(
                    session_id=session_id,
                    frame_idx=frame_idx,
                    obj_id=int(obj_id),
                    points=fmt_points,
                    frame_size=(W_frame, H_frame)
                )
        print("Prompts added.")

        # 5. Phase 2: Propagation
        print("Phase 2: Propagating segmentation across video...")
        
        # Store masks: masks_store[frame_idx][obj_id] = bbox (xywh pixels)
        masks_store = {}
        
        prop_req = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "start_frame_index": 0, # Should be min frame? Or 0 to propagate fwd?
                                    # If prompts are at 0, 0 is fine.
                                    # If prompt at 10, should we prop backward?
                                    # SAM3 propagate usually does forward from start_frame_index?
                                    # Actually SAM2 propagates 'whole video' usually logic-wise if called right.
                                    # Let's assume start 0.
        }
        
        frame_count = 0
        from contextlib import redirect_stdout
        import io
        
        # Suppress verbose propagation logs if possible
        # Iterate generator
        # Suppress verbose propagation logs if possible
        # Iterate generator
        for out in backend.propagate(prop_req):
            out_outputs = out.get("outputs")
            if out_outputs is None:
                # Direct output?
                out_outputs = out
            
            # handle keys
            # Keys found: ['out_obj_ids', 'out_probs', 'out_boxes_xywh', 'out_binary_masks', 'frame_stats']
            out_obj_ids = out_outputs.get("out_obj_ids")
            out_boxes = out_outputs.get("out_boxes_xywh")
            
            if out_obj_ids is None or out_boxes is None:
                 if frame_count == 0:
                     print(f"Debug: Missing keys in outputs. outputs keys: {out_outputs.keys()}")
                 frame_count += 1
                 continue
            
            # Extract boxes
            # out_boxes is likely tensor or list.
            if isinstance(out_boxes, torch.Tensor):
                out_boxes = out_boxes.cpu().numpy()
            elif isinstance(out_boxes, list):
                # Ensure structure
                pass
                
            f_idx = out.get("frame_index", out_outputs.get("frame_index", -1))
            if f_idx == -1:
                f_idx = frame_count 
            
            masks_store[f_idx] = {}
            
            # Iterate objects
            for i, o_id in enumerate(out_obj_ids):
                # Box i
                # Check shape. 
                # If out_boxes is (1, N, 4) or (N, 4)?
                # Usually (N, 4).
                if out_boxes.ndim == 3: # (B, N, 4)
                    box = out_boxes[0, i]
                elif out_boxes.ndim == 2: # (N, 4)
                    box = out_boxes[i]
                else:
                    box = out_boxes #?
                
                # Check if box is valid (not all zeros)
                if np.sum(box) > 0:
                     # Store as [nx, ny, nw, nh] (normalized xywh)
                     masks_store[f_idx][o_id] = box
            
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"Propagated {frame_count} frames...", end='\r')
                
        print(f"\nPropagation complete. Processed {frame_count} frames.")
        
        # 6. Phase 3: Classification
        object_metadata = {} # Store {label, score, edge_truncated}
        
        print("Phase 3: Running BioCLIP on crops from all frames...")
        
        for obj_id in objects.keys():
            obj_id = int(obj_id) # ensure int
            
            raw_candidates = [] # Tuples of (crop, is_near_edge, f_idx)
            
            # Sort frames to process in order
            sorted_frames = sorted(masks_store.keys())
            
            for f_idx in sorted_frames: 
                if f_idx >= len(frames): continue 
                
                if obj_id in masks_store[f_idx]:
                    bbox_norm = masks_store[f_idx][obj_id]
                    nx, ny, nw, nh = bbox_norm
                    
                    # Denormalize
                    H, W = frames[f_idx].shape[:2]
                    x = nx * W
                    y = ny * H
                    w = nw * W
                    h = nh * H
                    
                    if w < 10 or h < 10: continue

                    # Check edge proximity (e.g. within 2% of edge)
                    margin_x = W * 0.02
                    margin_y = H * 0.02
                    
                    # Check if box touches safety margin
                    is_near_edge = (x < margin_x) or (y < margin_y) or \
                                   ((x + w) > (W - margin_x)) or \
                                   ((y + h) > (H - margin_y))
                                   
                    # Crop with pad
                    pad = 0.5
                    x1 = int(max(0, x - w*pad))
                    y1 = int(max(0, y - h*pad))
                    x2 = int(min(W, x + w*(1+pad)))
                    y2 = int(min(H, y + h*(1+pad)))
                    
                    crop = frames[f_idx][y1:y2, x1:x2]
                    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0: continue
                    
                    raw_candidates.append({
                        "crop": crop,
                        "is_near_edge": is_near_edge,
                        "f_idx": f_idx
                    })

            if not raw_candidates:
                print(f"Obj {obj_id}: No valid crops found after propagation.")
                continue
            
            # Filter Strategy
            # If we have frames AWAY from edge, use ONLY them.
            # If all frames are NEAR edge, use all frames but flag as truncated.
            
            good_crops = [c for c in raw_candidates if not c["is_near_edge"]]
            
            final_candidates = []
            edge_truncated = False
            
            if good_crops:
                final_candidates = [c["crop"] for c in good_crops]
                # debug info
                if len(good_crops) < len(raw_candidates):
                    print(f"Obj {obj_id}: Filtered {len(raw_candidates) - len(good_crops)} near-edge frames. Using {len(good_crops)} frames.")
            else:
                # All near edge
                final_candidates = [c["crop"] for c in raw_candidates]
                edge_truncated = True
                print(f"Obj {obj_id}: All {len(raw_candidates)} frames are near edge. Marking as potentially truncated.")
                
            print(f"Obj {obj_id}: Predicting on {len(final_candidates)} frames...")
            
            results = predict_hierarchical(
                final_candidates, 
                threshold_species=0.30, 
                threshold_genus=0.20,
                threshold_family=0.20,
                allowed_regions=allowed_regions
            )
            
            if not results:
                print(f"Obj {obj_id}: BioCLIP returned no results.")
                continue
                
            # Aggregate: Find MAX score result
            best_score = -1.0
            best_res = None
            
            for res in results:
                score = res.get("score", 0.0)
                if score > best_score:
                    best_score = score
                    best_res = res
            
            if not best_res:
                print(f"Obj {obj_id}: No valid result found.")
                continue

            best_label = best_res.get("label", "Unknown")
            
            msg = f"Obj {obj_id} Final: {best_label} ({best_score:.2f}) [Truncated: {edge_truncated}]"
            print(msg)
            
            object_metadata[obj_id] = {
                "label": str(best_label),
                "confidence": float(best_score),
                "edge_truncated": edge_truncated,
                "family": best_res.get("family_pred", ""),
                "genus": best_res.get("genus_pred", ""),
                "species": best_res.get("species_pred", ""),
                "rank": best_res.get("rank", "unknown"),
                "is_marine": best_res.get("is_marine", False),
                "in_region": best_res.get("in_region", False)
            }
            
            # --- DEBUG: Save Best Crop ---
            if args.save_debug_crops:
                try:
                    debug_dir = os.path.join(cwd, "assets", "debug_crops")
                    if not os.path.exists(debug_dir):
                        os.makedirs(debug_dir)
                    
                    best_idx = results.index(best_res)
                    
                    if best_idx != -1:
                        best_crop_rgb = final_candidates[best_idx]
                        best_crop_bgr = cv2.cvtColor(best_crop_rgb, cv2.COLOR_RGB2BGR)
                        save_path = os.path.join(debug_dir, f"obj_{obj_id}_best.jpg")
                        cv2.imwrite(save_path, best_crop_bgr)
                except Exception as e:
                    print(f"  Failed to save debug crop: {e}")
            # -----------------------------
            
        # 7. Save Results
        final_data = []
        for p in prompts_list:
            oid = int(p["obj_id"])
            if oid in object_metadata:
                meta = object_metadata[oid]
                p["label"] = meta["label"]
                p["confidence"] = meta["confidence"]
                p["edge_truncated"] = meta["edge_truncated"]
                p["family"] = meta["family"]
                p["genus"] = meta["genus"]
                p["species"] = meta["species"]
                
                reasons = []
                if meta["confidence"] < 0.20:
                    reasons.append("Low confidence (<0.20)")
                if meta["edge_truncated"]:
                    reasons.append("Edge truncated")
                if not meta.get("is_marine", True):
                    reasons.append("Non-marine")
                # For region, we only flag if region filter was ACTIVE and it failed
                # But 'in_region' is set to True if no filter was active.
                # So checking False is enough if 'in_region' logic in bioclip_utils is correct.
                if not meta.get("in_region", True):
                    reasons.append("Region mismatch")
                    
                p["review_reason"] = reasons
                p["review_required"] = (len(reasons) > 0)
                
            final_data.append(p)
            
        with open(output_path, 'w') as f:
            json.dump(final_data, f, indent=2)
            
        print(f"Done. Saved propagated classification results to {output_path}")

    finally:
        print("Shutting down backend...")
        try:
            backend.shutdown()
        except:
            pass
            
if __name__ == "__main__":
    main()
