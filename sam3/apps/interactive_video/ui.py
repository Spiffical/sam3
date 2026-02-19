
from __future__ import annotations
import os
import shutil
import json
from pathlib import Path
from typing import Dict, Optional, List
import gradio as gr
import numpy as np

from .backend import PredictorBackend
from .state_manager import (
    _default_state, _ensure_display_dir, _get_session_data, _refresh_display_frames,
    _build_frames_payload, _detach_outputs, _format_object_summary, _format_click_summary,
    _SESSION_CACHE
)
from .video_loader import _load_frames, _copy_frame_uploads
from .assets import PLAYER_HTML, PLAYER_JS
try:
    from .bioclip_utils import classify_crops
except Exception:
    classify_crops = None

def on_run_bioclip(state_dict: Dict, candidates_str: str):
    if classify_crops is None:
         return state_dict, "BioCLIP is not installed in this environment."

    if state_dict is None or state_dict["session_id"] is None:
         raise gr.Error("No active session.")
    
    candidates = [c.strip() for c in candidates_str.split(",") if c.strip()]
    if not candidates:
         return state_dict, "Please provide comma-separated candidate labels."
         
    session_data = _get_session_data(state_dict["session_id"])
    frames = session_data.get("frames")
    if not frames:
         raise gr.Error("No frames loaded.")
         
    # Find all tracked objects and their best frame (largest area)
    # obj_id -> (frame_idx, area, bbox)
    best_views = {}
    
    for frame_idx, outputs in state_dict.get("frame_outputs", {}).items():
         if "out_obj_ids" not in outputs: continue
         
         ids = outputs["out_obj_ids"]
         masks = outputs.get("out_binary_masks", [])
         boxes = outputs.get("out_boxes_xywh", [])
         
         for i, obj_id in enumerate(ids):
              mask = masks[i]
              area = np.sum(mask)
              if area == 0: continue
              
              if obj_id not in best_views or area > best_views[obj_id][1]:
                   best_views[obj_id] = (frame_idx, area, boxes[i])
                   
    if not best_views:
         return state_dict, "No tracked objects found."
         
    # Extract crops
    batch_crops = []
    batch_ids = []
    
    for obj_id, (f_idx, _, box) in best_views.items():
         frame = frames[f_idx] # RGB numpy
         x, y, w, h = box
         
         # Pad slightly?
         h_img, w_img = frame.shape[:2]
         pad = 0.1
         x1 = int(max(0, x - w*pad))
         y1 = int(max(0, y - h*pad))
         x2 = int(min(w_img, x + w*(1+pad)))
         y2 = int(min(h_img, y + h*(1+pad)))
         
         crop = frame[y1:y2, x1:x2]
         if crop.size == 0: continue
         
         batch_crops.append(crop)
         batch_ids.append(obj_id)
         
    if not batch_crops:
         return state_dict, "Could not extract valid crops."
         
    # Run classification
    try:
         results = classify_crops(batch_crops, candidates)
    except Exception as e:
         return state_dict, f"BioCLIP error: {e}"
         
    # Update metadata
    count = 0
    state_dict.setdefault("obj_metadata", {})
    for obj_id, (label, score) in zip(batch_ids, results):
         # Update if score is decent? Or just always update?
         # Let's include score in label for now
         state_dict["obj_metadata"][obj_id] = f"{label} ({score:.2f})"
         count += 1
         
    # Refresh display
    updated_indices = _refresh_display_frames(state_dict)
    frame_payload = _build_frames_payload(state_dict, updated_indices)
    
    msg = f"BioCLIP classified {count} objects."
    return (
        state_dict,
        msg,
        _format_object_summary(state_dict),
        _format_click_summary(state_dict),
        frame_payload
    )

def build_interface(backend: PredictorBackend):
    def _cleanup_dir(dir_path: Optional[str]):
        if dir_path and Path(dir_path).exists():
            shutil.rmtree(dir_path, ignore_errors=True)

    def _drop_session(state_dict: Dict):
        if state_dict["session_id"]:
            try:
                backend.close_session(state_dict["session_id"])
            except gr.Error:
                pass
        _cleanup_dir(state_dict.get("temp_dir"))
        _cleanup_dir(state_dict.get("display_dir"))
        return _default_state()

    def start_session(state_dict, mp4_file, jpeg_files, image_size, offload_video):
        # Yield initial update
        yield (
             gr.update(), # state
             gr.update(), # status
             gr.update(), # obj summary
             gr.update(), # click summary
             gr.update(), # frame payload
             gr.update(), # frame pointer
             gr.update(value="Starting...", interactive=False) # BUTTON
        )
        
        state_dict = state_dict or _default_state()
        if not mp4_file and not jpeg_files:
            # Must yield final reset if error
            yield (
             state_dict, "", "", "", "", "0", gr.update(value="Start session", interactive=True)
            )
            raise gr.Error("Upload an MP4 or select JPEG frames first.")
        if mp4_file and jpeg_files:
            raise gr.Error("Pick either the MP4 input or the JPEG folder, not both.")

        state_dict = _drop_session(state_dict)
        temp_dir = None
        resource_path: Optional[str] = None
        orig_name: Optional[str] = None

        if mp4_file:
            resource_path = str(Path(mp4_file.name))
            if hasattr(mp4_file, 'orig_name'):
                 orig_name = mp4_file.orig_name
            else:
                 orig_name = os.path.basename(mp4_file.name)
        else:
            jpeg_list = list(jpeg_files)
            if not jpeg_list:
                raise gr.Error("Provide at least one JPEG frame.")
            temp_dir = _copy_frame_uploads(jpeg_list)
            resource_path = temp_dir
            # For folders, maybe use folder name? Hard with list of files.
            # default to None or generic
            orig_name = "image_sequence"

        frames = _load_frames(resource_path)
        print(f"[DEBUG] _load_frames -> {len(frames)} frames from {resource_path}")
        
        try:
             session_id = backend.start_session(
                 resource_path, 
                 image_size=int(image_size), 
                 offload_video_to_cpu=offload_video
            )
        except Exception as e:
             raise gr.Error(f"Failed to start session: {e}")
        
        # Store heavy frames in global cache
        _SESSION_CACHE[session_id] = {
            "frames": frames,
            "display_frames": [],
        }
        
        state_dict.update(
            {
                "session_id": session_id,
                "video_path": resource_path,
                "orig_name": orig_name,
                "temp_dir": temp_dir,
                "display_size": (0, 0),
                "frame_outputs": {},
                "clicks": {},
                "click_sources": {},
                "current_frame": 0,
                "frame_size": (frames[0].shape[1], frames[0].shape[0]),
                "status": f"Session {session_id[:8]} ready on {len(frames)} frames.",
            }
        )

        print(f"[DEBUG] Loaded session {session_id} with {len(frames)} frames")
        
        # Check for existing annotations in assets/annotations
        if orig_name:
             video_stem = os.path.splitext(orig_name)[0]
             # Also handle if it has suffix like 'clipped' or similar if consistent
             annot_path = os.path.join("assets", "annotations", f"{video_stem}_prompts.json")
             
             if os.path.exists(annot_path):
                  print(f"[DEBUG] Found annotations at {annot_path}, loading...")
                  try:
                       with open(annot_path, 'r') as f:
                            annotations = json.load(f)
                       
                       for item in annotations:
                            frame_idx = item.get("frame_idx", 0)
                            obj_id = item.get("obj_id", 1)
                            points_raw = item.get("points", []) 
                            source = item.get("source", "agent") # Default to agent for auto-loads if unspecified
                            
                            # Load taxonomic label if available
                            tax_label = item.get("label") or item.get("class") or item.get("taxonomy")
                            # Ensure it's not the integer 1 we sometimes save for "label" in older versions unless it's a string
                            if tax_label and (isinstance(tax_label, str)):
                                state_dict.setdefault("obj_metadata", {})[obj_id] = tax_label
                            elif tax_label and isinstance(tax_label, int) and tax_label != 1:
                                # If it's an int class ID other than default 1, we could use it, but user asked for string "Crab (1)"
                                # effectively so we probably want a name.
                                # But let's just convert to str if truthy
                                state_dict.setdefault("obj_metadata", {})[obj_id] = str(tax_label)
                            
                            frame_clicks = state_dict["clicks"].setdefault(frame_idx, {})
                            obj_clicks = frame_clicks.setdefault(obj_id, [])
                            
                            frame_sources = state_dict["click_sources"].setdefault(frame_idx, {})
                            obj_sources = frame_sources.setdefault(obj_id, [])
                            
                            current_points = []
                            for p in points_raw:
                                x, y, label = p
                                obj_clicks.append((x, y, int(label)))
                                current_points.append((x, y, int(label)))
                                obj_sources.append(source)
                            
                            backend.add_point_prompt(
                                session_id,
                                frame_idx,
                                obj_id,
                                current_points,
                                state_dict["frame_size"]
                            )
                            state_dict["frame_outputs"][frame_idx] = backend.add_point_prompt(
                                session_id,
                                frame_idx,
                                obj_id,
                                current_points,
                                state_dict["frame_size"]
                            )
                       
                       # Append status
                       state_dict["status"] += f" Loaded {len(annotations)} annotation objects."
                  except Exception as e:
                       print(f"[ERROR] Failed to load auto-annotations: {e}")
                       state_dict["status"] += f" Error loading annotations: {e}"

        updated = _refresh_display_frames(state_dict)
        print(f"[DEBUG] Refresh result indices={updated[:10]}")
        frame_payload = _build_frames_payload(state_dict)
        print(
            f"[DEBUG] start_session returning frame payload len={len(frame_payload)} pointer={state_dict['current_frame']}"
        )
        # Yield final result with button reset
        yield (
            state_dict,
            state_dict["status"],
            _format_object_summary(state_dict),
            _format_click_summary(state_dict),
            frame_payload,
            str(state_dict["current_frame"]),
            gr.update(value="Start session", interactive=True),
        )

    def handle_canvas_click(state_dict, click_payload: str):
        try:
            if state_dict is None or state_dict["session_id"] is None:
                raise gr.Error("Start a session first.")
            if not click_payload:
                return (
                    state_dict,
                    state_dict["status"],
                    _format_object_summary(state_dict),
                    _format_click_summary(state_dict),
                    _build_frames_payload(state_dict),
                )
            try:
                payload = json.loads(click_payload)
            except json.JSONDecodeError as exc:
                raise gr.Error(f"Invalid click payload: {exc}") from exc
            
            # print(f"[DEBUG] handle_canvas_click payload={payload}")

            frame_idx = int(payload.get("frame_index", state_dict.get("current_frame", 0)))
            
            session_data = _get_session_data(state_dict.get("session_id"))
            frames = session_data.get("frames")
            if not frames:
                raise gr.Error("Video frames are not loaded.")
            frame_idx = max(0, min(frame_idx, len(frames) - 1))

            rel_x = float(payload.get("rel_x", 0.0))
            rel_y = float(payload.get("rel_y", 0.0))
            label = int(payload.get("label", 1))
            obj_id = int(payload.get("obj_id", 1))

            width, height = state_dict["frame_size"]
            abs_x = rel_x * width
            abs_y = rel_y * height

            frame_clicks = state_dict["clicks"].setdefault(frame_idx, {})
            obj_clicks = frame_clicks.setdefault(obj_id, [])
            obj_clicks.append((abs_x, abs_y, label))
            
            # Store source
            frame_sources = state_dict.setdefault("click_sources", {}).setdefault(frame_idx, {})
            obj_sources = frame_sources.setdefault(obj_id, [])
            obj_sources.append("user_click")

            outputs = backend.add_point_prompt(
                state_dict["session_id"],
                frame_idx,
                obj_id,
                obj_clicks,
                state_dict["frame_size"],
            )
            state_dict["frame_outputs"][frame_idx] = outputs
            state_dict["current_frame"] = frame_idx
            state_dict["status"] = (
                f"Added {'+' if label else '-'} click on frame {frame_idx} (obj {obj_id})."
            )
            
            # Record history
            state_dict.setdefault("click_history", []).append((frame_idx, obj_id))

            updated_indices = _refresh_display_frames(state_dict, [frame_idx])
            frame_payload = _build_frames_payload(state_dict, updated_indices)
            return (
                state_dict,
                state_dict["status"],
                _format_object_summary(state_dict),
                _format_click_summary(state_dict),
                frame_payload,
            )
        except Exception as e:
            print(f"Error in handle_canvas_click: {e}")
            raise gr.Error(f"Error processing click: {e}")

    def on_undo(state_dict):
        if state_dict is None or state_dict["session_id"] is None:
            raise gr.Error("No active session.")

        history = state_dict.setdefault("click_history", [])
        if not history:
             return (
                 state_dict,
                 "Nothing to undo.",
                 _format_object_summary(state_dict),
                 _format_click_summary(state_dict),
                 _build_frames_payload(state_dict),
                 str(state_dict.get("current_frame", 0))
             )

        frame_idx, obj_id = history.pop()
        
        frame_clicks = state_dict["clicks"].get(frame_idx, {})
        obj_clicks = frame_clicks.get(obj_id, [])
        
        # We popped from history, now pop from actual clicks
        if obj_clicks:
             _removed = obj_clicks.pop()
        
        # Check if clicks remain for this object on this frame
        if not obj_clicks:
             # No more clicks for this object on this frame.
             # Check if there are clicks on ANY other frame for this object
             any_other_clicks = False
             for fidx, fclicks in state_dict["clicks"].items():
                 if obj_id in fclicks and fclicks.get(obj_id):
                      any_other_clicks = True
                      break
             
             if not any_other_clicks:
                  # Safe to remove object completely
                  print(f"[DEBUG] Undo: Object {obj_id} has no more clicks. Removing globally.")
                  backend.remove_object(state_dict["session_id"], obj_id)
                  # Remove from outputs
                  for fidx, outputs in state_dict["frame_outputs"].items():
                       if "out_obj_ids" in outputs and obj_id in outputs["out_obj_ids"]:
                            # Filter out this obj
                            indices = [i for i, oid in enumerate(outputs["out_obj_ids"]) if oid != obj_id]
                            # Reconstruct outputs
                            outputs["out_obj_ids"] = [outputs["out_obj_ids"][i] for i in indices]
                            outputs["out_probs"] = [outputs["out_probs"][i] for i in indices]
                            outputs["out_boxes_xywh"] = [outputs["out_boxes_xywh"][i] for i in indices]
                            if "out_binary_masks" in outputs:
                                 new_masks = []
                                 for i in indices:
                                      new_masks.append(outputs["out_binary_masks"][i])
                                 outputs["out_binary_masks"] = new_masks

                  msg = f"Undid last click. Object {obj_id} removed."
             else:
                  # It has clicks on other frames.
                  if frame_idx in state_dict["frame_outputs"]:
                        outputs = state_dict["frame_outputs"][frame_idx]
                        if "out_obj_ids" in outputs and obj_id in outputs["out_obj_ids"]:
                            indices = [i for i, oid in enumerate(outputs["out_obj_ids"]) if oid != obj_id]
                            outputs["out_obj_ids"] = [outputs["out_obj_ids"][i] for i in indices]
                            outputs["out_probs"] = [outputs["out_probs"][i] for i in indices]
                            outputs["out_boxes_xywh"] = [outputs["out_boxes_xywh"][i] for i in indices]
                            if "out_binary_masks" in outputs:
                                 new_masks = []
                                 for i in indices:
                                      new_masks.append(outputs["out_binary_masks"][i])
                                 outputs["out_binary_masks"] = new_masks
                  msg = f"Undid last click on frame {frame_idx}. Cleared local mask for obj {obj_id}."

        else:
             # Still has clicks, call add_point_prompt normally
             try:
                 outputs = backend.add_point_prompt(
                    state_dict["session_id"],
                    frame_idx,
                    obj_id,
                    obj_clicks,
                    state_dict["frame_size"]
                 )
                 state_dict["frame_outputs"][frame_idx] = outputs
                 msg = f"Undid last click on frame {frame_idx} (obj {obj_id})."
             except Exception as e:
                 print(f"Undo failed: {e}")
                 msg = f"Undo failed: {e}"

        state_dict["current_frame"] = frame_idx
        state_dict["status"] = msg
        
        updated_indices = _refresh_display_frames(state_dict, [frame_idx])
        frame_payload = _build_frames_payload(state_dict, updated_indices)
        
        return (
            state_dict,
            state_dict["status"],
            _format_object_summary(state_dict),
            _format_click_summary(state_dict),
            frame_payload,
            str(frame_idx)
        )

    def on_propagate(state_dict, pointer_value, max_frames):
        if state_dict is None or state_dict["session_id"] is None:
            raise gr.Error("No active session.")
        session_data = _get_session_data(state_dict["session_id"])
        frames = session_data.get("frames")
        if not frames:
            raise gr.Error("Load a session before propagating.")
        try:
            frame_idx = int(pointer_value)
        except (TypeError, ValueError):
            frame_idx = state_dict.get("current_frame", 0)
        frame_idx = max(0, min(frame_idx, len(frames) - 1))
        state_dict["current_frame"] = frame_idx

        req = {
            "type": "propagate_in_video",
            "session_id": state_dict["session_id"],
            "start_frame_idx": frame_idx,
        }
        if max_frames is not None and int(max_frames) > 0:
            req["max_frame_num_to_track"] = int(max_frames)

        print(f"[DEBUG] Propagation request {req}")
        
        # Stream responses to update progress
        responses = []
        generator = backend.propagate(req)
        
        # Estimate total frames. User reports full video propagation, so we use len(frames).
        total_frames = len(frames)
        if max_frames is not None and int(max_frames) > 0:
            total_frames = min(total_frames, int(max_frames))
            
        count = 0
        for resp in generator:
            responses.append(resp)
            count += 1
            if total_frames > 0:
                # Clamp count to total_frames for display to avoid confusing "303/221"
                display_count = min(count, total_frames)
                msg = f"Propagating frame {display_count}/{total_frames}..."
                yield (
                    state_dict,
                    msg, # Update status box
                    gr.update(), # object_summary
                    gr.update(), # frame_data_box
                    gr.update(), # frame_pointer_box
                )

        for resp in responses:
            state_dict["frame_outputs"][resp["frame_index"]] = _detach_outputs(
                resp["outputs"]
            )
        state_dict["status"] = f"Propagation complete. Propagated {len(responses)} frames starting at {frame_idx}."
        updated_indices = _refresh_display_frames(state_dict)
        frame_payload = _build_frames_payload(state_dict, updated_indices)
        yield (
            state_dict,
            state_dict["status"],
            _format_object_summary(state_dict),
            frame_payload,
            str(frame_idx),
        )

    def on_reset_session(state_dict):
        if state_dict is None or state_dict["session_id"] is None:
            raise gr.Error("No session to reset.")
        backend.reset_session(state_dict["session_id"])
        state_dict["frame_outputs"] = {}
        state_dict["clicks"] = {}
        state_dict["current_frame"] = 0
        _cleanup_dir(state_dict.get("display_dir"))
        state_dict["display_dir"] = None
        state_dict["status"] = "Session reset. All prompts cleared."
        updated_indices = _refresh_display_frames(state_dict)
        frame_payload = _build_frames_payload(state_dict, updated_indices)
        return (
            state_dict,
            state_dict["status"],
            _format_object_summary(state_dict),
            _format_click_summary(state_dict),
            frame_payload,
            "0",
        )

    def on_close_session(state_dict):
        if state_dict is None:
            new_state = _default_state()
            frame_payload = _build_frames_payload(new_state)
            return (
                new_state,
                "Session already closed.",
                "No tracked objects on this frame yet.",
                "No clicks logged on this frame.",
                frame_payload,
                "0",
                gr.update(value="Start session", interactive=True),
            )
        _cleanup_dir(state_dict.get("temp_dir"))
        if state_dict.get("session_id"):
            backend.close_session(state_dict["session_id"])
        new_state = _default_state()
        frame_payload = _build_frames_payload(new_state)
        return (
            new_state,
            "Session closed. You can upload a new video.",
            _format_object_summary(new_state),
            _format_click_summary(new_state),
        )

    def on_save_annotations(state_dict):
        if state_dict is None or state_dict["session_id"] is None:
            raise gr.Error("No active session.")
            
        orig_name = state_dict.get("orig_name")
        if not orig_name:
             return (state_dict, "No original filename found, cannot determine save location.")
             
        # Construct filename
        # Intelligent location: assets/annotations
        base_name = os.path.splitext(orig_name)[0]
        save_dir = os.path.join("assets", "annotations")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{base_name}_refined_prompts.json")
        
        # Collect prompts
        # state_dict["clicks"] structure: {frame_idx: {obj_id: [(x, y, label), ...]}}
        output_data = []
        for frame_idx, objs in state_dict["clicks"].items():
            for obj_id, clicks in objs.items():
                if not clicks:
                    continue
                
                # Get sources
                # state_dict["click_sources"] structure: {frame_idx: {obj_id: ["agent", ...]}}
                obj_sources = state_dict.get("click_sources", {}).get(frame_idx, {}).get(obj_id, [])
                
                # Flatten clicks
                points_list = []
                final_point_sources = []
                
                for idx, c in enumerate(clicks):
                     points_list.append([c[0], c[1], c[2]])
                     if idx < len(obj_sources):
                         final_point_sources.append(obj_sources[idx])
                     else:
                         final_point_sources.append("unknown")
                
                # Determine overall source
                unique = set(final_point_sources)
                if len(unique) == 1:
                     overall_source = list(unique)[0]
                elif "user_click" in unique:
                     overall_source = "user_refined"
                else:
                     overall_source = "mixed"

                entry = {
                    "frame_idx": frame_idx,
                    "obj_id": obj_id,
                    "points": points_list,
                    "label": 1,
                    "source": overall_source,
                    "point_sources": final_point_sources
                }
                output_data.append(entry)
                
        try:
            with open(save_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            msg = f"Saved annotations to {save_path}"
        except Exception as e:
            msg = f"Failed to save: {e}"
            
        return state_dict, msg

    def on_export_yolo(state_dict):
        if state_dict is None or state_dict["session_id"] is None:
             raise gr.Error("No active session.")
        
        orig_name = state_dict.get("orig_name")
        if not orig_name:
             return (state_dict, "No original filename found.")
        
        stem = os.path.splitext(orig_name)[0]
        # YOLOv11 structure
        # dataset/
        #   images/
        #     train/
        #       frame_000.jpg
        #   labels/
        #     train/
        #       frame_000.txt
        #   data.yaml
        
        # Shared images directory
        # assets/yolo_images/{stem}/train/*.jpg
        shared_img_root = os.path.join("assets", "yolo_images", stem)
        shared_img_train = os.path.join(shared_img_root, "train")
        os.makedirs(shared_img_train, exist_ok=True)
        
        # Dataset directories
        base_dir_det = os.path.join("assets", "yolo_dataset_det", stem)
        base_dir_seg = os.path.join("assets", "yolo_dataset_seg", stem)
        
        for d in [base_dir_det, base_dir_seg]:
             # Create labels/train
             os.makedirs(os.path.join(d, "labels", "train"), exist_ok=True)
             
             # Create symlink for images
             # Link: assets/yolo_dataset_det/{stem}/images -> assets/yolo_images/{stem}
             # Target must contain 'train' folder
             link_path = os.path.join(d, "images")
             target_path = os.path.abspath(shared_img_root)
             
             if os.path.islink(link_path) or os.path.exists(link_path):
                  os.remove(link_path)
             
             os.symlink(target_path, link_path)
        
        session_data = _get_session_data(state_dict["session_id"])
        frames = session_data.get("frames")
        if not frames:
             raise gr.Error("No frames loaded.")
             
        frame_outputs = state_dict.get("frame_outputs", {})
        
        # Check coverage
        if len(frame_outputs) < len(frames):
             print(f"[WARNING] Exporting partial results: {len(frame_outputs)}/{len(frames)} frames have data.")
        
        count_labeled = 0
        import cv2
        
        # Iterate all frames
        for i, frame in enumerate(frames):
             if i not in frame_outputs:
                  continue
             
             outputs = frame_outputs[i]
             if not outputs or "out_binary_masks" not in outputs:
                  continue
             
             # Save Image ONCE to shared location
             bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
             img_name = f"{stem}_frame_{i:06d}.jpg"
             cv2.imwrite(os.path.join(shared_img_train, img_name), bgr_frame)
             
             # Process Labels
             masks = outputs["out_binary_masks"] # list of (H, W) bool/uint8
             obj_ids = outputs["out_obj_ids"]
             
             H, W = frame.shape[:2]
             
             lines_det = []
             lines_seg = []
             
             for mask, oid in zip(masks, obj_ids):
                  class_id = 0
                  
                  # 1. Detection (BBox)
                  ys, xs = np.where(mask > 0)
                  if len(ys) == 0:
                       continue
                  
                  x_min, x_max = xs.min(), xs.max()
                  y_min, y_max = ys.min(), ys.max()
                  
                  bbox_w = x_max - x_min
                  bbox_h = y_max - y_min
                  xc = x_min + bbox_w / 2
                  yc = y_min + bbox_h / 2
                  
                  norm_xc = xc / W
                  norm_yc = yc / H
                  norm_w = bbox_w / W
                  norm_h = bbox_h / H
                  
                  lines_det.append(f"{class_id} {norm_xc:.6f} {norm_yc:.6f} {norm_w:.6f} {norm_h:.6f}")
                  
                  # 2. Segmentation (Polygon)
                  mask_uint8 = mask.astype(np.uint8) * 255
                  contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                  
                  if contours:
                       c = max(contours, key=cv2.contourArea)
                       epsilon = 0.001 * cv2.arcLength(c, True)
                       c = cv2.approxPolyDP(c, epsilon, True)
                       
                       if len(c) >= 3:
                            points = c.reshape(-1, 2)
                            norm_points = []
                            for (px, py) in points:
                                 norm_points.append(f"{px / W:.6f} {py / H:.6f}")
                            
                            poly_str = " ".join(norm_points)
                            lines_seg.append(f"{class_id} {poly_str}")

             if lines_det or lines_seg:
                  lbl_name = f"{stem}_frame_{i:06d}.txt"
                  count_labeled += 1
                  
                  if lines_det:
                       with open(os.path.join(base_dir_det, "labels", "train", lbl_name), 'w') as f:
                            f.write("\n".join(lines_det))
                            
                  if lines_seg:
                       with open(os.path.join(base_dir_seg, "labels", "train", lbl_name), 'w') as f:
                            f.write("\n".join(lines_seg))
        
        # Write data.yaml for both
        for d in [base_dir_det, base_dir_seg]:
             yaml_path = os.path.join(d, "data.yaml")
             with open(yaml_path, 'w') as f:
                  f.write(f"path: {os.path.abspath(d)}\n")
                  f.write("train: images/train\n")
                  f.write("val: images/train\n")
                  f.write("nc: 1\n")
                  f.write("names: ['object']\n")
             
        msg = f"Exported shared images to assets/yolo_images/{stem} and datasets to assets/yolo_dataset_det|seg. Labeled {count_labeled} frames."
        return state_dict, msg

    def on_shutdown(state_dict):
        if state_dict is not None:
            _cleanup_dir(state_dict.get("temp_dir"))
        backend.shutdown()
        new_state = _default_state()
        frame_payload = _build_frames_payload(new_state)
        return (
            new_state,
            "Predictor shut down. Restart the script for a new session.",
            _format_object_summary(new_state),
            _format_click_summary(new_state),
            frame_payload,
            "0",
            gr.update(value="Start session", interactive=True),
        )

    # Inject JS into HTML
    final_html = PLAYER_HTML.replace("{PLAYER_JS}", PLAYER_JS)

    with gr.Blocks(title="SAM 3 Video Click Helper") as demo:
        state = gr.State(_default_state())
        gr.Markdown(
            """
            ### SAM 3 Helper
            1. Upload an MP4 **or** a batch of JPEG frames (select the folder).
            2. Configure optimization settings if needed (for long videos).
            3. Click **Start session** to spin up SAM 3 on that video.
            4. Click on frames to add positive (green) or negative (red) prompts, then hit **Propagate**.

            """
        )

        with gr.Row():
            mp4_input = gr.File(
                label="Upload MP4 video",
                file_types=[".mp4"],
                height=120,
            )
            jpeg_input = gr.File(
                label="Upload JPEG frames",
                file_types=["image"],
                file_count="multiple",
                height=180,
            )
        
        with gr.Accordion("Optimization settings (for long/large videos)", open=False):
             with gr.Row():
                  image_size_drop = gr.Dropdown(
                       choices=[1008],
                       value=1008,
                       label="Processing Size (Fixed)",
                       info="Model requires fixed resolution of 1008px."
                  )
                  offload_cpu_chk = gr.Checkbox(
                       label="Offload frames to CPU",
                       value=True,
                       info="Saves VRAM by keeping video features on CPU RAM. Slightly slower."
                  )

        with gr.Accordion("Taxonomic Analysis (BioCLIP-2)", open=False):
             with gr.Row():
                  bioclip_candidates = gr.Textbox(
                       label="Candidate Labels (comma separated)",
                       placeholder="e.g. Crab, Fish, Starfish",
                       scale=3
                  )
                  bioclip_btn = gr.Button("Run BioCLIP Classification", scale=1)

        with gr.Row():
            start_button = gr.Button("Start session", variant="primary")
            undo_button = gr.Button("Undo click")
            reset_button = gr.Button("Reset session", interactive=True)
            save_annotations_btn = gr.Button("Save Annotations")
            export_yolo_btn = gr.Button("Export YOLO Dataset (Det + Seg)")
            close_button = gr.Button("Close session")
            shutdown_button = gr.Button("Shutdown predictor", variant="stop")

        status_box = gr.Markdown(
            value=_default_state()["status"], 
            elem_id="sam3-status-box"
        )
        


        with gr.Row():
            gr.Number(
                label="Object ID",
                value=1,
                precision=0,
                elem_id="sam3-obj-id",
                interactive=True,
                scale=1,
            )
            max_frames_box = gr.Number(
                label="Max Frames (0 = all)",
                value=0,
                precision=0,
                elem_id="sam3-max-frames",
                interactive=True,
                scale=1,
            )
            with gr.Column(scale=2):
                gr.Markdown(
                    "**Controls:** Left-click to Add (+), Right-click to Remove (-)."
                )



        gr.HTML(
            value=final_html,
            elem_id="sam3-player-root",
            sanitize_html=False,
        )

        object_summary = gr.Markdown(_format_object_summary(_default_state()))
        click_summary = gr.Markdown(_format_click_summary(_default_state()))

        frame_data_box = gr.Textbox(
            value='{"type":"clear"}',
            elem_id="sam3-frame-data",
            show_label=False,
            interactive=False,
        )
        frame_pointer_box = gr.Textbox(
            value="0",
            elem_id="sam3-frame-pointer",
            show_label=False,
            interactive=False,
        )
        click_payload_box = gr.Textbox(
            value="",
            elem_id="sam3-click-payload",
            show_label=False,
            interactive=False,
        )
        click_trigger = gr.Button("Submit Click", elem_id="sam3-click-trigger")
        propagate_trigger = gr.Button(
            "Trigger Propagate", elem_id="sam3-propagate-trigger"
        )

        start_button.click(
            fn=start_session,
            inputs=[state, mp4_input, jpeg_input, image_size_drop, offload_cpu_chk],
            outputs=[
                state,
                status_box,
                object_summary,
                click_summary,
                frame_data_box,
                frame_pointer_box,
                start_button,
            ],
            show_progress=True,
        )

        click_trigger.click(
            fn=handle_canvas_click,
            inputs=[state, click_payload_box],
            outputs=[
                state,
                status_box,
                object_summary,
                click_summary,
                frame_data_box,
            ]
        )
        
        propagate_trigger.click(
            fn=on_propagate,
            inputs=[state, frame_pointer_box, max_frames_box],
            outputs=[
                 state,
                 status_box,
                 object_summary,
                 frame_data_box,
                 frame_pointer_box,
            ],
            show_progress=True
        )

        bioclip_btn.click(
             fn=on_run_bioclip,
             inputs=[state, bioclip_candidates],
             outputs=[
                  state,
                  status_box,
                  object_summary,
                  click_summary,
                  frame_data_box
             ]
        )

        undo_button.click(
            on_undo,
            inputs=[state],
            outputs=[
                state,
                status_box,
                object_summary,
                click_summary,
                frame_data_box,
                frame_pointer_box,
            ]
        )

        reset_button.click(
            on_reset_session,
            inputs=[state],
             outputs=[
                state,
                status_box,
                object_summary,
                click_summary,
                frame_data_box,
                frame_pointer_box,
            ],
        )
        


        close_button.click(
            on_close_session,
            inputs=[state],
            outputs=[
                state,
                status_box,
                object_summary,
                click_summary,
                frame_data_box,
                frame_pointer_box,
                start_button,
            ],
        )

        shutdown_button.click(
            on_shutdown,
            inputs=[state],
            outputs=[
                state,
                status_box,
                object_summary,
                click_summary,
                frame_data_box,
                frame_pointer_box,
                start_button,
            ],
        )


        
        save_annotations_btn.click(
             on_save_annotations,
             inputs=[state],
             outputs=[
                  state,
                  status_box
             ]
        )
        
        export_yolo_btn.click(
             on_export_yolo,
             inputs=[state],
             outputs=[
                  state,
                  status_box
             ]
        )
        
    return demo
