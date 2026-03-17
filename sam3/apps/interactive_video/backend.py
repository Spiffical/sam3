
from __future__ import annotations
import os
import threading
import traceback
import torch
from contextlib import nullcontext
from typing import Dict, List, Tuple

from sam3.model_builder import build_sam3_video_predictor
from .state_manager import _detach_outputs
import torch.nn.functional as F
import numpy as np

try:
    import gradio as gr
    GradioError = gr.Error
except Exception:
    gr = None
    GradioError = RuntimeError

def matrix_nms(masks_binary: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.7) -> torch.Tensor:
    """
    Memory efficient NMS using matrix multiplication on downsampled masks.
    masks_binary: (N, H, W) bool
    scores: (N,) float
    """
    if masks_binary.numel() == 0:
        return torch.empty(0, dtype=torch.long)
        
    # Downsample for IoU calculation to 64x64 or smaller
    target_size = 64
    N, H, W = masks_binary.shape
    
    # Calculate stride to get roughly target_size
    stride_h = max(1, H // target_size)
    stride_w = max(1, W // target_size)
    
    # Slice first (no memory copy if just viewing, but we convert to float next)
    # This keeps it small: (N, 64, 64) approx
    masks_small = masks_binary[:, ::stride_h, ::stride_w].float()
    
    flat_masks = masks_small.flatten(1) # (N, S)
    
    # Intersection = A @ B.T
    intersection = flat_masks @ flat_masks.t() # (N, N)
    
    # Area
    areas = flat_masks.sum(dim=1) # (N,)
    
    # Union = AreaA + AreaB - Intersection
    union = areas.unsqueeze(1) + areas.unsqueeze(0) - intersection
    
    ious = intersection / union.clamp(min=1e-6)
    
    # Standard NMS greedy
    # Remove diagonal and lower triangle
    ious = torch.triu(ious, diagonal=1)
    
    # Sort by score
    sorted_idx = torch.argsort(scores, descending=True)
    keep = []
    suppressed = torch.zeros(N, dtype=torch.bool, device=masks_binary.device)
    
    # CPU loop for NMS logic (N is ~3000, 3000 iterations is fast in python)
    # But checking against sorted tensor is faster
    
    ious_sorted = ious[sorted_idx][:, sorted_idx]
    
    # Actually, simpler to just use torchvision.ops.nms if we had boxes.
    # But we have masks. 
    # Let's do simple greedy loop.
    
    indices = sorted_idx.tolist()
    while indices:
        current = indices.pop(0)
        keep.append(current)
        
        # Find indices that have high IoU with current
        # Since we popped, we only need to check remaining
        if not indices:
             break
             
        # Get row 'current' from ious matrix (original ordering)
        # iou with remaining
        # We can implement this faster if we keep everything in tensor
        pass
        
    # Re-implementing greedy NMS loop in pure pytorch is slightly tricky to do efficiently without custom kernel
    # But since N=3000, we can use the CPU implementation from `torchvision` if boxes were involved.
    # Since we have IoU matrix, we can use the implementation from sam3.perflib.nms
    
    from sam3.perflib.nms import generic_nms
    # generic_nms takes IoU matrix and scores
    kept_inds = generic_nms(ious, scores, iou_threshold)
    return kept_inds

class PredictorBackend:
    def __init__(self, gpu_ids: List[int]):
        if not gpu_ids:
            raise RuntimeError("No GPU IDs supplied. Pass --gpus 0 (for example).")
        self.predictor = build_sam3_video_predictor(gpus_to_use=gpu_ids)
        self._lock = threading.Lock()
        self._is_shutdown = False
        if torch.cuda.is_available():
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                self._amp_dtype = torch.bfloat16
            else:
                self._amp_dtype = torch.float16
        else:
            self._amp_dtype = None

    def _guard(self):
        if self._is_shutdown:
            raise GradioError("Predictor was shut down. Restart the script to use it again.")

    def _amp_context(self):
        if self._amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self._amp_dtype)

    def start_session(
        self,
        resource_path: str,
        image_size: int = 1008,
        offload_video_to_cpu: bool = False,
    ) -> str:
        self._guard()
        with self._lock:
            with self._amp_context():
                response = self.predictor.handle_request(
                    request=dict(
                        type="start_session",
                        resource_path=resource_path,
                        image_size=image_size,
                        offload_video_to_cpu=offload_video_to_cpu
                    )
                )
        session_id = response["session_id"]
        
        # Initialize cache immediately to support warm-up
        state = self._get_inference_state(session_id)
        if state is not None:
            cache = state.get("cached_frame_outputs", {})
            if not cache:
                num_frames = state.get("num_frames", 0)
                state["cached_frame_outputs"] = {idx: {} for idx in range(num_frames)}

        # Warm-up: Perform a dummy click to initialize lazy components/compilation
        # This prevents the first user click from being slow.
        if str(os.environ.get("SAM3_DISABLE_WARMUP", "")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            print(f"[DEBUG] Warm-up disabled by SAM3_DISABLE_WARMUP for {session_id}.")
            return session_id
        try:
            print(f"[DEBUG] Warming up session {session_id}...")
            with self._lock:
                with self._amp_context():
                    # Add dummy prompt
                    self.predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=0,
                            obj_id=1,
                            points=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
                            point_labels=torch.tensor([1], dtype=torch.int32),
                        )
                    )
                    # Remove dummy object instead of resetting session
                    # This preserves the feature cache (fixing slow first click)
                    # and keeps the cached_frame_outputs structure (fixing AssertionError)
                    self.predictor.handle_request(
                         request=dict(
                            type="remove_object",
                            session_id=session_id,
                            obj_id=1,
                            is_user_action=False
                        )
                    )
            print(f"[DEBUG] Warm-up complete for {session_id}.")
        except Exception as e:
            # Keep serving even if warm-up fails, but log actionable diagnostics.
            print(f"[WARNING] Warm-up failed: {type(e).__name__}: {repr(e)}")
            traceback.print_exc()
            # Recovery: reset prompt/object state to avoid carrying partial warm-up side effects.
            try:
                with self._lock:
                    with self._amp_context():
                        self.predictor.handle_request(
                            request=dict(type="reset_session", session_id=session_id)
                        )
                print(f"[DEBUG] Warm-up recovery reset complete for {session_id}.")
            except Exception as reset_exc:
                print(
                    "[WARNING] Warm-up recovery reset failed: "
                    f"{type(reset_exc).__name__}: {repr(reset_exc)}"
                )
                traceback.print_exc()

        return session_id

    def reset_session(self, session_id: str) -> None:
        self._guard()
        with self._lock:
            with self._amp_context():
                self.predictor.handle_request(
                    request=dict(type="reset_session", session_id=session_id)
                )
        state = self._get_inference_state(session_id)
        if state is not None:
            cache = state.get("cached_frame_outputs", {})
            if not cache:
                num_frames = state.get("num_frames", 0)
                state["cached_frame_outputs"] = {idx: {} for idx in range(num_frames)}

    def close_session(self, session_id: str) -> None:
        self._guard()
        with self._lock:
            with self._amp_context():
                self.predictor.handle_request(
                    request=dict(type="close_session", session_id=session_id)
                )

    def add_point_prompt(
        self,
        session_id: str,
        frame_idx: int,
        obj_id: int,
        points: List[Tuple[float, float, int]],
        frame_size: Tuple[int, int],
    ) -> Dict:
        self._guard()
        width, height = frame_size
        if not points:
             coords = torch.empty((0, 2), dtype=torch.float32)
        else:
             coords = torch.tensor(
                [[x / width, y / height] for (x, y, _label) in points],
                dtype=torch.float32,
             )
        labels = torch.tensor([label for (_x, _y, label) in points], dtype=torch.int32)
        with self._lock:
            with self._amp_context():
                response = self.predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=frame_idx,
                        obj_id=obj_id,
                        points=coords,
                        point_labels=labels,
                    )
                )
        frame_idx = response.get("frame_index", frame_idx)
        self.mark_frame_has_prompt(session_id, frame_idx)
        return _detach_outputs(response["outputs"])

    def add_mask_prompt(
        self,
        session_id: str,
        frame_idx: int,
        obj_id: int,
        mask: np.ndarray | torch.Tensor,
    ) -> Dict:
        self._guard()
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask)
        elif isinstance(mask, torch.Tensor):
            mask_tensor = mask.detach().cpu()
        else:
            mask_tensor = torch.as_tensor(mask)

        if mask_tensor.ndim != 2:
            raise ValueError(
                f"Mask prompt must be 2D (H, W); got shape {tuple(mask_tensor.shape)}"
            )

        mask_tensor = (mask_tensor > 0).to(dtype=torch.bool)
        with self._lock:
            with self._amp_context():
                response = self.predictor.handle_request(
                    request=dict(
                        type="add_mask_prompt",
                        session_id=session_id,
                        frame_index=frame_idx,
                        obj_id=obj_id,
                        mask=mask_tensor,
                    )
                )
        frame_idx = response.get("frame_index", frame_idx)
        self.mark_frame_has_prompt(session_id, frame_idx)
        return _detach_outputs(response["outputs"])

    def propagate(self, request: Dict):
        self._guard()
        with self._lock:
            with self._amp_context():
                # Yield from generator to maintain context
                yield from self.predictor.handle_stream_request(request=request)

    def _get_inference_state(self, session_id: str):
        session = self.predictor._ALL_INFERENCE_STATES.get(session_id)
        if not session:
            return None
        return session.get("state")

    def mark_frame_has_prompt(self, session_id: str, frame_idx: int):
        state = self._get_inference_state(session_id)
        if state is None:
            return
        prev = state.get("previous_stages_out")
        if prev is None:
            return
        if 0 <= frame_idx < len(prev):
            prev[frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"
        cache = state.setdefault("cached_frame_outputs", {})
        cache.setdefault(frame_idx, {})

    def remove_object(self, session_id: str, obj_id: int):
        self._guard()
        with self._lock:
            with self._amp_context():
                self.predictor.handle_request(
                    request=dict(
                        type="remove_object",
                        session_id=session_id,
                        obj_id=obj_id,
                        is_user_action=True,
                    )
                )

    def shutdown(self) -> None:
        if self._is_shutdown:
            return
        with self._lock:
            self.predictor.shutdown()
            self._is_shutdown = True

    def segment_frame(
        self,
        session_id: str,
        frame_idx: int,
        grid_size: int = 32,
        chunk_size: int = 8
    ) -> Dict:
        """
        Run segmentation on the entire frame using a grid of point prompts.
        Returns a dict with 'masks', 'scores', 'bboxes' (all lists/arrays on CPU).
        Does NOT update the tracker state.
        Uses chunked inference to avoid OOM and proper broadcasting.
        """
        self._guard()
        import copy
        from sam3.model.geometry_encoders import Prompt
        
        state = self._get_inference_state(session_id)
        if state is None:
             raise ValueError("Session not found.")
        
        # Access the underlying model components
        inference_model = self.predictor.model # Sam3VideoInference
        video_base = inference_model # Sam3VideoInference IS Sam3VideoBase
        detector = video_base.detector # Sam3Image
        device = inference_model.device
        
        # 1. Prepare Text Features (from cache or compute)
        feature_cache = state["feature_cache"]
        input_batch = state["input_batch"]
        text_batch_key = tuple(input_batch.find_text_batch)
        if "text" in feature_cache and text_batch_key in feature_cache["text"]:
             text_outputs = feature_cache["text"][text_batch_key]
        else:
             with self._lock:
                with self._amp_context():
                     text_outputs = detector.backbone.forward_text(
                          input_batch.find_text_batch, device=device
                     )
        
        # 2. Run Image Backbone for CURRENT frame (Batch=1)
        # Fetch image tensor
        if isinstance(input_batch.img_batch, torch.Tensor):
             image = input_batch.img_batch[frame_idx].unsqueeze(0)
        else: 
             image = input_batch.img_batch[frame_idx].unsqueeze(0)
        
        image = image.to(dtype=torch.float32, device=device)
        
        with self._lock:
            with torch.no_grad():
                with self._amp_context():
                     image_out = detector.backbone.forward_image(image)
        
        # Combine into backbone_out for forward_grounding
        # features are indexable by [0]
        backbone_out = {
             **image_out,
             **text_outputs
        }
        
        # 3. Create grid points
        xs = torch.linspace(0, 1, grid_size, device=device)
        ys = torch.linspace(0, 1, grid_size, device=device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
        batch_points = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2) 
        total_points = batch_points.shape[0]
        print(f"[DEBUG] segment_frame: grid_size={grid_size}, total_points={total_points}")
        
        all_logits = []
        all_masks = []
        all_boxes = []
        
        # 4. Chunked Inference
        base_find_input = input_batch.find_inputs[frame_idx]
        
        for start in range(0, total_points, chunk_size):
             end = min(start + chunk_size, total_points)
             current_batch_size = end - start
             
             # Prepare Prompt for this chunk
             # Points: (SeqLen=1, Batch=Chunk, 2)
             chunk_p = batch_points[start:end].unsqueeze(0)
             chunk_labels = torch.ones(1, current_batch_size, dtype=torch.long, device=device)
             chunk_mask = torch.zeros(current_batch_size, 1, dtype=torch.bool, device=device)
             
             geo_prompt = Prompt(
                  point_embeddings=chunk_p,
                  point_labels=chunk_labels,
                  point_mask=chunk_mask
             )
             
             # Modify find_input to have correct img_ids length (all 0s)
             # this broadcasts the single image features to the prompt batch
             chunk_find_input = copy.copy(base_find_input)
             chunk_find_input.img_ids = torch.zeros(
                  current_batch_size, dtype=torch.long, device=device
             )
             if chunk_find_input.text_ids is not None and chunk_find_input.text_ids.numel() > 0:
                  # Broadcast text_ids to match the batch size of the chunk
                  if chunk_find_input.text_ids.shape[0] == 1:
                       chunk_find_input.text_ids = chunk_find_input.text_ids.repeat(current_batch_size)
                  elif chunk_find_input.text_ids.shape[0] != current_batch_size:
                       pass
             
             with self._lock:
                with torch.no_grad():
                    with self._amp_context():
                         out = detector.forward_grounding(
                              backbone_out=backbone_out,
                              find_input=chunk_find_input,
                              find_target=None,
                              geometric_prompt=geo_prompt
                         )
             
             # Filter masks: Only keep those that cover the prompt point
             # masks are logits (B, N, H, W). B=1 usually.
             
             masks_logits = out["pred_masks"].detach().cpu() # (B, N, H, W)
             logits_gpu = out["pred_logits"].detach().cpu() # (B, N)
             boxes_gpu = out.get("pred_boxes_xyxy", out.get("pred_boxes")).detach().cpu() # (B, N, 4)
             
             B, N, H, W = masks_logits.shape
             
             # chunk_p is (1, B, 2) normalized
             # We want pixel coordinates
             # We iterate over batch B
             
             batch_masks_bool = [] # List of (N_selected, H, W)
             batch_logits = []     # List of (N_selected)
             batch_boxes = []      # List of (N_selected, 4)
             
             for i in range(current_batch_size):
                  # Point for this item
                  # chunk_p shape (1, B, 2)
                  px, py = chunk_p[0, i].tolist()
                  
                  pixel_x = int(px * W)
                  pixel_y = int(py * H)
                  # Clamp
                  pixel_x = min(max(pixel_x, 0), W - 1)
                  pixel_y = min(max(pixel_y, 0), H - 1)
                  
                  # Check alignment: mask at (pixel_y, pixel_x) > 0
                  # masks_logits[i, :, pixel_y, pixel_x] -> (N,)
                  mask_at_point = masks_logits[i, :, pixel_y, pixel_x] > 0.0
                  
                  # Create mask of indices to keep
                  keep_indices = mask_at_point
                  
                  if keep_indices.sum() == 0:
                       continue
                  
                  batch_masks_bool.append(masks_logits[i][keep_indices] > 0.0)
                  batch_logits.append(logits_gpu[i][keep_indices])
                  batch_boxes.append(boxes_gpu[i][keep_indices])

             if batch_masks_bool:
                  all_logits.append(torch.cat(batch_logits, dim=0))
                  all_masks.append(torch.cat(batch_masks_bool, dim=0))
                  all_boxes.append(torch.cat(batch_boxes, dim=0))
        
        if not all_masks:
            return {
                 "scores": [],
                 "masks": np.zeros((0, *grid_x.shape), dtype=bool), # dummy
                 "bboxes": []
             }
             
        # 5. Aggregate and Format
        flat_logits = torch.cat(all_logits, dim=0) # (Total_Selected)
        flat_masks = torch.cat(all_masks, dim=0)   # (Total_Selected, H, W)
        flat_boxes = torch.cat(all_boxes, dim=0)   # (Total_Selected, 4)
        
        # Sigmoid scores
        scores = flat_logits.sigmoid()

        if scores.numel() > 0:
             print(f"[Stats] Raw Scores: min={scores.min():.3f}, max={scores.max():.3f}, mean={scores.mean():.3f}")
             print(f"[Stats] Count > 0.1: {(scores>0.1).sum().item()}")
             print(f"[Stats] Count > 0.5: {(scores>0.5).sum().item()}")
        
        # --- Run NMS ---
        # Filter low scores first to reduce NMS load
        # Default threshold 0.1 usually enough to kill noise
        # Ensure scores is 1D
        scores = scores.flatten()
        keep_filter = scores > 0.0
        if keep_filter.sum() == 0:
             # Fallback if everything is low confidence
             # return top 10 or empty?
             # Let's keep empty
             return {
                 "scores": [],
                 "masks": np.zeros((0, *flat_masks.shape[1:]), dtype=bool),
                 "bboxes": []
             }
             
        filtered_scores = scores[keep_filter]
        filtered_masks = flat_masks[keep_filter]
        filtered_boxes = flat_boxes[keep_filter]
        
        # We need to pass float masks to matrix_nms if we want interpolation, 
        # but 'filtered_masks' is bool.
        # matrix_nms handles bool by casting to float.
        # Ensure on CPU or GPU? If N is large, GPU is better for matmul.
        # But we detached to CPU earlier. Moving to GPU 3000x64x64 is fast.
        
        # Run NMS on GPU if available for speed
        if torch.cuda.is_available():
             nms_device = "cuda"
        else:
             nms_device = "cpu"
             
        # Move to device for NMS
        filtered_masks_dev = filtered_masks.to(nms_device)
        filtered_scores_dev = filtered_scores.to(nms_device)
        
        keep_indices = matrix_nms(filtered_masks_dev, filtered_scores_dev, iou_threshold=0.7)
        # keep_indices are indices into filtered_*
        
        final_scores = filtered_scores[keep_indices.cpu()]
        final_masks = filtered_masks[keep_indices.cpu()]
        final_boxes = filtered_boxes[keep_indices.cpu()]
        
        return {
            "scores": final_scores.float().numpy().tolist(),
            "masks": final_masks.numpy(), # Returns boolean numpy array
            "bboxes": final_boxes.float().numpy().tolist()
        }
