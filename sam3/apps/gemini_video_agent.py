
import os
import argparse
import json
import cv2
import torch
import numpy as np
import time
from PIL import Image
import google.generativeai as genai
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Adjust path to find sam3 if running from root
import sys
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from sam3.agent.agent_core import agent_inference
from sam3.agent.client_sam3 import sam3_inference, remove_overlapping_masks
from sam3.agent.viz import visualize
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.apps.interactive_video.backend import PredictorBackend

# -- Gemini Client Adapter --

def _convert_openai_messages_to_gemini(messages: List[Dict[str, Any]]):
    gemini_history = []
    
    for msg in messages:
        # Map roles: user->user, assistant->model. System handled separately.
        if msg["role"] == "system":
            continue
            
        role = "user" if msg["role"] == "user" else "model"
        parts = []
        
        content = msg.get("content", [])
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item["text"])
                    elif item.get("type") == "image":
                        # Load image
                        img_path = item["image"]
                        try:
                            # Gemini accepts PIL Image
                            img = Image.open(img_path)
                            parts.append(img)
                        except Exception as e:
                            print(f"[Warn] Could not load image {img_path} for Gemini: {e}")
                            
        if parts:
            gemini_history.append({"role": role, "parts": parts})
            
    return gemini_history

def get_gemini_client(api_key, model_name="gemini-2.5-flash"):
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"[Warn] Failed to initialize model '{model_name}'. Falling back to 'gemini-1.5-flash'. Error: {e}")
        model = genai.GenerativeModel("gemini-1.5-flash")
    return model

def gemini_send_request(messages, model):
    """
    Adapter function to match signature expected by agent_inference.
    messages: list of dicts (OpenAI style)
    """
    # Extract system prompt
    system_prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join([x["text"] for x in content if x.get("type") == "text"])
            system_prompt += str(content) + "\n\n"
            
    gemini_messages = _convert_openai_messages_to_gemini(messages)
    
    if not gemini_messages:
        return None
        
    # Prepend system prompt to the first message if it exists
    if system_prompt:
        if gemini_messages[0]['role'] == 'user':
            gemini_messages[0]['parts'].insert(0, system_prompt)
        else:
            # Should not happen typically as conversation starts with User after System
            pass

    # Retry loop for rate limiting
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Separate history and last message
            history = gemini_messages[:-1]
            last_msg = gemini_messages[-1]
            
            # Add a small delay to be polite and avoid burst limits
            if attempt == 0:
                time.sleep(2)
            
            chat = model.start_chat(history=history)
            response = chat.send_message(last_msg['parts'])
            
            try:
                text = response.text
                return text
            except Exception as e:
                # If we get a response but can't read text (e.g. finish reason STOP but empty),
                # it might be safety or just weirdness.
                print(f"[Error] Failed to get text from Gemini response (Attempt {attempt+1}/{max_retries}). Error: {e}")
                if response.candidates:
                    print(f"[Debug] Finish Reason: {response.candidates[0].finish_reason}")
                    print(f"[Debug] Safety Ratings: {response.prompt_feedback}")
                
                # If it's a finish_reason issue that isn't transient, retrying might not help unless we change params.
                # But if it's completely empty, maybe retry helps.
                time.sleep(5)
                continue
                
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Resource has been exhausted" in err_str:
                sleep_time = 20 * (attempt + 1)
                print(f"[Warn] Rate limit hit. Sleeping for {sleep_time} seconds before retry...")
                time.sleep(sleep_time)
                continue
            else:
                print(f"Gemini Request failed: {e}")
                # For non-rate-limit errors, maybe don't retry? Or retry strictly on network issues?
                # Let's retry just in case it's transient.
                time.sleep(5)
                continue
                
    return None

# -- SAM3 Service Adapter --

class LocalSam3Service:
    def __init__(self, processor, output_dir):
        self.processor = processor
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def call_service(self, image_path, text_prompt, output_folder_path=None):
        if output_folder_path is None:
            output_folder_path = self.output_dir
            
        print(f"Call SAM3 Service: {image_path}, prompt={text_prompt}")
        
        try:
            # Reusing code from sam3.agent.client_sam3.call_sam_service logic
            # but invoking local processor
            
            # 1. Run inference
            outputs = sam3_inference(self.processor, image_path, text_prompt)
            
            # 2. Cleanup
            outputs = remove_overlapping_masks(outputs)
            
            safe_prompt = text_prompt.replace("/", "_").replace(" ", "_")
            out_name = f"{os.path.basename(image_path)}_{safe_prompt}"
            
            output_image_path = os.path.join(output_folder_path, f"{out_name}.png")
            output_json_path = os.path.join(output_folder_path, f"{out_name}.json")
            
            outputs = {
                "original_image_path": image_path,
                "output_image_path": output_image_path,
                **outputs
            }
            
            # Sort by scores
            if "pred_scores" in outputs and outputs["pred_scores"]:
                score_indices = sorted(
                    range(len(outputs["pred_scores"])),
                    key=lambda i: outputs["pred_scores"][i],
                    reverse=True,
                )
                outputs["pred_scores"] = [outputs["pred_scores"][i] for i in score_indices]
                outputs["pred_boxes"] = [outputs["pred_boxes"][i] for i in score_indices]
                outputs["pred_masks"] = [outputs["pred_masks"][i] for i in score_indices]

            # Filter short masks
            valid_masks = []
            valid_boxes = []
            valid_scores = []
            for i, rle in enumerate(outputs["pred_masks"]):
                if len(rle) > 4:
                    valid_masks.append(rle)
                    valid_boxes.append(outputs["pred_boxes"][i])
                    valid_scores.append(outputs["pred_scores"][i])
            outputs["pred_masks"] = valid_masks
            outputs["pred_boxes"] = valid_boxes
            outputs["pred_scores"] = valid_scores
            
            # Save JSON
            with open(output_json_path, "w") as f:
                json.dump(outputs, f, indent=4)
                
            # Render
            viz = visualize(outputs)
            viz.save(output_image_path)
            
            return output_json_path
            
        except Exception as e:
            print(f"Error in SAM3 service: {e}")
            raise e

# -- Main Video Runner --

def mask_to_points(mask, num_points=1):
    """
    Convert a binary mask to a set of points inside the mask.
    Simple method: find coordinates where mask is True and random sample.
    """
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        return []
    
    # Select random points
    coords = list(zip(x_indices, y_indices))
    import random
    if len(coords) > num_points:
        return random.sample(coords, num_points)
    return coords

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--prompt", type=str, default="Identify and segment any biological creatures.", help="Initial prompt for Agent")
    parser.add_argument("--api_key", type=str, help="Gemini API Key")
    parser.add_argument("--output_dir", type=str, default="sam3_video_agent_out")
    parser.add_argument("--gpus", type=str, default="0", help="GPUs to use")
    parser.add_argument("--save_prompts", action="store_true", help="Save prompts to JSON and exit without propagation")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Please provide --api_key or set GEMINI_API_KEY/GOOGLE_API_KEY env var")
        return

    # 1. Setup Models
    print("Loading SAM3 Image Model (for Agent)...")
    
    # Try multiple paths for BPE vocab
    possible_bpe_paths = [
        "assets/bpe_simple_vocab_16e6.txt.gz",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/bpe_simple_vocab_16e6.txt.gz"),
        os.path.join(os.getcwd(), "assets/bpe_simple_vocab_16e6.txt.gz")
    ]
    
    bpe_path = None
    for p in possible_bpe_paths:
        if os.path.exists(p):
            bpe_path = p
            break
            
    if not bpe_path:
        print(f"Error: Could not find bpe_simple_vocab_16e6.txt.gz. Checked: {possible_bpe_paths}")
        return

    print(f"Using BPE path: {bpe_path}")
        
    image_model = build_sam3_image_model(bpe_path=bpe_path)
    image_processor = Sam3Processor(image_model, confidence_threshold=0.4)
    local_service = LocalSam3Service(image_processor, os.path.join(args.output_dir, "sam_service"))
    
    print("Loading Gemini...")
    gemini_model = get_gemini_client(api_key)
    
    # 2. Extract Frame 0
    cap = cv2.VideoCapture(args.video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return
    cap.release()
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_0_path = os.path.join(args.output_dir, "frame_0.jpg")
    os.makedirs(args.output_dir, exist_ok=True)
    Image.fromarray(frame_rgb).save(frame_0_path)
    print(f"Extracted Frame 0 to {frame_0_path}")
    
    # 3. Run Agent Inference on Frame 0
    print("Running Agent Inference on Frame 0...")
    
    from functools import partial
    send_req = partial(gemini_send_request, model=gemini_model)
    call_sam = local_service.call_service
    
    # 3. Run Agent Inference on Frame 0
    print("Running Agent Inference on Frame 0...")
    
    from functools import partial
    send_req = partial(gemini_send_request, model=gemini_model)
    call_sam = local_service.call_service
    
    try:
        history, final_outputs, rendered_img = agent_inference(
             img_path=frame_0_path,
             initial_text_prompt=args.prompt,
             send_generate_request=send_req,
             call_sam_service=call_sam,
             output_dir=os.path.join(args.output_dir, "agent_out"),
             debug=True
        )
    except Exception as e:
        print(f"Agent inference failed: {e}")
        return
    
    print("Agent inference complete.")
    
    selected_masks_rle = final_outputs.get("pred_masks", [])
    print(f"Agent selected {len(selected_masks_rle)} masks.")
    
    if not selected_masks_rle:
        print("No masks found by agent. Exiting video propagation.")
        return

    # 4. Initialize Video Predictor (Only if NOT saving prompts only)
    backend = None
    if not args.save_prompts:
        print("Initializing Video Predictor...")
        gpu_ids = [int(x) for x in args.gpus.split(",")] # Use args.gpus
        backend = PredictorBackend(gpu_ids=gpu_ids)
        
        # Start session
        img0 = cv2.imread(frame_0_path)
        height, width = img0.shape[:2]
        image_size = 1024 # Standard SAM image size
        
        try:
            session_id = backend.start_session(
                resource_path=args.video_path,
                image_size=image_size
            )
            print(f"Video Session Started: {session_id}")
        except Exception as e:
            print(f"Failed to start video session: {e}")
            return
    else:
        # Load frame 0 to get dimensions for creating prompts
        img0 = cv2.imread(frame_0_path)
        height, width = img0.shape[:2]
        print("Skipping Video Predictor initialization (save_prompts mode).")

    # Convert Agent Masks to Point Prompts for Video Predictor
    # The Agent returns RLE masks.
    # We need to turn them into prompts.
    # We could use the mask directly if PredictorBackend supports it, but add_point_prompt is the main API.
    # We can sample points from the mask.
    
    from pycocotools import mask as mask_util
    
    def decode_rle_to_mask(rle, h, w):
        # Debug
        # print(f"Debugging RLE: {type(rle)}")
        if isinstance(rle, str):
            # Probably just the counts string
            rle = {
                'counts': rle.encode('utf-8'),
                'size': [h, w]
            }
        elif isinstance(rle, dict) and 'counts' in rle:
            counts = rle['counts']
            if isinstance(counts, str):
                try:
                    rle['counts'] = counts.encode('utf-8')
                except Exception as e:
                    print(f"Error encoding counts: {e}")
        
        # mask_util.decode often prefers a list of RLEs
        try:
            m = mask_util.decode([rle])
            # m shape is (H, W, 1)
            return m[:, :, 0]
        except Exception as e:
            print(f"Decode failed: {e}. RLE: {rle}")
            # Try without list?
            m = mask_util.decode(rle)
            return m

    print("Processing masks to points...")
    
    # helper for finding positive point in mask
    def get_center_point(mask):
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) > 0:
            # Simple centroid or just middle point
            # Let's pick a random point or centroid
            idx = len(y_indices) // 2
            return (x_indices[idx], y_indices[idx])
        return None

    # Collect prompts for storage or propagation
    generated_prompts = []
    
    # Process each mask
    for i, rle in enumerate(selected_masks_rle):
        try:
            mask_array = decode_rle_to_mask(rle, height, width)
            # Handle if mask_array is 3D (H, W, 1) or 2D
            if len(mask_array.shape) == 3:
                mask_array = mask_array[:, :, 0]
        except Exception as e:
            print(f"Skipping mask {i+1} due to decode error: {e}")
            continue
            
        point = get_center_point(mask_array)
        if point:
            x, y = point
            # Points format: [(x, y, label)]
            prompt_data = {
                "frame_idx": 0,
                "obj_id": i + 1,
                "points": [[float(x), float(y), 1]], # 1=positive
                "label": 1,
                "source": "agent"
            }
            generated_prompts.append(prompt_data)
            
            if backend:
                print(f"Adding prompt for object {i+1} at {x}, {y}")
                backend.add_point_prompt(
                    session_id=session_id,
                    frame_idx=0,
                    obj_id=i+1,
                    points=[(float(x), float(y), 1)],
                    frame_size=(height, width) # Pass frame size
                )
        else:
            print(f"Could not find valid point for mask {i+1}")
            
    if args.save_prompts:
        # Save prompts to JSON and exit
        # Intelligent location: assets/annotations
        # Filename: [video_stem]_prompts.json
        
        video_stem = os.path.splitext(os.path.basename(args.video_path))[0]
        json_filename = f"{video_stem}_prompts.json"
        
        # We assume assets/annotations is the central store
        # But we should respect if the user is working in a completely different dir?
        # For this project, assets/annotations is the convention requested.
        save_dir = os.path.join("assets", "annotations")
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, json_filename)
        
        with open(save_path, 'w') as f:
            json.dump(generated_prompts, f, indent=2)
            
        print(f"Prompts saved to {save_path}")
        print("Exiting as --save_prompts was specified.")
        return

    # Propagate
    if backend:
        print("Running Propagation...")  
    # 6. Propagate
    print("Propagating masks...")
    prop_gen = backend.propagate({"session_id": session_id, "type": "propagate_in_video"})
    
    # Consume generator
    import tqdm
    results = {}
    for output in tqdm.tqdm(prop_gen):
        # output is dictionary with frame_index, outputs
        f_idx = output["frame_index"]
        results[f_idx] = output["outputs"]
    
    print("Propagation complete.")
    
    # 7. Render Video
    print("Rendering output video...")
    out_video_path = os.path.join(args.output_dir, "output_video.mp4")
    
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx in results:
            # Overlay masks
            # results[frame_idx] contains 'pred_masks', 'pred_scores'? 
            # Sam3VideoPredictor outputs from propagate_in_video?
            # It returns "outputs" which likely contains "masks" (N, H, W) or tracks
            # "track_ids", "pred_masks"
            
            res = results[frame_idx]
            if "pred_masks" in res:
                # Merge masks
                ms = res["pred_masks"] # Tensor or list
                if isinstance(ms, torch.Tensor):
                    ms = ms.cpu().numpy()
                
                # Simple Overlay
                # Create colored mask
                mask_overlay = np.zeros_like(frame)
                
                for obj_idx, m in enumerate(ms):
                    # m is (1, H, W)?
                    if len(m.shape) == 3: m = m[0]
                    
                    color = ((obj_idx * 50) % 255, (obj_idx * 100 + 50) % 255, (obj_idx * 150 + 100) % 255)
                    mask_overlay[m > 0] = color
                    
                frame = cv2.addWeighted(frame, 1, mask_overlay, 0.5, 0)
        
        writer.write(frame)
        frame_idx += 1
        
    cap.release()
    writer.release()
    print(f"Video saved to {out_video_path}")
    backend.shutdown()

if __name__ == "__main__":
    main()
