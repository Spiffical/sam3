# Temporal Underwater SAM3 Pipeline Plan

## Problem
Single-seed-frame propagation misses creatures that enter later in a clip and can be derailed by corrupt/blank frames.

## Goals
1. Detect keyframes where new creatures are clearly visible.
2. Keep stable object IDs across multiple keyframe updates.
3. Avoid using corrupt/blank frames for tracking and output rendering.
4. Keep compatibility with the current `run_video_agent_openai.py` workflow.

## Phase 1 (Implemented)
1. Frame quality scan:
   - Detect near-black, near-white, and low-entropy low-variance frames.
   - Mark them as invalid.
2. Keyframe discovery:
   - Always include the first valid frame.
   - Add motion-based event candidates from frame-to-frame differences.
   - Enforce a minimum temporal gap between keyframes.
3. Multi-keyframe SAM3 agent loop:
   - Run the agent on each selected keyframe.
   - Convert selected masks to point prompts.
   - Assign object IDs by matching with existing masks on that frame (IoU).
   - Add prompts to tracker and re-propagate.
4. Corrupt-frame-aware export:
   - Optionally drop invalid frames from final rendered video.
   - Persist invalid-frame metadata in run outputs.
5. Compatibility fix:
   - Support both `start_frame_index` and `start_frame_idx` in predictor stream requests.

## Object-ID Consistency Rules
1. At each keyframe, decode new masks from agent output.
2. Compare each new mask against currently tracked masks on that frame.
3. Reuse existing ID if IoU >= threshold (default 0.3).
4. Otherwise allocate a new ID from an incrementing counter.
5. Never reuse removed IDs.

## Phase 2 (Implemented)
1. MLLM temporal discovery pass over short windows:
   - Uses windowed analysis and strict JSON parsing.
   - Returns event candidates with `first_seen_frame`, `best_visible_frame`, `confidence`.
2. One-image-compatible temporal context:
   - Optional collage mode encodes multiple sequential frames into one image.
   - Works with backends that allow only one image per request.
3. Domain-tuned discovery prompting:
   - Underwater-specific temporal discovery template:
     `sam3/agent/system_prompts/system_prompt_temporal_discovery_underwater.txt`
   - Strict JSON retry loop on parse failure to stabilize structured outputs.
3. Discovery modes:
   - `motion`: motion-only keyframe discovery.
   - `mllm`: MLLM-driven discovery (motion fallback if empty).
   - `hybrid`: MLLM primary, motion fill/fallback.
4. Artifacts:
   - `keyframe_discovery_motion.json`
   - `keyframe_discovery_mllm.json`
   - unified `keyframe_discovery.json`
5. Robust execution behavior:
   - Per-keyframe agent failures are logged and skipped (do not abort whole run).
   - Strengthened frame-quality checks reject near-black/near-white/low-information frames.

## Current Limits
1. MLLM event quality still depends on model reliability and context limits.
2. No dedicated local re-check loop yet for borderline events.
3. Temporal object lifecycle metadata export is still minimal.

## Phase 3 (Next)
1. Local re-check refinement for each MLLM event candidate (`±N` frame micro-window).
2. Event-aware bounded propagation windows (instead of always both directions full span).
3. Rich track table export (`first_seen`, `best_seen`, `last_seen`, confidence history).
4. Optional second-pass consistency sweep for ID drift correction.

## Recommended CLI Baseline
Use the temporal mode and drop invalid frames:

```bash
python nibi_model_compare/run_video_agent_openai.py \
  --video_path ... \
  --server_url ... \
  --model Qwen/Qwen3.5-27B \
  --prompt "identify and segment small creatures in the underwater scene" \
  --prompt-profile underwater \
  --temporal_keyframe_pipeline \
  --max_keyframes 6 \
  --min_keyframe_gap 24 \
  --drop_invalid_frames
```
