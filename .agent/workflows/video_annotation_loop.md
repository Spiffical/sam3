---
description: Semi-automated video annotation workflow using Gemini and SAM3
---

# Video Annotation Loop

This workflow allows you to use the Gemini MLLM agent to generate initial segmentation masks for a video, refine them interactively, and save the results for training or later processing.

## 1. Automated Initial Annotation (Agent Scraper)

Run the `gemini_video_agent.py` script with the `--save_prompts` flag. This will analyze the first frame, identify objects (e.g., "small creature"), and save the corresponding point prompts to `assets/annotations/[video_name]_prompts.json` automatically.

```bash
# Example for chinacreekclipped.mp4
# Adjust --prompt as needed for your specific clip content
.venv/bin/python sam3/apps/gemini_video_agent.py \
    --video_path assets/videos/chinacreekclipped.mp4 \
    --prompt "small creature" \
    --output_dir sam3_video_agent_out \
    --save_prompts
```

**Output:** `assets/annotations/chinacreekclipped_prompts.json`

## 2. Interactive Refinement (SAM3 UI)

Launch the interactive video UI to visualize and refine the masks.

```bash
python -m sam3.apps.interactive_video.app
```

1.  **Open Browser:** Go to the local URL (usually `http://127.0.0.1:7860`).
2.  **Upload Video:** Upload the same video (`assets/videos/chinacreekclipped.mp4`).
3.  **Start Session:** Click **Start session**.
    *   **Auto-Load:** The app will automatically detect the presence of `assets/annotations/chinacreekclipped_prompts.json` and load the initial positive points onto Frame 0.
    *   You should see green (positive) click points appear on Frame 0 corresponding to the agent's detections.
4.  **Refine:**
    *   Use the UI to add more points (Left Click) or remove false positives (Right Click).
    *   Use "Segment Frame (Everything)" to see if other objects were missed.
5.  **Propagate:**
    *   Click **Trigger Propagate** to track these objects across the video.
    *   Review the propagation. Correct specific frames if tracking fails.
6.  **Save Results:**
    *   Click **Save Annotations**.
    *   This will save a new JSON file (e.g., `assets/annotations/chinacreekclipped_refined_prompts.json`) containing your verified prompts.

## 3. Future Processing

The saved `_refined_prompts.json` contains the ground-truth point prompts for the video objects. You can use this file to:
*   Re-run the SAM3 video predictor in batch mode to generate final mask videos.
*   Generate training data (bounding boxes, masks) for fine-tuning SAM3 or YOLOv11.
