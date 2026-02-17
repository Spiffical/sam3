
import sys
import os
import torch
from PIL import Image
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from sam3.apps.interactive_video.bioclip_utils import load_bioclip_classifier, predict_hierarchical
from bioclip.predict import Rank

def debug_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    print(f"Loading {image_path}...")
    img = Image.open(image_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading BioCLIP on {device}...")
    classifier = load_bioclip_classifier(device)
    
    print("Running prediction (Top 10 Species)...")
    # Get raw predictions
    predictions = classifier.predict([image_path], Rank.SPECIES, k=10)
    
    if predictions and isinstance(predictions, list):
        # predictions is list of lists (one per image) or list of dicts (if one image)?
        # pybioclip behavior: predict([path]) returns [ [pred1, pred2...] ]
        preds = predictions[0]
        
        print("\n--- Top 10 Raw Predictions ---")
        print("\n--- Top 10 Raw Predictions ---")
        print(f"Debug: preds type: {type(preds)}")
        print(f"Debug preds content: {preds}")

        if isinstance(preds, list):
            for i, p in enumerate(preds):
                if isinstance(p, dict):
                    name = p.get('species')
                    score = p.get('score')
                    print(f"{i+1}. {name} ({score:.4f})")
                else:
                    print(f"{i+1}. {p} (Unexpected type)")
        elif isinstance(preds, dict):
             print("Single prediction returned:")
             print(preds)
            
    # Also run our wrapper to see what pass filters
    print("\n--- Hierarchical Wrapper Result (Marine Filtered) ---")
    # wrapper expects list of images or numpy arrays
    # But wrapper takes crops (arrays or PIL images).
    # predict_hierarchical handles list of PIL images.
    
    # We need to mock the allowed_regions if we want to test that too.
    allowed_regions = ["Pacific", "Canada", "British Columbia", "Washington"]
    print(f"Test Regions: {allowed_regions}")
    
    result = predict_hierarchical([img], allowed_regions=allowed_regions, top_k_check=20)
    print("Result:", result)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_bioclip_single_image.py <image_path>")
        # Default to obj 2
        default_path = "assets/debug_crops/obj_2_best.jpg"
        if os.path.exists(default_path):
            print(f"No path provided, using {default_path}")
            debug_image(default_path)
    else:
        debug_image(sys.argv[1])
