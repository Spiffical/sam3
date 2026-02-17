
"""
Launch the SAM 3 Gradio helper UI.
Run with: python -m sam3.apps.interactive_video.app
"""
import argparse
import atexit
import torch
import gradio as gr
from .backend import PredictorBackend
from .ui import build_interface

def parse_args():
    parser = argparse.ArgumentParser(description="Launch the SAM 3 Gradio helper UI.")
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma separated GPU ids to use (default: all visible GPUs).",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for Gradio.")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio.")
    parser.add_argument("--share", action="store_true", help="Enable public Gradio share link.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This UI currently requires CUDA. No GPUs were found.")
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
    
    print(f"Initializing PredictorBackend with GPUs: {gpu_ids}")
    backend = PredictorBackend(gpu_ids=gpu_ids)
    atexit.register(backend.shutdown)
    
    demo = build_interface(backend)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=["/tmp"],
    )

if __name__ == "__main__":
    main()
