#!/usr/bin/env python3
"""
Legacy wrapper for SAM 3 Video Clicker.
This tool has been moved to sam3.apps.interactive_video.
"""
import sys
import os

# Ensure sam3 is in path if running from examples folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sam3.apps.interactive_video.app import main

if __name__ == "__main__":
    print("-" * 60)
    print("NOTE: The interactive video UI has been moved to:")
    print("      sam3.apps.interactive_video")
    print("      You can run it directly with:")
    print("      python -m sam3.apps.interactive_video.app")
    print("-" * 60)
    main()
