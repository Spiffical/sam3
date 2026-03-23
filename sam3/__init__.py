# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

__version__ = "0.1.0"


def build_sam3_image_model(*args, **kwargs):
    """Lazily import the heavy model builder so lightweight utilities can import sam3."""
    from .model_builder import build_sam3_image_model as _build_sam3_image_model

    return _build_sam3_image_model(*args, **kwargs)


__all__ = ["build_sam3_image_model"]
