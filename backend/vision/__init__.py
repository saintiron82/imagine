"""
Vision Analysis Module - Phase 4: Descriptive Axis

This module provides AI-powered vision analysis for images:
- Caption generation (Florence-2/Qwen)
- Tag extraction
- OCR (text recognition)
- Object detection
- Style analysis
"""

from .base import BaseVisionAnalyzer

__all__ = ['BaseVisionAnalyzer', 'VisionAnalyzer']


def __getattr__(name):
    if name == 'VisionAnalyzer':
        from .analyzer import VisionAnalyzer
        return VisionAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
