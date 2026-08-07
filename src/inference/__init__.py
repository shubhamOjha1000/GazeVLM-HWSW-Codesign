"""Inference -- the gating module.

At runtime no frame is encoded and DINOv2 never runs: only the eye-velocity signal
streams in, the trained gaze encoder turns it into z_M, two heads produce S_frame and
S_gaze, and FilterFrameForVLM decides whether the frame reaches the VLM.
"""

from src.inference.gate import (DISCARD, SEND, Decision, GazeGate, StreamingGate,
                                filter_frame_for_vlm)

__all__ = ["GazeGate", "StreamingGate", "Decision", "filter_frame_for_vlm",
           "SEND", "DISCARD"]
