#!/usr/bin/env python3
"""Convenient voice cloning and testing script using OmniVoice."""

import argparse
import logging
import os
import sys
import soundfile as sf
import torch

from omnivoice import OmniVoice
from omnivoice.utils.common import get_best_device

def main():
    parser = argparse.ArgumentParser(description="OmniVoice Voice Cloning Test")
    parser.add_argument(
        "--ref_audio",
        type=str,
        required=True,
        help="Path to 3-10s clean audio of your voice (WAV/FLAC/MP3)",
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default=None,
        help="Exact transcript of the reference audio. If omitted, Whisper auto-transcribes it.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello! This is a test of my cloned voice using OmniVoice. How natural does this sound?",
        help="Text you want the cloned voice to speak.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cloned_voice_output.wav",
        help="Path for generated output audio (default: cloned_voice_output.wav)",
    )
    parser.add_argument(
        "--num_step",
        type=int,
        default=32,
        help="Diffusion steps (32 for high quality, 16 for faster speed)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (default: auto-detected, e.g. mps or cpu)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.ref_audio):
        print(f"Error: Reference audio file '{args.ref_audio}' does not exist.")
        sys.exit(1)

    default_model = os.path.abspath("checkpoints/OmniVoice") if os.path.exists("checkpoints/OmniVoice") else "k2-fsa/OmniVoice"

    device = args.device or get_best_device()
    print(f"[*] Detected device: {device}")
    print(f"[*] Loading OmniVoice model from: {default_model}")

    load_device_map = "cpu" if device == "mps" else device
    model = OmniVoice.from_pretrained(
        default_model,
        device_map=load_device_map,
        dtype=torch.float32,
    )
    if device == "mps":
        model = model.to("mps")

    print(f"[*] Generating speech for text:\n    \"{args.text}\"")
    audios = model.generate(
        text=args.text,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        num_step=args.num_step,
        denoise=True,
    )

    sf.write(args.output, audios[0], model.sampling_rate)
    print(f"[✓] Voice clone successfully created and saved to: {args.output} (Sampling rate: {model.sampling_rate}Hz)")

if __name__ == "__main__":
    main()
