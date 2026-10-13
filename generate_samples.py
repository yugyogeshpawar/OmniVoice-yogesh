#!/usr/bin/env python3
"""Generate sample audios in English and Hindi with different voices using OmniVoice."""

import os
import time
import soundfile as sf
import torch

from omnivoice import OmniVoice
from omnivoice.utils.common import get_best_device

def main():
    output_dir = "generated_samples"
    os.makedirs(output_dir, exist_ok=True)

    device = get_best_device()
    model_path = os.path.abspath("checkpoints/OmniVoice")
    print(f"[*] Using local model path: {model_path}")

    dtype = torch.float32

    start_time = time.time()
    # On Mac/MPS, loading directly with device_map="mps" in accelerate deadlocks.
    # Loading on CPU first and calling .to(device) loads in 2 seconds and runs smoothly.
    load_device_map = "cpu" if device == "mps" else device
    model = OmniVoice.from_pretrained(
        model_path,
        device_map=load_device_map,
        dtype=dtype,
    )
    if device == "mps":
        model = model.to("mps")
    print(f"[✓] Model loaded in {time.time() - start_time:.2f} seconds on {device}.")

    test_cases = [
        {
            "id": "english_female_british",
            "title": "English Female (British Accent)",
            "text": "Hello, welcome to OmniVoice! Zero-shot speech synthesis can now produce remarkably natural voices.",
            "language": "en",
            "instruct": "female, young adult, moderate pitch, british accent",
        },
        {
            "id": "english_male_american",
            "title": "English Male (American Accent, Low Pitch)",
            "text": "Hey there! This is an American male voice generated with a deeper pitch. It sounds solid and confident.",
            "language": "en",
            "instruct": "male, middle-aged, low pitch, american accent",
        },
        {
            "id": "english_female_indian",
            "title": "English Female (Indian Accent)",
            "text": "Hello everyone, this is an Indian accent English voice test with natural conversational pacing.",
            "language": "en",
            "instruct": "female, young adult, indian accent",
        },
        {
            "id": "hindi_female_natural",
            "title": "Hindi Female (Natural / Young Adult)",
            "text": "नमस्ते! ओम्नीवॉयस मॉडल में आपका स्वागत है। यह हिंदी में बहुत ही स्पष्ट और प्राकृतिक आवाज़ उत्पन्न कर सकता है।",
            "language": "hi",
            "instruct": "female, young adult, moderate pitch",
        },
        {
            "id": "hindi_male_deep",
            "title": "Hindi Male (Deep Pitch)",
            "text": "नमस्कार दोस्तों! आज हम इस मॉडल की बहुभाषी क्षमता की जाँच कर रहे हैं। क्या यह आवाज़ आपको पेशेवर लगती है?",
            "language": "hi",
            "instruct": "male, low pitch",
        },
    ]

    print(f"\n[*] Starting generation of {len(test_cases)} sample audios...")

    for i, item in enumerate(test_cases, 1):
        filename = os.path.join(output_dir, f"{item['id']}.wav")
        print(f"\n[{i}/{len(test_cases)}] Generating: {item['title']}...")
        print(f"    Text: {item['text']}")
        print(f"    Instruct: {item['instruct']}")

        t0 = time.time()
        audios = model.generate(
            text=item["text"],
            language=item["language"],
            instruct=item["instruct"],
            num_step=20,  # 20 steps: fast & high quality
            denoise=True,
        )
        duration = len(audios[0]) / model.sampling_rate
        elapsed = time.time() - t0

        sf.write(filename, audios[0], model.sampling_rate)
        print(f"    [✓] Saved to {filename} ({duration:.2f}s audio generated in {elapsed:.2f}s, RTF: {elapsed/duration:.2f})")

    print("\n[✓] All sample audios generated successfully!")
    print(f"    Folder: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()
