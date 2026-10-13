#!/usr/bin/env python3
"""FastAPI Backend for OmniVoice Studio with Professional Voice Cloning."""

import os
import io
import json
import time
import base64
import uuid
import soundfile as sf
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

from omnivoice import OmniVoice, VoiceClonePrompt
from omnivoice.utils.common import get_best_device
from storage import storage

app = FastAPI(title="OmniVoice Studio", description="Fast, interactive voice generation and cloning")

# Global state
model = None
device = None
OUTPUT_DIR = "generated_samples"
PROFILES_DIR = "profiles"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs(PROFILES_DIR, exist_ok=True)

PROFILES_JSON = os.path.join(PROFILES_DIR, "profiles.json")

# Predefined Voice Design Presets
VOICE_PRESETS = {
    "en_female_british": {
        "name": "British Female (Emma)",
        "language": "en",
        "instruct": "female, young adult, moderate pitch, british accent",
        "sample_text": "Good day! It is a pleasure to demonstrate this voice model for you."
    },
    "en_male_american": {
        "name": "American Male (David)",
        "language": "en",
        "instruct": "male, middle-aged, low pitch, american accent",
        "sample_text": "Hey everyone, this is a smooth American male voice ready for podcasts and presentations."
    },
    "en_female_indian": {
        "name": "Indian English Female (Priya)",
        "language": "en",
        "instruct": "female, young adult, moderate pitch, indian accent",
        "sample_text": "Hello! Welcome to this multilingual voice test. How clearly does this sound to you?"
    },
    "hi_female_natural": {
        "name": "Hindi Female (Aditi)",
        "language": "hi",
        "instruct": "female, young adult, moderate pitch",
        "sample_text": "नमस्ते! ओम्नीवॉयस में आपका स्वागत है। यह आवाज़ बहुत ही स्पष्ट और मधुर है।"
    },
    "hi_male_deep": {
        "name": "Hindi Male (Rajesh)",
        "language": "hi",
        "instruct": "male, middle-aged, low pitch",
        "sample_text": "नमस्कार दोस्तों! आज हम इस बहुभाषी मॉडल का परीक्षण कर रहे हैं।"
    },
    "en_whisper": {
        "name": "Whisper Style (Soft)",
        "language": "en",
        "instruct": "female, young adult, whisper",
        "sample_text": "Shh, listen closely... this is synthesized using whisper voice style."
    }
}

# Guided calibration scripts
CALIBRATION_SCRIPTS = [
    {
        "id": "en_1",
        "lang": "en",
        "title": "English - Balanced Phonetics",
        "text": "The quick brown fox jumps over the lazy dog. Voice cloning technology enables natural communication across languages."
    },
    {
        "id": "en_2",
        "lang": "en",
        "title": "English - Professional Speech",
        "text": "Good communication is at the heart of every great idea. Today, we are testing advanced neural speech synthesis."
    },
    {
        "id": "en_3",
        "lang": "en",
        "title": "English - Conversational Tone",
        "text": "Artificial intelligence and modern voice models have transformed how humans interact with technology every day."
    },
    {
        "id": "hi_1",
        "lang": "hi",
        "title": "Hindi - सरल व स्पष्ट (Clear & Natural)",
        "text": "नमस्ते, आज हम आधुनिक आवाज़ क्लोनिंग तकनीक का परीक्षण कर रहे हैं। यह आवाज़ साफ़ और स्पष्ट है।"
    },
    {
        "id": "hi_2",
        "lang": "hi",
        "title": "Hindi - पेशेवर वक्तव्य (Professional Statement)",
        "text": "सफलता केवल मेहनत से ही नहीं, बल्कि सही दिशा और स्पष्ट संवाद से भी मिलती है।"
    },
    {
        "id": "hi_3",
        "lang": "hi",
        "title": "Hindi - सामान्य बातचीत (Conversational)",
        "text": "ओम्नीवॉयस मॉडल विभिन्न भाषाओं में स्वाभाविक और सजीव स्वर उत्पन्न करने में पूरी तरह सक्षम है।"
    }
]

# In-memory prompt cache: profile_id -> VoiceClonePrompt
prompt_cache = {}

def get_profiles_data() -> list:
    if os.path.exists(PROFILES_JSON):
        try:
            with open(PROFILES_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_profiles_data(profiles: list):
    with open(PROFILES_JSON, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

def seed_default_profiles(m):
    """Seed initial profiles from already generated samples if empty."""
    existing = get_profiles_data()
    if existing:
        return

    seeds = [
        {
            "id": "seed_british_female",
            "name": "Emma (British Female Cloned)",
            "lang": "en",
            "audio": "generated_samples/english_female_british.wav",
            "transcript": "Hello, welcome to OmniVoice! Zero-shot speech synthesis can now produce remarkably natural voices."
        },
        {
            "id": "seed_hindi_female",
            "name": "Aditi (Hindi Female Cloned)",
            "lang": "hi",
            "audio": "generated_samples/hindi_female_natural.wav",
            "transcript": "नमस्ते! ओम्नीवॉयस मॉडल में आपका स्वागत है। यह हिंदी में बहुत ही स्पष्ट और प्राकृतिक आवाज़ उत्पन्न कर सकता है।"
        }
    ]

    new_profiles = []
    for s in seeds:
        if os.path.exists(s["audio"]):
            pt_path = os.path.join(PROFILES_DIR, f"{s['id']}.pt")
            try:
                prompt = m.create_voice_clone_prompt(ref_audio=s["audio"], ref_text=s["transcript"])
                prompt.save(pt_path)
                data, sr = sf.read(s["audio"])
                dur = round(len(data) / sr, 2)
                rms = round(float(np.sqrt(np.mean(data**2))), 3)
                new_profiles.append({
                    "id": s["id"],
                    "name": s["name"],
                    "language": s["lang"],
                    "transcript": s["transcript"],
                    "audio_url": f"/api/audio/{os.path.basename(s['audio'])}",
                    "prompt_file": pt_path,
                    "duration": dur,
                    "rms": rms,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            except Exception as e:
                print(f"[!] Warning seeding profile {s['id']}: {e}")

    if new_profiles:
        save_profiles_data(new_profiles)

def load_omnivoice_model():
    global model, device
    if model is not None:
        return model

    device = get_best_device()
    model_path = os.path.abspath("checkpoints/OmniVoice")
    if not os.path.exists(model_path):
        model_path = "k2-fsa/OmniVoice"

    print(f"[*] Initializing OmniVoice on device: {device}...")
    t0 = time.time()
    load_device_map = "cpu" if device == "mps" else device
    model = OmniVoice.from_pretrained(
        model_path,
        device_map=load_device_map,
        dtype=torch.float32,
    )
    if device == "mps":
        model = model.to("mps")
    print(f"[✓] Model ready in {time.time() - t0:.2f}s!")

    seed_default_profiles(model)
    return model

@app.on_event("startup")
async def startup_event():
    load_omnivoice_model()

class GenerateRequest(BaseModel):
    text: str
    preset_id: Optional[str] = None
    profile_id: Optional[str] = None
    language: Optional[str] = "en"
    instruct: Optional[str] = None
    num_step: Optional[int] = 16
    speed: Optional[float] = 1.0

@app.get("/api/presets")
def get_presets():
    return JSONResponse(VOICE_PRESETS)

@app.get("/api/calibration_scripts")
def get_calibration_scripts():
    return JSONResponse(CALIBRATION_SCRIPTS)

@app.get("/api/profiles")
def get_profiles():
    profiles = get_profiles_data()
    return JSONResponse(profiles)

@app.delete("/api/profiles/{profile_id}")
def delete_profile(profile_id: str):
    profiles = get_profiles_data()
    for p in profiles:
        if p["id"] == profile_id:
            storage.delete_blob(p.get("audio_url", ""), folder="profiles")
            if os.path.exists(p.get("prompt_file", "")):
                try:
                    os.remove(p["prompt_file"])
                except Exception:
                    pass
    updated = [p for p in profiles if p["id"] != profile_id]
    save_profiles_data(updated)
    if profile_id in prompt_cache:
        del prompt_cache[profile_id]
    return {"success": True, "deleted": profile_id}

@app.post("/api/enroll_voice")
async def enroll_voice(
    file: UploadFile = File(...),
    name: str = Form("My Voice"),
    language: str = Form("en"),
    transcript: str = Form("")
):
    """Enroll a new cloned voice with verification and pre-cached prompt."""
    m = load_omnivoice_model()

    profile_id = f"voice_{uuid.uuid4().hex[:8]}"
    audio_filename = f"{profile_id}.wav"
    audio_path = os.path.join(PROFILES_DIR, audio_filename)
    prompt_path = os.path.join(PROFILES_DIR, f"{profile_id}.pt")

    # Read bytes and validate
    content = await file.read()
    try:
        data, sr = sf.read(io.BytesIO(content), dtype="float32", always_2d=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio format. Please provide WAV or PCM audio: {e}")

    # Convert to mono if needed
    if data.shape[1] > 1:
        data = np.mean(data, axis=1, keepdims=True)

    duration = float(len(data) / sr)
    rms = float(np.sqrt(np.mean(data**2)))

    # Verification checks
    if duration < 2.0:
        raise HTTPException(
            status_code=400,
            detail=f"Audio is too short ({duration:.1f}s). Please record at least 3 to 10 seconds of speech for accurate voice cloning."
        )
    if rms < 0.008:
        raise HTTPException(
            status_code=400,
            detail="Audio is too quiet or silent. Please check your microphone and speak clearly."
        )

    # Encode to clean WAV buffer
    wav_buf = io.BytesIO()
    sf.write(wav_buf, data, sr, format="WAV")
    wav_buf.seek(0)
    clean_wav_bytes = wav_buf.read()

    # Save to storage (local or Vercel Blob)
    audio_url = storage.save_audio(clean_wav_bytes, audio_filename, folder="profiles")

    clean_transcript = transcript.strip()
    if not clean_transcript:
        raise HTTPException(status_code=400, detail="Transcript is required for accurate voice enrollment.")

    # Create & cache OmniVoice VoiceClonePrompt
    t0 = time.time()
    try:
        prompt = m.create_voice_clone_prompt(ref_audio=audio_path, ref_text=clean_transcript)
        prompt.save(prompt_path)
        prompt_cache[profile_id] = prompt
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice enrollment failed: {str(e)}")

    enroll_time = round(time.time() - t0, 2)

    profile_entry = {
        "id": profile_id,
        "name": name.strip() or "Custom Voice",
        "language": language,
        "transcript": clean_transcript,
        "audio_url": audio_url,
        "prompt_file": prompt_path,
        "duration": round(duration, 2),
        "rms": round(rms, 3),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    profiles = get_profiles_data()
    profiles.insert(0, profile_entry)
    save_profiles_data(profiles)

    return {
        "success": True,
        "profile": profile_entry,
        "enroll_time": enroll_time,
        "message": f"Voice '{profile_entry['name']}' successfully enrolled and ready for instant generation!"
    }

@app.get("/api/profile_audio/{filename}")
def get_profile_audio(filename: str):
    path = os.path.join(PROFILES_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Profile audio file not found")
    return FileResponse(path, media_type="audio/wav")

@app.post("/api/generate")
async def generate_speech(req: GenerateRequest):
    global model
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    m = load_omnivoice_model()

    clean_text = req.text.strip()
    words = clean_text.split()
    is_short = len(words) <= 3 or len(clean_text) < 15

    # Determine generation mode: Cloned Voice Profile vs Voice Design Preset
    clone_prompt = None
    instruct = None
    lang = req.language or "en"

    if req.profile_id:
        # Load from in-memory cache or disk
        if req.profile_id in prompt_cache:
            clone_prompt = prompt_cache[req.profile_id]
        else:
            pt_path = os.path.join(PROFILES_DIR, f"{req.profile_id}.pt")
            if os.path.exists(pt_path):
                clone_prompt = VoiceClonePrompt.load(pt_path)
                prompt_cache[req.profile_id] = clone_prompt
            else:
                raise HTTPException(status_code=404, detail="Voice profile prompt not found.")
    elif req.preset_id in VOICE_PRESETS:
        preset = VOICE_PRESETS[req.preset_id]
        instruct = preset["instruct"]
        lang = preset["language"]
    elif req.instruct:
        instruct = req.instruct

    # Terminal punctuation normalization for smooth prosody
    if clean_text and clean_text[-1] not in ".!?,:;।~":
        if lang == "hi" or any('\u0900' <= c <= '\u097F' for c in clean_text):
            clean_text += "।"
        else:
            clean_text += "."

    num_step = max(8, min(req.num_step or 16, 32))
    speed = max(0.5, min(req.speed or 1.0, 2.0))

    # For short words (1-3 words), provide a safe duration floor
    duration = None
    postprocess = True
    if is_short and not clone_prompt:
        duration = 1.8 if len(words) >= 2 else 1.6
        postprocess = False

    t0 = time.time()
    try:
        if clone_prompt:
            audios = m.generate(
                text=clean_text,
                language=lang,
                voice_clone_prompt=clone_prompt,
                num_step=num_step,
                speed=speed,
                denoise=True,
            )
        else:
            audios = m.generate(
                text=clean_text,
                language=lang,
                instruct=instruct,
                num_step=num_step,
                speed=speed,
                duration=duration,
                postprocess_output=postprocess,
                denoise=True,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    gen_time = time.time() - t0
    audio_dur = len(audios[0]) / m.sampling_rate

    # Write WAV to buffer and disk
    buf = io.BytesIO()
    sf.write(buf, audios[0], m.sampling_rate, format="WAV")
    buf.seek(0)
    wav_bytes = buf.read()

    filename = f"gen_{uuid.uuid4().hex[:8]}.wav"
    audio_url = storage.save_audio(wav_bytes, filename, folder="generated_samples")

    b64_audio = base64.b64encode(wav_bytes).decode("utf-8")

    return {
        "success": True,
        "audio_base64": f"data:audio/wav;base64,{b64_audio}",
        "audio_url": audio_url,
        "gen_time": round(gen_time, 2),
        "duration": round(audio_dur, 2),
        "rtf": round(gen_time / max(audio_dur, 0.001), 2),
        "text": req.text,
        "filename": filename
    }

@app.get("/api/audio/{filename}")
def get_audio_file(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path, media_type="audio/wav")

# Serve static web frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=7860, reload=False)
