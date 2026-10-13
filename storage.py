#!/usr/bin/env python3
"""Unified Storage Adapter for OmniVoice.

Supports:
- Local filesystem storage (development default)
- Vercel Blob Storage (activated when BLOB_READ_WRITE_TOKEN or VERCEL_ENV is set)
"""

import os
import io
import json
import logging
import urllib.request
import urllib.error
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Environment variable provided automatically by Vercel when Blob store is linked
VERCEL_BLOB_TOKEN = os.getenv("BLOB_READ_WRITE_TOKEN")
IS_VERCEL = bool(os.getenv("VERCEL") or os.getenv("VERCEL_ENV") or VERCEL_BLOB_TOKEN)

LOCAL_SAMPLE_DIR = "generated_samples"
LOCAL_PROFILE_DIR = "profiles"
LOCAL_UPLOAD_DIR = "uploads"

os.makedirs(LOCAL_SAMPLE_DIR, exist_ok=True)
os.makedirs(LOCAL_PROFILE_DIR, exist_ok=True)
os.makedirs(LOCAL_UPLOAD_DIR, exist_ok=True)

class StorageAdapter:
    def __init__(self):
        self.is_production = bool(VERCEL_BLOB_TOKEN)
        if self.is_production:
            logger.info("[Storage] Vercel Blob Storage enabled (production mode).")
        else:
            logger.info("[Storage] Local filesystem storage enabled (development mode).")

    def save_audio(
        self,
        audio_bytes: bytes,
        filename: str,
        folder: str = "generated_samples",
        content_type: str = "audio/wav"
    ) -> str:
        """Save audio and return its public/accessible URL.
        
        Args:
            audio_bytes: Binary WAV/audio data.
            filename: Target file name (e.g. 'gen_123.wav').
            folder: Target category folder ('generated_samples', 'profiles', 'uploads').
            content_type: MIME type of the audio.
            
        Returns:
            URL string (Vercel Blob CDN URL in production, or local API route in dev).
        """
        # Always save local cache copy
        local_path = os.path.join(folder, filename)
        with open(local_path, "wb") as f:
            f.write(audio_bytes)

        if not self.is_production:
            if folder == "profiles":
                return f"/api/profile_audio/{filename}"
            return f"/api/audio/{filename}"

        # Production: Upload to Vercel Blob Storage
        try:
            pathname = f"{folder}/{filename}"
            blob_url = self._upload_to_vercel_blob(pathname, audio_bytes, content_type)
            return blob_url
        except Exception as e:
            logger.error(f"[Storage] Failed to upload to Vercel Blob: {e}. Falling back to local route.")
            return f"/api/audio/{filename}"

    def _upload_to_vercel_blob(self, pathname: str, data: bytes, content_type: str) -> str:
        """Upload binary data to Vercel Blob Storage via REST API."""
        url = f"https://blob.vercel-storage.com/{pathname}"
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {VERCEL_BLOB_TOKEN}",
                "x-content-type": content_type,
                "x-add-random-suffix": "false",
            },
            method="PUT"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            return resp_data.get("url")

    def delete_blob(self, blob_url_or_filename: str, folder: str = "profiles") -> bool:
        """Delete file from storage."""
        # Clean local file
        local_path = os.path.join(folder, os.path.basename(blob_url_or_filename))
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except Exception:
                pass

        if not self.is_production or not blob_url_or_filename.startswith("http"):
            return True

        # Production Vercel Blob Delete
        try:
            req_data = json.dumps({"urls": [blob_url_or_filename]}).encode("utf-8")
            req = urllib.request.Request(
                "https://blob.vercel-storage.com/delete",
                data=req_data,
                headers={
                    "Authorization": f"Bearer {VERCEL_BLOB_TOKEN}",
                    "Content-Type": "application/json"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"[Storage] Failed to delete Vercel Blob: {e}")
            return False

# Global storage instance
storage = StorageAdapter()
