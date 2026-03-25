"""
Video Understanding Skill for Media Agent.
Extracts keyframes, transcribes audio, and generates captions for short videos.
"""

import os
import tempfile
import requests
import cv2
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from faster_whisper import WhisperModel
from loguru import logger
import torch

class VideoUnderstandingSkill:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # For faster-whisper, device can be "cuda" or "cpu"
        self.whisper_model = None
        self.blip_processor = None
        self.blip_model = None

    def _load_whisper(self):
        if self.whisper_model is None:
            logger.info("Loading faster-whisper model (base)...")
            # Compute type: int8_float16 or float16/float32; using int8 for speed
            self.whisper_model = WhisperModel("base", device=self.device, compute_type="int8_float16")

    def _load_blip(self):
        if self.blip_model is None:
            logger.info("Loading BLIP model...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

    def _download_video(self, url: str) -> str:
        try:
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            fd, path = tempfile.mkstemp(suffix=".mp4")
            with os.fdopen(fd, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return path
        except Exception as e:
            logger.error(f"Failed to download video from {url}: {e}")
            return None

    def _extract_keyframes(self, video_path: str, num_frames=5):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return []
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        cap.release()
        return frames

    def _caption_frame(self, frame):
        self._load_blip()
        image = Image.fromarray(frame)
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        out = self.blip_model.generate(**inputs, max_length=50)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption

    def _transcribe_audio(self, video_path: str):
        self._load_whisper()
        segments, info = self.whisper_model.transcribe(video_path, language="zh")
        # Join all segments into a single string
        transcript = " ".join(segment.text for segment in segments)
        return transcript

    def process(self, video_url: str) -> dict:
        video_path = self._download_video(video_url)
        if not video_path:
            return {"error": "Download failed", "transcript": "", "frame_captions": []}

        try:
            frames = self._extract_keyframes(video_path)
            captions = [self._caption_frame(frame) for frame in frames] if frames else []
            transcript = self._transcribe_audio(video_path)

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()

            return {
                "transcript": transcript,
                "frame_captions": captions,
                "num_frames": len(frames),
                "duration": duration,
                "error": None
            }
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            return {"error": str(e), "transcript": "", "frame_captions": []}
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)