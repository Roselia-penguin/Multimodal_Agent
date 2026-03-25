"""
Multimodal Sentiment Analysis Skill (text + visual only).
Audio part temporarily disabled to avoid dependency issues.
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoFeatureExtractor
from loguru import logger

class MultimodalSentimentSkill:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.text_tokenizer = None
        self.text_model = None
        self.visual_feature_extractor = None
        self.visual_model = None

    def _load_text_model(self):
        if self.text_model is None:
            logger.info("Loading text sentiment model...")
            model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
            self.text_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

    def _load_visual_model(self):
        if self.visual_model is None:
            logger.info("Loading visual sentiment model...")
            model_name = "trpakov/vit-face-expression"
            self.visual_feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.visual_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

    def text_sentiment(self, text: str) -> dict:
        self._load_text_model()
        inputs = self.text_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        neg_score = scores[0] + scores[1]
        neu_score = scores[2]
        pos_score = scores[3] + scores[4]
        label = "positive" if pos_score > max(neg_score, neu_score) else "negative" if neg_score > max(pos_score, neu_score) else "neutral"
        return {"label": label, "scores": {"positive": float(pos_score), "neutral": float(neu_score), "negative": float(neg_score)}}

    def visual_sentiment(self, image) -> dict:
        self._load_visual_model()
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        inputs = self.visual_feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.visual_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        # Example mapping for emotion classes: 0=angry,1=disgust,2=fear,3=happy,4=neutral,5=sad,6=surprise
        pos_emotions = [3, 6]
        neg_emotions = [0, 1, 2, 5]
        pos_score = sum(probs[i] for i in pos_emotions if i < len(probs))
        neg_score = sum(probs[i] for i in neg_emotions if i < len(probs))
        neu_score = probs[4] if len(probs) > 4 else 0
        label = "positive" if pos_score > max(neg_score, neu_score) else "negative" if neg_score > max(pos_score, neu_score) else "neutral"
        return {"label": label, "scores": {"positive": float(pos_score), "neutral": float(neu_score), "negative": float(neg_score)}}

    def process(self, text: str = None, image=None, audio_path: str = None) -> dict:
        results = {}
        if text:
            results['text'] = self.text_sentiment(text)
        if image is not None:
            results['visual'] = self.visual_sentiment(image)

        if len(results) == 1:
            return list(results.values())[0]
        else:
            combined = {"positive": 0, "neutral": 0, "negative": 0}
            for mod in results.values():
                for k in combined:
                    combined[k] += mod["scores"][k]
            total = sum(combined.values())
            if total > 0:
                combined = {k: v/total for k, v in combined.items()}
            label = max(combined, key=combined.get)
            return {"label": label, "scores": combined}