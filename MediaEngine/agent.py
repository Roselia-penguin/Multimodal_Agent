"""
Media Agent - Main agent class integrating multimodal analysis skills.
"""

import os
import sys
from typing import Dict, List, Any, Optional
from loguru import logger

from .base import LLMClient
from .search import BochaMultimodalSearch, load_agent_from_config
from .skills import VideoUnderstandingSkill, MultimodalSentimentSkill, StructuredDataSkill
from .nodes.summary_node import SummaryNode

class MediaAgent:
    """
    Media Agent responsible for analyzing videos, images, and multimodal content.
    Enhanced with three skills:
        - VideoUnderstandingSkill: extract keyframes, captions, transcripts.
        - MultimodalSentimentSkill: combine text/visual/audio sentiment.
        - StructuredDataSkill: parse modal cards and structured data.
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.llm_client = self._init_llm_client()
        self.search_client = load_agent_from_config()
        self.video_skill = VideoUnderstandingSkill()
        self.sentiment_skill = MultimodalSentimentSkill()
        self.structured_skill = StructuredDataSkill(llm_client=self.llm_client)
        self.state = {}  # holds intermediate data

    def _init_llm_client(self) -> LLMClient:
        from .utils.config import settings
        return LLMClient(
            api_key=settings.MEDIA_ENGINE_API_KEY,
            model_name=settings.MEDIA_ENGINE_MODEL_NAME,
            base_url=settings.MEDIA_ENGINE_BASE_URL
        )

    def _enhance_results(self, search_response) -> Dict:
        """
        Apply skills to enrich raw search results.
        """
        enhanced = {
            "original": search_response,
            "video_analysis": [],
            "sentiment": [],
            "structured": {}
        }

        # --- Video analysis from images that are actually videos ---
        if hasattr(search_response, 'images'):
            for img in search_response.images:
                # 检查 content_url 是否存在且不为 None
                url = getattr(img, 'content_url', None)
                if url and isinstance(url, str):
                    url_lower = url.lower()
                    if url_lower.endswith(('.mp4', '.mov', '.webm')) or 'video' in url_lower:
                        logger.info(f"Detected video URL: {url}")
                        video_result = self.video_skill.process(url)
                        if video_result and 'error' not in video_result:
                            enhanced["video_analysis"].append(video_result)
                            text_to_analyze = video_result.get('transcript', '') + ' ' + ' '.join(video_result.get('frame_captions', []))
                            if text_to_analyze.strip():
                                sentiment = self.sentiment_skill.process(text=text_to_analyze)
                                enhanced["sentiment"].append(sentiment)
                                
        # --- Structured data from modal cards ---
        if hasattr(search_response, 'modal_cards'):
            enhanced["structured"] = self.structured_skill.process([search_response])

        return enhanced

    def search(self, query: str, tool_name: str = "comprehensive_search", **kwargs) -> Dict:
        """
        Perform a search using the specified tool and apply enhancements.
        """
        tool_method = getattr(self.search_client, tool_name, None)
        if not tool_method:
            logger.error(f"Unknown search tool: {tool_name}")
            return {"error": f"Tool {tool_name} not found"}

        logger.info(f"MediaAgent searching with {tool_name}: {query}")
        response = tool_method(query, **kwargs)
        enhanced = self._enhance_results(response)
        self.state['last_search'] = enhanced
        return enhanced

    def summarize(self, title: str, content: str, search_results: Any) -> str:
        """
        Generate a summary using the enhanced data and original prompts.
        """
        # Build a combined text from all enhanced data
        enhanced = self.state.get('last_search', {})
        video_text = ""
        for v in enhanced.get('video_analysis', []):
            video_text += f"Video transcript: {v.get('transcript', '')}\n"
            video_text += f"Frame captions: {', '.join(v.get('frame_captions', []))}\n"
        sentiment_text = ""
        for s in enhanced.get('sentiment', []):
            sentiment_text += f"Sentiment: {s.get('label', 'unknown')} (confidence: {max(s.get('scores', {}).values(), default=0):.2f})\n"
        structured_text = ""
        structured = enhanced.get('structured', {})
        if structured.get('modal_cards'):
            structured_text += f"Structured data: {structured['modal_cards']}\n"

        # Augment user prompt
        extra_context = f"""
Additional multimedia analysis:
{video_text}
{sentiment_text}
{structured_text}
"""
        full_user_prompt = content + "\n\n" + extra_context

        # Use the original summary prompt (from prompts.py)
        from .prompts import SYSTEM_PROMPT_FIRST_SUMMARY
        response = self.llm_client.invoke(
            system_prompt=SYSTEM_PROMPT_FIRST_SUMMARY,
            user_prompt=full_user_prompt
        )
        return response

    def run_analysis(self, query: str) -> Dict:
        """
        High-level method: search → enhance → summarize.
        """
        search_result = self.search(query)
        summary = self.summarize("", query, search_result)
        return {
            "query": query,
            "search_result": search_result,
            "summary": summary
        }