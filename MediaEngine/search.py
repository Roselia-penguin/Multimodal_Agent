"""
Multi-modal search toolset for AI Agent (simplified version).
Supports Bocha API.
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger
import requests

from .utils.config import settings

@dataclass
class WebpageResult:
    name: str
    url: str
    snippet: str
    display_url: Optional[str] = None
    date_last_crawled: Optional[str] = None

@dataclass
class ImageResult:
    name: str
    content_url: str
    host_page_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None

@dataclass
class ModalCardResult:
    card_type: str
    content: Dict[str, Any]

@dataclass
class BochaResponse:
    query: str
    conversation_id: Optional[str] = None
    answer: Optional[str] = None
    follow_ups: List[str] = field(default_factory=list)
    webpages: List[WebpageResult] = field(default_factory=list)
    images: List[ImageResult] = field(default_factory=list)
    modal_cards: List[ModalCardResult] = field(default_factory=list)

class BochaMultimodalSearch:
    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = settings.BOCHA_WEB_SEARCH_API_KEY
            if not api_key:
                raise ValueError("Bocha API Key missing.")
        self._headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
        self.base_url = settings.BOCHA_BASE_URL or "https://api.bocha.cn/v1/ai-search"

    def _parse_response(self, response_dict: Dict, query: str) -> BochaResponse:
        res = BochaResponse(query=query)
        res.conversation_id = response_dict.get('conversation_id')
        for msg in response_dict.get('messages', []):
            if msg.get('role') != 'assistant':
                continue
            ctype = msg.get('content_type')
            content = msg.get('content', '{}')
            try:
                data = json.loads(content)
            except:
                data = content
            if msg.get('type') == 'answer' and ctype == 'text':
                res.answer = data
            elif msg.get('type') == 'follow_up' and ctype == 'text':
                res.follow_ups.append(data)
            elif msg.get('type') == 'source':
                if ctype == 'webpage':
                    for item in data.get('value', []):
                        res.webpages.append(WebpageResult(
                            name=item.get('name'),
                            url=item.get('url'),
                            snippet=item.get('snippet'),
                            display_url=item.get('displayUrl'),
                            date_last_crawled=item.get('dateLastCrawled')
                        ))
                elif ctype == 'image':
                    res.images.append(ImageResult(
                        name=data.get('name'),
                        content_url=data.get('contentUrl'),
                        host_page_url=data.get('hostPageUrl'),
                        thumbnail_url=data.get('thumbnailUrl')
                    ))
                else:
                    res.modal_cards.append(ModalCardResult(card_type=ctype, content=data))
        return res

    def _search(self, query: str, **kwargs) -> BochaResponse:
        payload = {"query": query, "stream": False}
        payload.update(kwargs)
        try:
            r = requests.post(self.base_url, headers=self._headers, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            if data.get("code") != 200:
                logger.error(f"API error: {data.get('msg')}")
                return BochaResponse(query=query)
            return self._parse_response(data, query)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return BochaResponse(query=query)

    def comprehensive_search(self, query: str, max_results: int = 10) -> BochaResponse:
        return self._search(query, count=max_results, answer=True)

    def web_search_only(self, query: str, max_results: int = 15) -> BochaResponse:
        return self._search(query, count=max_results, answer=False)

    def search_for_structured_data(self, query: str) -> BochaResponse:
        return self._search(query, count=5, answer=True)

    def search_last_24_hours(self, query: str) -> BochaResponse:
        return self._search(query, freshness='oneDay', answer=True)

    def search_last_week(self, query: str) -> BochaResponse:
        return self._search(query, freshness='oneWeek', answer=True)

def load_agent_from_config():
    if not settings.BOCHA_WEB_SEARCH_API_KEY:
        raise ValueError("Bocha API Key not configured.")
    logger.info("Using BochaMultimodalSearch")
    return BochaMultimodalSearch()