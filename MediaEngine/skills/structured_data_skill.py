"""
Structured Data Extraction Skill.
Parses modal cards, tables, and key-value pairs from search results.
"""

import json
import pandas as pd
from bs4 import BeautifulSoup
import requests
from loguru import logger
from typing import List, Dict, Any

class StructuredDataSkill:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def parse_modal_cards(self, modal_cards: List[Any]) -> List[Dict]:
        parsed = []
        for card in modal_cards:
            parsed.append({
                "type": card.card_type,
                "data": card.content
            })
        return parsed

    def extract_tables_from_html(self, html: str) -> List[Dict]:
        try:
            tables = pd.read_html(html)
            structured = [df.to_dict(orient="records") for df in tables]
            return structured
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
            return []

    def extract_key_values(self, text: str, query_topic: str) -> Dict:
        if not self.llm_client:
            return {}
        prompt = f"""
Extract key-value pairs from the following text that are relevant to the topic "{query_topic}". Return only a JSON object with the extracted pairs.

Text: {text[:2000]}
"""
        try:
            response = self.llm_client.invoke(system_prompt="You are a data extraction assistant.", user_prompt=prompt)
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}
        except Exception as e:
            logger.error(f"Key-value extraction failed: {e}")
            return {}

    def process(self, search_results: List[Any]) -> Dict:
        structured_output = {
            "modal_cards": [],
            "tables": [],
            "key_values": []
        }
        for result in search_results:
            if hasattr(result, 'modal_cards') and result.modal_cards:
                structured_output["modal_cards"].extend(self.parse_modal_cards(result.modal_cards))
            # Optionally fetch webpages for tables (simplified)
        return structured_output