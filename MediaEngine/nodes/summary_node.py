"""
Summary node – uses enhanced data from skills.
"""
from loguru import logger
from typing import Any

class SummaryNode:
    def __init__(self, llm_client, agent_state):
        self.llm_client = llm_client
        self.state = agent_state

    def execute(self, title: str, content: str, search_results: Any) -> str:
        # Retrieve enhanced data from agent state (if any)
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

        extra = f"""
Additional multimedia analysis:
{video_text}
{sentiment_text}
{structured_text}
"""
        full_user_prompt = content + "\n\n" + extra
        from ..prompts import SYSTEM_PROMPT_FIRST_SUMMARY
        response = self.llm_client.invoke(SYSTEM_PROMPT_FIRST_SUMMARY, full_user_prompt)
        return response