from typing import Optional, List, Dict
from .base import LLMClient

try:
    import openai
except Exception:
    openai = None


class OpenAIClient(LLMClient):
    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        if openai is None:
            raise RuntimeError("openai package is required. Install with: pip install openai")
        self.model = model
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)  # type: ignore[attr-defined]

    def generate(self, prompt: str) -> str:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "你是一个文本纠错与润色助手。"},
            {"role": "user", "content": prompt},
        ]
        rsp = self.client.chat.completions.create(model=self.model, messages=messages)
        content = rsp.choices[0].message.content if rsp and rsp.choices else ''
        return content or '' 