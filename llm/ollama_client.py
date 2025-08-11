from typing import Optional
import requests
from .base import LLMClient


class OllamaClient(LLMClient):
    def __init__(self, model: str, base_url: Optional[str] = None):
        self.model = model
        self.base_url = (base_url or 'http://localhost:11434').rstrip('/')

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                # Sensible defaults; can be made configurable later
                "temperature": 0.2,
            },
        }
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data.get('response', '') 