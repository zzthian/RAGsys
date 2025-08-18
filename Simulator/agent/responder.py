import json
from config.config import *
from openai import OpenAI
import httpx


class Responder:
    def __init__(self, prompt, rag_system):
        self.prompt = prompt
        self.client = OpenAI(
            base_url=API_BASE,
            api_key=API_KEY,
            http_client=httpx.Client(base_url=API_BASE, follow_redirects=True),
        )

    def generate(self):
        result = (
            self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": self.prompt}],
                temperature=TEMPERATURE,
                n=1,
            )
            .choices[0]
            .message.content.strip()
        )

        print("Prompt to RAG: \n" + self.prompt)
        print("=======================")
        print("Response from RAG: \n" + result)
        return result
