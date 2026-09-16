"""Hugging Face hosted open-model adapter for LangChain pipelines."""

from __future__ import annotations

import os

from huggingface_hub import InferenceClient
from langchain_core.runnables import RunnableLambda

DEFAULT_MODEL = "openai/gpt-oss-20b"


def _to_hf_messages(prompt_value) -> list[dict[str, str]]:
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    messages = []
    for message in prompt_value.to_messages():
        content = message.content if isinstance(message.content, str) else str(message.content)
        messages.append({"role": role_map.get(message.type, "user"), "content": content})
    return messages


def get_llm(temperature: float = 0.2):
    """Create a LangChain Runnable backed by an open-weight HF model."""
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACEHUB_API_TOKEN is not set in .env")

    client = InferenceClient(api_key=token, provider="auto", timeout=120)
    model = os.getenv("HF_MODEL", DEFAULT_MODEL)
    max_tokens = int(os.getenv("HF_MAX_TOKENS", "1600"))

    def generate(prompt_value) -> str:
        response = client.chat_completion(
            model=model,
            messages=_to_hf_messages(prompt_value),
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"reasoning_effort": "low"},
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError(
                "The Hugging Face model used its full response budget before answering. "
                "Set HF_MAX_TOKENS=2400 in .env and retry."
            )
        return content.strip()

    return RunnableLambda(generate)
