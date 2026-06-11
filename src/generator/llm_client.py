"""
LLM 生成模块（骨架）
实现具体的 LLM API 调用
"""
import sys
from pathlib import Path
from typing import List, Dict, AsyncGenerator

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import llm_config


class LLMClient:
    """LLM 客户端封装（支持多种后端）"""

    def __init__(self):
        self.provider = llm_config.PROVIDER
        self.api_key = llm_config.API_KEY
        self.base_url = llm_config.BASE_URL
        self.model = llm_config.MODEL
        self.temperature = llm_config.TEMPERATURE
        self.max_tokens = llm_config.MAX_TOKENS

        self._client = None

    def _get_client(self):
        """获取 LLM 客户端"""
        if self._client is None:
            if not self.api_key:
                raise ValueError("请配置 LLM_API_KEY")

            # 使用 OpenAI 兼容接口（DeepSeek、Qwen、通义千问 等都兼容）
            from openai import OpenAI, AsyncOpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

        return self._client

    def generate(self, system_prompt: str, user_message: str) -> str:
        """同步生成"""
        client = self._get_client()

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content

    async def generate_stream(self, system_prompt: str,
                               user_message: str) -> AsyncGenerator[str, None]:
        """流式生成"""
        client = self._get_client()

        stream = await self._async_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def is_available(self) -> bool:
        """检查 LLM 是否可用"""
        return bool(self.api_key)


# 全局单例
_llm_client = None


def get_llm_client() -> LLMClient:
    """获取 LLM 客户端单例"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
