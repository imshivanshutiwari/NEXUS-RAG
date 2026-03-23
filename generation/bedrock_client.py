"""AWS Bedrock Claude-3-Sonnet client."""
import json
import os
from typing import Iterator

from utils.logger import get_logger

logger = get_logger(__name__)

_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
_REGION = "us-east-1"


class BedrockClient:
    """Call AWS Bedrock Claude-3-Sonnet for text generation."""

    def __init__(self) -> None:
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        try:
            import boto3

            self._client = boto3.client(
                "bedrock-runtime",
                region_name=os.getenv("AWS_DEFAULT_REGION", _REGION),
            )
            logger.info("BedrockClient: boto3 client initialised.")
        except Exception as exc:
            logger.warning("BedrockClient: boto3 init failed: %s", exc)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        """Generate a response from Claude-3-Sonnet."""
        if self._client is None:
            raise RuntimeError("Bedrock client not initialised. Check AWS credentials.")
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        try:
            response = self._client.invoke_model(
                modelId=os.getenv("BEDROCK_MODEL_ID", _MODEL_ID),
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            return result["content"][0]["text"]
        except Exception as exc:
            logger.error("BedrockClient.generate failed: %s", exc)
            raise

    def stream_generate(self, prompt: str, max_tokens: int = 2048) -> Iterator[str]:
        """Streaming generation via invoke_model_with_response_stream."""
        if self._client is None:
            raise RuntimeError("Bedrock client not initialised.")
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        try:
            response = self._client.invoke_model_with_response_stream(
                modelId=os.getenv("BEDROCK_MODEL_ID", _MODEL_ID),
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            for event in response["body"]:
                chunk = json.loads(event["chunk"]["bytes"])
                if chunk.get("type") == "content_block_delta":
                    delta = chunk.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta.get("text", "")
        except Exception as exc:
            logger.error("BedrockClient.stream_generate failed: %s", exc)
            raise
