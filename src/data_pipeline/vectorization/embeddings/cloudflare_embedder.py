"""
Cloudflare AI embedder implementation.
"""

import time
from typing import List, Optional

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from .base_embedder import BaseEmbedder, EmbeddingConfig, EmbeddingResult


class CloudflareEmbedder(BaseEmbedder):
    """Cloudflare AI embedder using Cloudflare's embedding models."""

    # Available Cloudflare embedding models
    AVAILABLE_MODELS = {
        "@cf/baai/bge-base-en-v1.5": 768,
        "@cf/baai/bge-small-en-v1.5": 384,
        "@cf/baai/bge-large-en-v1.5": 1024,
    }

    def __init__(
        self,
        config: EmbeddingConfig,
        account_id: str,
        api_token: str,
        base_url: Optional[str] = None,
    ):
        """
        Initialize Cloudflare embedder.

        Args:
            config: Embedding configuration
            account_id: Cloudflare account ID
            api_token: Cloudflare API token
            base_url: Base URL for Cloudflare AI API
        """
        if not HAS_HTTPX:
            raise ImportError(
                "Cloudflare embedder requires httpx package. " "Install with: pip install httpx"
            )

        super().__init__(config)

        self.account_id = account_id
        self.api_token = api_token
        self.base_url = base_url or "https://api.cloudflare.com/client/v4"

        # Validate model
        if config.model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {config.model_name} not supported. "
                f"Available models: {list(self.AVAILABLE_MODELS.keys())}"
            )

        # Set embedding dimension if not specified
        if not config.embedding_dimension:
            config.embedding_dimension = self.AVAILABLE_MODELS[config.model_name]

        # Initialize HTTP client
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    def embed_text(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text using Cloudflare AI.

        Args:
            text: Input text

        Returns:
            EmbeddingResult: Embedding result
        """
        text = self._preprocess_text(text)
        start_time = time.time()

        def _embed():
            return self._call_cloudflare_api([text])

        # Execute with retry
        response = self._retry_with_backoff(_embed)

        # Extract embedding
        embedding = response["result"]["data"][0]

        # Post-process embedding
        embedding = self._post_process_embedding(embedding, text)

        # Calculate processing time
        processing_time = time.time() - start_time

        return self._create_embedding_result(
            text=text, embedding=embedding, processing_time=processing_time
        )

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Generate embeddings for a batch of texts using Cloudflare AI.

        Args:
            texts: List of input texts

        Returns:
            List[EmbeddingResult]: List of embedding results
        """
        if not texts:
            return []

        # Preprocess all texts
        processed_texts = [self._preprocess_text(text) for text in texts]

        # Process in batches
        results = []
        batch_size = min(self.config.batch_size, 100)  # Cloudflare limit

        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i : i + batch_size]
            batch_results = self._embed_batch_chunk(batch_texts)
            results.extend(batch_results)

        return results

    def _embed_batch_chunk(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Embed a single batch chunk.

        Args:
            texts: List of texts to embed

        Returns:
            List[EmbeddingResult]: Embedding results
        """
        start_time = time.time()

        def _embed():
            return self._call_cloudflare_api(texts)

        # Execute with retry
        response = self._retry_with_backoff(_embed)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Create results
        results = []
        embeddings_data = response["result"]["data"]

        for i, embedding in enumerate(embeddings_data):
            text = texts[i]

            # Post-process embedding
            embedding = self._post_process_embedding(embedding, text)

            result = self._create_embedding_result(
                text=text,
                embedding=embedding,
                processing_time=processing_time / len(texts),  # Distribute time
            )
            results.append(result)

        return results

    def _call_cloudflare_api(self, texts: List[str]) -> dict:
        """
        Call Cloudflare AI API for embeddings.

        Args:
            texts: List of texts to embed

        Returns:
            dict: API response
        """
        url = f"{self.base_url}/accounts/{self.account_id}/ai/run/{self.config.model_name}"

        payload = {"text": texts}

        # Add custom options
        payload.update(self.config.custom_options)

        response = self.client.post(url, json=payload)

        if response.status_code != 200:
            error_msg = f"Cloudflare API error: {response.status_code} - {response.text}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

        return response.json()

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.

        Returns:
            int: Embedding dimension
        """
        return self.AVAILABLE_MODELS[self.config.model_name]

    def health_check(self) -> bool:
        """
        Perform health check on Cloudflare embedder.

        Returns:
            bool: True if healthy
        """
        try:
            # Try to embed a simple test text
            test_text = "Health check test."
            result = self.embed_text(test_text)

            # Verify result
            expected_dim = self.get_embedding_dimension()
            if len(result.embedding) != expected_dim:
                self.logger.error(
                    f"Unexpected embedding dimension: {len(result.embedding)} "
                    f"(expected {expected_dim})"
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Cloudflare health check failed: {e}")
            return False

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "client"):
            self.client.close()
