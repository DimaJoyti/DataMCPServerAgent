"""
OpenAI embedder implementation.
"""

import logging
import time
from typing import List, Optional

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from .base_embedder import BaseEmbedder, EmbeddingConfig, EmbeddingResult

class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedder using OpenAI's embedding models."""

    # Available OpenAI embedding models
    AVAILABLE_MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, config: EmbeddingConfig, api_key: Optional[str] = None):
        """
        Initialize OpenAI embedder.

        Args:
            config: Embedding configuration
            api_key: OpenAI API key (if not set in environment)
        """
        if not HAS_OPENAI:
            raise ImportError(
                "OpenAI embedder requires openai package. "
                "Install with: pip install openai"
            )

        super().__init__(config)

        # Initialize OpenAI client
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = openai.OpenAI()  # Uses OPENAI_API_KEY env var

        # Validate model
        if config.model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {config.model_name} not supported. "
                f"Available models: {list(self.AVAILABLE_MODELS.keys())}"
            )

        # Set embedding dimension if not specified
        if not config.embedding_dimension:
            config.embedding_dimension = self.AVAILABLE_MODELS[config.model_name]

    def embed_text(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text using OpenAI.

        Args:
            text: Input text

        Returns:
            EmbeddingResult: Embedding result
        """
        text = self._preprocess_text(text)
        start_time = time.time()

        def _embed():
            response = self.client.embeddings.create(
                model=self.config.model_name,
                input=text,
                **self.config.custom_options
            )
            return response

        # Execute with retry
        response = self._retry_with_backoff(_embed)

        # Extract embedding
        embedding_data = response.data[0]
        embedding = embedding_data.embedding

        # Post-process embedding
        embedding = self._post_process_embedding(embedding, text)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Get token count from usage if available
        token_count = getattr(response, 'usage', {}).get('total_tokens')

        return self._create_embedding_result(
            text=text,
            embedding=embedding,
            processing_time=processing_time,
            token_count=token_count
        )

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Generate embeddings for a batch of texts using OpenAI.

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
        batch_size = min(self.config.batch_size, 2048)  # OpenAI limit

        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i:i + batch_size]
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
            response = self.client.embeddings.create(
                model=self.config.model_name,
                input=texts,
                **self.config.custom_options
            )
            return response

        # Execute with retry
        response = self._retry_with_backoff(_embed)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Get token count from usage if available
        total_tokens = getattr(response, 'usage', {}).get('total_tokens', 0)
        avg_tokens_per_text = total_tokens // len(texts) if total_tokens else None

        # Create results
        results = []
        for i, embedding_data in enumerate(response.data):
            text = texts[i]
            embedding = embedding_data.embedding

            # Post-process embedding
            embedding = self._post_process_embedding(embedding, text)

            result = self._create_embedding_result(
                text=text,
                embedding=embedding,
                processing_time=processing_time / len(texts),  # Distribute time
                token_count=avg_tokens_per_text
            )
            results.append(result)

        return results

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.

        Returns:
            int: Embedding dimension
        """
        return self.AVAILABLE_MODELS[self.config.model_name]

    def get_max_tokens(self) -> int:
        """
        Get maximum tokens for the model.

        Returns:
            int: Maximum tokens
        """
        # OpenAI embedding models typically support 8192 tokens
        return 8192

    def estimate_cost(self, texts: List[str]) -> float:
        """
        Estimate cost for embedding texts.

        Args:
            texts: List of texts to embed

        Returns:
            float: Estimated cost in USD
        """
        # Rough token estimation (4 characters per token)
        total_chars = sum(len(text) for text in texts)
        estimated_tokens = total_chars // 4

        # Pricing per 1M tokens (as of 2024)
        pricing = {
            "text-embedding-3-small": 0.02,
            "text-embedding-3-large": 0.13,
            "text-embedding-ada-002": 0.10,
        }

        price_per_million = pricing.get(self.config.model_name, 0.10)
        estimated_cost = (estimated_tokens / 1_000_000) * price_per_million

        return estimated_cost

    def health_check(self) -> bool:
        """
        Perform health check on OpenAI embedder.

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
            self.logger.error(f"OpenAI health check failed: {e}")
            return False
