"""
HuggingFace embedder implementation.
"""

import time
from typing import List, Optional

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import torch
    from transformers import AutoModel, AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from .base_embedder import BaseEmbedder, EmbeddingConfig, EmbeddingResult


class HuggingFaceEmbedder(BaseEmbedder):
    """HuggingFace embedder using sentence-transformers or transformers."""

    # Popular sentence transformer models
    POPULAR_MODELS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "all-MiniLM-L12-v2": 384,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "distilbert-base-nli-stsb-mean-tokens": 768,
        "roberta-large-nli-stsb-mean-tokens": 1024,
    }

    def __init__(
        self,
        config: EmbeddingConfig,
        device: Optional[str] = None,
        use_sentence_transformers: bool = True,
    ):
        """
        Initialize HuggingFace embedder.

        Args:
            config: Embedding configuration
            device: Device to use ('cpu', 'cuda', 'mps', etc.)
            use_sentence_transformers: Whether to use sentence-transformers library
        """
        super().__init__(config)

        self.use_sentence_transformers = use_sentence_transformers
        self.device = device or self._get_best_device()

        if use_sentence_transformers:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "sentence-transformers required for HuggingFace embedder. "
                    "Install with: pip install sentence-transformers"
                )
            self._init_sentence_transformer()
        else:
            if not HAS_TRANSFORMERS:
                raise ImportError(
                    "transformers required for HuggingFace embedder. "
                    "Install with: pip install transformers torch"
                )
            self._init_transformer()

    def _get_best_device(self) -> str:
        """Get the best available device."""
        if HAS_TRANSFORMERS:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        return "cpu"

    def _init_sentence_transformer(self):
        """Initialize sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.config.model_name, device=self.device)

            # Get embedding dimension
            if not self.config.embedding_dimension:
                self.config.embedding_dimension = self.model.get_sentence_embedding_dimension()

            self.logger.info(
                f"Loaded sentence transformer {self.config.model_name} "
                f"on {self.device} with dimension {self.config.embedding_dimension}"
            )

        except Exception as e:
            self.logger.error(f"Failed to load sentence transformer: {e}")
            raise

    def _init_transformer(self):
        """Initialize transformer model and tokenizer."""
        try:

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModel.from_pretrained(self.config.model_name)
            self.model.to(self.device)
            self.model.eval()

            # Estimate embedding dimension (usually hidden_size)
            if not self.config.embedding_dimension:
                self.config.embedding_dimension = self.model.config.hidden_size

            self.logger.info(
                f"Loaded transformer {self.config.model_name} "
                f"on {self.device} with dimension {self.config.embedding_dimension}"
            )

        except Exception as e:
            self.logger.error(f"Failed to load transformer: {e}")
            raise

    def embed_text(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text using HuggingFace.

        Args:
            text: Input text

        Returns:
            EmbeddingResult: Embedding result
        """
        text = self._preprocess_text(text)
        start_time = time.time()

        if self.use_sentence_transformers:
            embedding = self._embed_with_sentence_transformer([text])[0]
        else:
            embedding = self._embed_with_transformer([text])[0]

        # Post-process embedding
        embedding = self._post_process_embedding(embedding, text)

        # Calculate processing time
        processing_time = time.time() - start_time

        return self._create_embedding_result(
            text=text, embedding=embedding, processing_time=processing_time
        )

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Generate embeddings for a batch of texts using HuggingFace.

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
        batch_size = self.config.batch_size

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

        if self.use_sentence_transformers:
            embeddings = self._embed_with_sentence_transformer(texts)
        else:
            embeddings = self._embed_with_transformer(texts)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Create results
        results = []
        for i, embedding in enumerate(embeddings):
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

    def _embed_with_sentence_transformer(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using sentence-transformers.

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of embedding vectors
        """
        embeddings = self.model.encode(
            texts, batch_size=self.config.batch_size, show_progress_bar=False, convert_to_numpy=True
        )

        # Convert to list of lists
        return [embedding.tolist() for embedding in embeddings]

    def _embed_with_transformer(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using transformers.

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of embedding vectors
        """
        import torch

        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_input_length,
            return_tensors="pt",
        )

        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)

            # Use mean pooling of last hidden states
            embeddings = self._mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])

        # Convert to list
        embeddings = embeddings.cpu().numpy()
        return [embedding.tolist() for embedding in embeddings]

    def _mean_pooling(self, token_embeddings, attention_mask):
        """
        Perform mean pooling on token embeddings.

        Args:
            token_embeddings: Token embeddings from transformer
            attention_mask: Attention mask

        Returns:
            Pooled embeddings
        """
        import torch

        # Expand attention mask to match token embeddings dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum embeddings and divide by number of tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.

        Returns:
            int: Embedding dimension
        """
        return self.config.embedding_dimension

    def health_check(self) -> bool:
        """
        Perform health check on HuggingFace embedder.

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
            self.logger.error(f"HuggingFace health check failed: {e}")
            return False
