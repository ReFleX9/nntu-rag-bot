"""Qwen3 + Sentence Transformers за стандартным интерфейсом LangChain."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from langchain_core.embeddings import Embeddings


@dataclass
class EmbeddingConfig:
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    revision: str | None = None
    max_length: int = 2048
    instruction: str = (
        "Given a question about software documentation, retrieve relevant passages "
        "that answer the question."
    )


class QwenEmbeddings(Embeddings):
    def __init__(self, config: EmbeddingConfig, device: str = "auto", batch_size: int = 8,
                 local_files_only: bool = False):
        import torch
        from sentence_transformers import SentenceTransformer

        if not 1 <= config.max_length <= 32768 or batch_size < 1:
            raise ValueError("max_length: 1..32768; batch_size: положительное число")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна в текущем PyTorch. Используйте --device cpu.")
        self.config, self.batch_size = config, batch_size
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = SentenceTransformer(
            config.model, revision=config.revision, device=device,
            local_files_only=local_files_only, trust_remote_code=False,
            model_kwargs={"torch_dtype": dtype, "attn_implementation": "sdpa"},
        )
        # Setting the tokenizer directly also works across the v5.7 rename
        # from constructor tokenizer_kwargs to processor_kwargs.
        self.model.tokenizer.padding_side = "left"
        self.model.max_seq_length = config.max_length
        # Pin an HF snapshot in the built index when the model exposes its commit.
        auto_model = getattr(self.model[0], "auto_model", None)
        commit = getattr(getattr(auto_model, "config", None), "_commit_hash", None)
        if config.revision is None and commit:
            config.revision = commit

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Never silently truncate a table or an enriched passage.
        for start in range(0, len(texts), 128):
            ids = self.model.tokenizer(
                texts[start:start+128], padding=False, truncation=False,
                add_special_tokens=True, return_attention_mask=False,
            )["input_ids"]
            for offset, tokens in enumerate(ids):
                if len(tokens) > self.config.max_length:
                    raise ValueError(
                        f"Текст {start+offset}: {len(tokens)} токенов, лимит "
                        f"{self.config.max_length}. Увеличьте --max-length и пересоздайте индекс."
                    )
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, prompt="", normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return vectors.astype("float32").tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        # Exactly one instruction for the query, none for corpus passages.
        prompt = f"Instruct: {self.config.instruction}\nQuery: {text}"
        return self._encode([prompt])[0]

    def specification(self) -> dict:
        return {"backend": "qwen3_sentence_transformers", **asdict(self.config)}
