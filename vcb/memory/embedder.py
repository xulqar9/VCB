"""CPU sentence embeddings via fastembed (bge-small-en, 384-d).

Kept off the GPU on purpose: the LLM and TTS own the VRAM, and short-text
embedding is fast enough on CPU that it never gates the voice loop.
"""

import numpy as np


class Embedder:
    def __init__(self, model_name: str, query_instruction: str = "") -> None:
        self._model_name = model_name
        self._query_instruction = query_instruction
        self._model = None  # lazy: fastembed fetches weights on first load()

    def load(self) -> None:
        if self._model is not None:
            return
        from fastembed import TextEmbedding

        self._model = TextEmbedding(
            model_name=self._model_name,
            providers=["CPUExecutionProvider"],
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized row vectors so cosine similarity is a dot product."""
        self.load()
        vecs = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def embed_query(self, text: str) -> np.ndarray:
        """Encode a search query; bge separates queries from passages better with
        its instruction prefix."""
        return self.embed([self._query_instruction + text])[0]
