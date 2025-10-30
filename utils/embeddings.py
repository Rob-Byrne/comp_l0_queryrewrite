from typing import List

try:
    from sentence_transformers import SentenceTransformer
    import torch
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    torch = None  # type: ignore


class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers not available. Install optional dependency."
            )
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str]):
        return self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

def cosine_deduplicate(texts: List[str], model: EmbeddingModel, threshold: float = 0.88) -> List[str]:
    if not texts:
        return []
    embs = model.encode(texts)
    keep = []
    kept_embs = []
    for i, t in enumerate(texts):
        e = embs[i]
        is_dup = False
        for ke in kept_embs:
            sim = float((e * ke).sum())
            if sim >= threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(t)
            kept_embs.append(e)
    return keep

