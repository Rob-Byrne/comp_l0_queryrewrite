from typing import List

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


def generate_hyde(query: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", k: int = 2) -> List[str]:
    """
    Simple HyDE-like generation using sentence-transformers prompt-style generation
    fallback: returns templated pseudo-docs if model not available.
    """
    if SentenceTransformer is None:
        # minimal fallback that keeps things runnable without optional deps
        return [
            f"A detailed passage about: {query}",
            f"Background and context: {query}",
        ][:k]

    # Not truly generative; produce pseudo docs with query expansions.
    # This keeps dependencies light while offering helpful context probes.
    # Using embeddings model name solely for consistency of configuration.
    docs = [
        f"Comprehensive explanation and related facts about {query}.",
        f"Concise overview, definitions, and key terms for {query}.",
        f"Real-world examples and scenarios concerning {query}.",
    ]
    return docs[:k]

