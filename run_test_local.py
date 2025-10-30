#!/usr/bin/env python3
"""Local runner to execute rewrite test from within this folder.

Usage (from this directory):
  python run_test_local.py
"""
import os
import sys


def main():
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(pkg_dir, ".."))
    # Ensure the project root (parent of package) is importable
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Inline quick test to avoid dependency on removed tests package
    from comp_l0_queryrewrite.models.rewrite_model import SimpleRewriter
    from comp_l0_queryrewrite.core.rewrite import select_topk_unique
    from comp_l0_queryrewrite.utils.hyde import generate_hyde
    from comp_l0_queryrewrite.utils.embeddings import EmbeddingModel, cosine_deduplicate

    TEST_QUERY = "how to speed up python list operations?"
    MODEL_NAME = "google/flan-t5-base"
    DEVICE = "cpu"
    NUM_REWRITES = 5
    ENABLE_HYDE = True
    HYDE_K = 2
    ENABLE_DEDUPE = True
    EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DEDUP_THRESHOLD = 0.88
    DETERMINISTIC = False

    print("== comp_l0_queryrewrite: local rewrite test ==")
    print(f"Query: {TEST_QUERY}")
    print(f"Model: {MODEL_NAME} | Device: {DEVICE}")

    rewriter = SimpleRewriter(model_name=MODEL_NAME, device=DEVICE)
    raw = rewriter.rewrite(
        TEST_QUERY,
        num_return_sequences=max(NUM_REWRITES, 5),
        do_sample=not DETERMINISTIC,
    )
    rewrites = select_topk_unique(raw, k=max(NUM_REWRITES, 5))

    hyde_docs = []
    if ENABLE_HYDE:
        print("HyDE: enabled")
        hyde_docs = generate_hyde(TEST_QUERY, k=HYDE_K)
        rewrites.extend(hyde_docs)
    else:
        print("HyDE: disabled")

    final = rewrites
    if ENABLE_DEDUPE:
        print("Deduplication: cosine similarity enabled")
        try:
            emb = EmbeddingModel(model_name=EMBEDDINGS_MODEL, device=DEVICE)
            final = cosine_deduplicate(rewrites, emb, threshold=DEDUP_THRESHOLD)
        except Exception as e:
            print(f"[warn] embeddings unavailable or failed: {e}")
            final = select_topk_unique(rewrites, k=NUM_REWRITES)
    else:
        print("Deduplication: disabled")

    final = final[:NUM_REWRITES]

    print("\nRewrites:")
    for i, r in enumerate(final, 1):
        print(f"  {i}. {r}")

    if hyde_docs:
        print("\nHyDE docs:")
        for i, h in enumerate(hyde_docs, 1):
            print(f"  {i}. {h}")


if __name__ == "__main__":
    main()
