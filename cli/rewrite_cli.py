#!/usr/bin/env python3
import json
from typing import Optional

import click

from comp_l0_queryrewrite.core.rewrite import RewriteResult, select_topk_unique
from comp_l0_queryrewrite.models.rewrite_model import SimpleRewriter
from comp_l0_queryrewrite.utils.embeddings import EmbeddingModel, cosine_deduplicate
from comp_l0_queryrewrite.utils.hyde import generate_hyde


@click.command()
@click.option("--query", "query", type=str, required=True, help="Input user query")
@click.option("--model", type=str, default="t5-small", show_default=True, help="HF model name")
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option("--num", type=int, default=5, show_default=True, help="Number of rewrites")
@click.option("--hyde/--no-hyde", default=False, show_default=True, help="Enable HyDE generation")
@click.option("--hyde-k", type=int, default=2, show_default=True, help="HyDE doc count")
@click.option("--dedupe/--no-dedupe", default=True, show_default=True, help="Cosine similarity dedupe")
@click.option("--embeddings", type=str, default="sentence-transformers/all-MiniLM-L6-v2", show_default=True)
@click.option("--threshold", type=float, default=0.88, show_default=True, help="Dedupe cosine threshold")
@click.option("--json-out/--text-out", default=True, show_default=True, help="Output JSON or plain text")
@click.option("--no-sample", is_flag=True, default=False, help="Disable sampling for deterministic output")
def main(query: str, model: str, device: str, num: int, hyde: bool, hyde_k: int,
         dedupe: bool, embeddings: str, threshold: float, json_out: bool, no_sample: bool):
    """Run query rewrite pipeline from terminal."""
    # Rewrite candidates
    rewriter = SimpleRewriter(model_name=model, device=device)
    # generate a few more than requested to allow dedupe to trim
    raw = rewriter.rewrite(query, num_return_sequences=max(num, 5), do_sample=not no_sample)
    rewrites = select_topk_unique(raw, k=max(num, 5))

    hyde_docs = []
    if hyde:
        hyde_docs = generate_hyde(query, k=hyde_k)
        rewrites.extend(hyde_docs)

    final = rewrites
    if dedupe:
        try:
            emb = EmbeddingModel(model_name=embeddings, device=device)
            final = cosine_deduplicate(rewrites, emb, threshold=threshold)
        except Exception:
            # If embeddings are not available, keep unique-by-text only
            final = select_topk_unique(rewrites, k=num)

    # trim to requested count prioritizing rewrites then hyde
    final = final[:num]

    result = RewriteResult(rewrites=final, hyde=hyde_docs if hyde else [])

    if json_out:
        click.echo(json.dumps({
            "query": query,
            "rewrites": result.rewrites,
            "hyde": result.hyde,
        }, ensure_ascii=False))
    else:
        click.echo(f"Query: {query}")
        click.echo("Rewrites:")
        for i, r in enumerate(result.rewrites, 1):
            click.echo(f"  {i}. {r}")
        if result.hyde:
            click.echo("HyDE:")
            for i, h in enumerate(result.hyde, 1):
                click.echo(f"  {i}. {h}")


if __name__ == "__main__":
    main()
