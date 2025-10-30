# comp_l0_queryrewrite

Lightweight query understanding and rewriting module for local pipelines. Provides:
- Small T5/BART rewrite model wrapper
- Optional HyDE-style pseudo docs
- Cosine-similarity deduplication (sentence-transformers)
- Click-based CLI, simple local runners

## Quickstart

1) Install deps

```
cd comp_l0_queryrewrite
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Run the CLI (from this folder)

```
./run_cli_local.py --query "how to speed up python list operations?" --num 5 --no-sample
```

3) Try the local test runner

```
./run_test_local.py
```

## CLI Options

- `--query` Input user query (required)
- `--model` HF model name (default: `t5-small`)
- `--device` Compute device (default: `cpu`)
- `--num` Number of rewrites (default: 5)
- `--hyde/--no-hyde` Enable HyDE pseudo docs (default: off)
- `--hyde-k` HyDE doc count (default: 2)
- `--dedupe/--no-dedupe` Cosine similarity dedup (default: on)
- `--embeddings` Embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `--threshold` Dedup cosine threshold (default: 0.88)
- `--json-out/--text-out` Output format (default: JSON)
- `--no-sample` Disable sampling for deterministic output

Examples:
```
./run_cli_local.py --query "best way to clean a cast iron pan" --num 5 --no-sample
./cli/rewrite_cli.py --query "install node on ubuntu" --hyde --num 6
```

## Notes
- `sentence-transformers` is optional; HyDE and dedupe gracefully fallback if unavailable.
- For Apple Silicon or constrained environments, consider installing `torch` via Conda.
- Outputs are lightly cleaned to remove prompt artifacts (e.g., stray "paraphrase:" prefixes).

## Project Layout

- `cli/rewrite_cli.py` CLI command (Click)
- `core/rewrite.py` Cleanup and result structures
- `models/rewrite_model.py` T5/BART wrapper
- `utils/embeddings.py` Embeddings + cosine dedupe
- `utils/hyde.py` HyDE pseudo doc generator
- `run_cli_local.py` Local CLI runner
- `run_test_local.py` Local quick test
- `requirements.txt` Dependencies
- `config.yaml` Example configuration

