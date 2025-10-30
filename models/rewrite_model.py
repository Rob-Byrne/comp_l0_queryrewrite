from typing import List

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore
    AutoModelForSeq2SeqLM = None  # type: ignore


class SimpleRewriter:
    """
    Lightweight wrapper around a small seq2seq model (T5/BART).
    Defaults to a small T5 model to keep local requirements minimal.
    """

    def __init__(self, model_name: str = "t5-small", device: str = "cpu"):
        if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            raise RuntimeError(
                "transformers not available. Please install from requirements.txt"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(device)
        self.device = device

    def rewrite(
        self,
        query: str,
        num_return_sequences: int = 3,
        max_new_tokens: int = 32,
        do_sample: bool = True,
    ) -> List[str]:
        # Use a clearer instruction format; T5 often works fine without special prefix.
        prompt = f"paraphrase this: {query}"
        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            do_sample=do_sample,
            top_p=0.92 if do_sample else None,
            top_k=50 if do_sample else None,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=3,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
