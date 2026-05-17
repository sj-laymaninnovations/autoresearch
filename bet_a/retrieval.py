"""
retrieval.py — Simple BM25-ish retrieval over the distilled A+E corpus.

For Bet A's (D) test: probe whether existing trained models (5M/25M/50M)
improve when given retrieved ARM64 context at inference time — without any
retrieval-aware retraining. This is CLAUDE.md's "retrieval holds specifics"
hypothesis tested in its simplest form.

Index: all distilled Q&A pairs from Pipeline A+E (qa_records_v1.jsonl +
qa_records_e_v1.jsonl). At query time, retrieve the top-K most similar
QA pairs and prepend them as context to the prompt.

Retrieval method: TF-IDF cosine similarity over question+answer text.
No GPU, no embedding model needed. Pure sklearn.

Augmented prompt format (matches the Q:/A: training template):
    RELEVANT CONTEXT:
    Q: {retrieved_q_1}
    A: {retrieved_a_1}

    Q: {retrieved_q_2}
    A: {retrieved_a_2}

    Q: {problem}
    A:

Usage:
    from retrieval import Retriever
    r = Retriever()
    aug = r.augment_prompt("When should I use LDP/STP?", top_k=2)
    # pass aug to model instead of bare prompt
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
SOURCES = [
    ROOT / "distilled" / "qa_records_v1.jsonl",
    ROOT / "distilled" / "qa_records_e_v1.jsonl",
]

CONTEXT_HEADER = "RELEVANT CONTEXT:\n"
PROMPT_TEMPLATE = "Q: {problem}\nA: "


class Retriever:
    def __init__(self, sources: list[Path] | None = None, top_k: int = 2):
        sources = sources or SOURCES
        self.records: list[dict] = []
        for src in sources:
            if src.exists():
                for line in src.open(encoding="utf-8", errors="replace"):
                    try:
                        self.records.append(json.loads(line))
                    except Exception:
                        pass

        # Build TF-IDF index over question+answer text
        corpus = [f"{r['question']} {r['answer']}" for r in self.records]
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.matrix = self.vectorizer.fit_transform(corpus)
        print(f"[Retriever] indexed {len(self.records)} docs "
              f"({self.matrix.shape[1]} TF-IDF features)")

    def retrieve(self, query: str, top_k: int = 2,
                 exclude_question: str | None = None) -> list[dict]:
        """Return the top-k most similar records to the query."""
        qvec = self.vectorizer.transform([query])
        scores = cosine_similarity(qvec, self.matrix).flatten()
        # Exclude exact-match to the query itself (prevents retrieval of
        # the training pair that IS the question)
        if exclude_question:
            for i, r in enumerate(self.records):
                if r["question"].strip() == exclude_question.strip():
                    scores[i] = -1.0
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.records[i] for i in top_idx if scores[i] > 0]

    def augment_prompt(self, question: str, top_k: int = 2,
                        max_context_chars: int = 1200) -> str:
        """Build a retrieval-augmented prompt for the given question.

        Returns a string in the Q:/A: template with retrieved context
        prepended. The total context is capped so it fits in the model's
        seq_len with room for generation.
        """
        hits = self.retrieve(question, top_k=top_k, exclude_question=question)
        if not hits:
            return PROMPT_TEMPLATE.format(problem=question)

        ctx_parts: list[str] = []
        ctx_chars = 0
        for h in hits:
            snippet = f"Q: {h['question']}\nA: {h['answer']}"
            if ctx_chars + len(snippet) > max_context_chars:
                # Truncate the snippet to fit
                allowed = max_context_chars - ctx_chars - 20
                if allowed > 100:
                    snippet = snippet[:allowed] + "..."
                else:
                    break
            ctx_parts.append(snippet)
            ctx_chars += len(snippet)

        context_block = CONTEXT_HEADER + "\n\n".join(ctx_parts) + "\n\n"
        return context_block + PROMPT_TEMPLATE.format(problem=question)


if __name__ == "__main__":
    r = Retriever()
    questions = [
        "When optimizing ARM64 NEON code for performance, what is the difference in benefit between in-order cores like Cortex-A53 and out-of-order cores like Cortex-A72?",
        "Why is the ARM64 SVE2 SDOT instruction well-suited for 4-tap or 8-tap convolution filters in video codecs?",
    ]
    for q in questions:
        print(f"\n{'='*72}")
        print(f"QUERY: {q}")
        print(f"{'='*72}")
        hits = r.retrieve(q, top_k=2)
        for i, h in enumerate(hits):
            print(f"\n--- hit {i+1} (gran={h.get('granularity')}) ---")
            print(f"Q: {h['question'][:120]}")
            print(f"A: {h['answer'][:300]}...")
        print(f"\n--- AUGMENTED PROMPT (first 600 chars) ---")
        prompt = r.augment_prompt(q, top_k=2)
        print(prompt[:600])
