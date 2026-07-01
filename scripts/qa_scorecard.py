from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter
from pathlib import Path

DEFAULT_CASES: list[dict[str, object]] = [
    {
        "question": "Who was granted Bear Island after House Stark won it?",
        "expected_any": ["house mormont", "mormont"],
    },
    {
        "question": "Which house was founded by Karlon Stark?",
        "expected_any": ["house karstark", "karstark"],
    },
    {
        "question": "What city is the seat of House Hightower?",
        "expected_any": ["oldtown"],
    },
    {
        "question": "What does the text say happens after the Faceless Men accept your offer?",
        "expected_any": ["the man you named is dead", "is dead"],
    },
    {
        "question": "What do the servants in the House of Black and White preach?",
        "expected_any": ["preach no sermons", "no sermons"],
    },
    {
        "question": "Who did Robert defeat in battle according to the text?",
        "expected_any": ["battle of ashford", "ashford"],
    },
    {
        "question": "What is reverse-mode automatic differentiation used for in the notes?",
        "expected_any": ["computing the gradient", "compute the gradient", "gradient"],
    },
    {
        "question": "Which optimization method is used throughout the course?",
        "expected_any": ["stochastic gradient descent", "gradient descent"],
    },
    {
        "question": "What does the notes say about variational autoencoder?",
        "expected_any": ["variational autoencoder", "vae"],
    },
    {
        "question": "What controls the strength of regularization in the latent variable model?",
        "expected_any": ["sigma", "σ2", "prior variance", "regularization strength"],
    },
    {
        "question": "What does the document say about quantum entanglement?",
        "expected_status": "insufficient_evidence",
    },
    {
        "question": "Who won the soccer world cup in 2018?",
        "expected_status": "insufficient_evidence",
    },
]


def post_query(url: str, q: str, top_k: int) -> dict:
    payload = json.dumps({"query": q, "top_k": top_k}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_cases(path: Path | None) -> list[dict[str, object]]:
    if path is None:
        return DEFAULT_CASES
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Cases file must be a JSON array.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QA scorecard against /query endpoint.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/query", help="Query endpoint URL")
    parser.add_argument("--top-k", type=int, default=3, help="top_k to send per query")
    parser.add_argument("--cases", type=Path, default=None, help="Optional path to JSON file of cases")
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save full JSON results")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    rows: list[dict[str, object]] = []

    for case in cases:
        question = str(case["question"])
        expected_status = str(case.get("expected_status", "retrieval_complete"))
        expected_any = [str(s).lower() for s in case.get("expected_any", [])]

        try:
            data = post_query(args.url, question, args.top_k)
        except Exception as exc:
            rows.append(
                {
                    "question": question,
                    "status": f"ERROR: {exc}",
                    "status_ok": False,
                    "evidence_ok": False,
                    "answer_ok": False,
                    "citations_ok": False,
                }
            )
            continue

        status = data.get("status")
        diag = data.get("diagnostics") or {}
        chunks = data.get("retrieved_chunks") or []
        generated_answer = str(data.get("generated_answer") or "")
        generated_lower = generated_answer.lower()
        cited_ids = data.get("cited_chunk_ids") or []
        retrieved_ids = {c.get("chunk_id") for c in chunks if c.get("chunk_id")}
        invalid_citations = [cid for cid in cited_ids if cid not in retrieved_ids]

        chunk_text = " ".join((c.get("text") or "").lower() for c in chunks)
        evidence_ok = True if not expected_any else any(tok in chunk_text for tok in expected_any)

        if status == "retrieval_complete":
            answer_ok = True if not expected_any else (
                bool(generated_answer) and any(tok in generated_lower for tok in expected_any)
            )
            citations_ok = bool(cited_ids) and not invalid_citations
        else:
            answer_ok = status == expected_status
            citations_ok = len(cited_ids) == 0

        rows.append(
            {
                "question": question,
                "status": status,
                "expected_status": expected_status,
                "status_ok": status == expected_status,
                "intent": diag.get("intent"),
                "hits": len(chunks),
                "generation_present": bool(generated_answer),
                "cited_count": len(cited_ids),
                "invalid_citations": invalid_citations,
                "evidence_ok": evidence_ok,
                "answer_ok": answer_ok,
                "citations_ok": citations_ok,
                "top_source": chunks[0].get("source_file") if chunks else None,
            }
        )

    counts = Counter(str(r["status"]) for r in rows)
    retrieval_rows = [r for r in rows if r.get("status") == "retrieval_complete"]
    summary = {
        "total_cases": len(rows),
        "status_pass": sum(1 for r in rows if r.get("status_ok")),
        "retrieval_cases": len(retrieval_rows),
        "evidence_pass": sum(1 for r in retrieval_rows if r.get("evidence_ok")),
        "answer_pass": sum(1 for r in retrieval_rows if r.get("answer_ok")),
        "citation_pass": sum(1 for r in retrieval_rows if r.get("citations_ok")),
        "counts": dict(counts),
    }

    payload = {
        "summary": summary,
        "rows": rows,
    }

    print("SUMMARY=", json.dumps(summary, ensure_ascii=False))
    print("ROWS=")
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
