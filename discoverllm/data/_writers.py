"""
Output formatters for :mod:`discoverllm.data.build_dataset`.

Three target formats — HF dataset, JSON array, JSONL — kept here so the
main module stays focused on row extraction and reward recalculation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

try:
    # Optional dependency: only needed for ``save_format == "hf"``.
    from datasets import Dataset  # type: ignore
except Exception:  # pragma: no cover - handled at call time
    Dataset = None  # type: ignore[assignment]


def save_dataset(rows: List[Dict[str, Any]], output_path: str, save_format: str) -> None:
    """
    Write ``rows`` to ``output_path`` in one of three formats.

    * ``"hf"``    — HuggingFace ``Dataset.save_to_disk``. Nested
                    ``criteria_history`` lists are JSON-serialised first to
                    avoid Arrow schema issues.
    * ``"jsonl"`` — newline-delimited JSON, one row per line.
    * ``"json"``  — a single pretty-printed JSON array (default for any
                    other value of ``save_format``).
    """
    path = Path(output_path)

    if save_format == "hf":
        if Dataset is None:
            raise ImportError(
                "The `datasets` package is required for --save_format hf. "
                "Install it with `pip install datasets`."
            )
        # Arrow can't infer schemas for deeply nested lists like criteria_history;
        # round-trip them through JSON strings.
        hf_rows = []
        for row in rows:
            hf_row = row.copy()
            if "criteria_history" in hf_row and isinstance(hf_row["criteria_history"], list):
                hf_row["criteria_history"] = json.dumps(hf_row["criteria_history"])
            hf_rows.append(hf_row)
        Dataset.from_list(hf_rows).save_to_disk(str(path))  # type: ignore[call-arg]
        return

    if save_format == "jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return

    # Default: json.
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def save_conversations_jsonl(conversations: List[Dict[str, Any]], output_path: str) -> None:
    """
    Write ``conversations`` (list of ``{"messages": [...]}`` dicts) to a JSONL file.
    """
    with Path(output_path).open("w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
