#!/usr/bin/env python3
"""Rewrite parquet without legacy HuggingFace schema metadata; preserve Arrow rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    source, output, manifest = map(Path, (args.source, args.output, args.manifest))
    table = pq.read_table(source)
    clean = table.replace_schema_metadata(None)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(clean, tmp, compression="zstd")
    check = pq.read_table(tmp)
    assert check.num_rows == table.num_rows
    assert check.column_names == table.column_names
    assert check.equals(clean), "sanitized parquet changed Arrow values"
    tmp.replace(output)
    info = {
        "source": str(source.resolve()),
        "source_sha256": digest(source),
        "output": str(output.resolve()),
        "output_sha256": digest(output),
        "rows": table.num_rows,
        "columns": table.column_names,
        "transformation": "remove parquet Arrow schema metadata only",
        "arrow_values_equal": True,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
