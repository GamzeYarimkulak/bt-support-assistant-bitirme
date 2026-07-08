"""
Prepare raw ticket and KB data for indexing.

This script is intentionally standalone. It reads the current raw data layout,
normalizes heterogeneous CSV/JSONL schemas, and writes processed outputs without
moving or deleting any source files.

Generated outputs:
- data/processed/tickets.csv
- data/processed/tickets.parquet
- data/processed/kb_documents.csv
- data/processed/kb_chunks.jsonl
- data/processed/data_summary.json

The data is synthetic and open-source-assisted test data; it is not real
corporate data.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TICKET_INPUT_DIRS = [
    RAW_DIR / "tickets" / "synthetic",
    RAW_DIR / "tickets" / "original",
    RAW_DIR / "tickets" / "external",
    RAW_DIR / "tickets" / "converted",
]

KB_INPUT_DIR = RAW_DIR / "kb"

TICKET_COLUMNS = [
    "ticket_id",
    "created_at",
    "category",
    "subcategory",
    "short_description",
    "description",
    "resolution",
    "priority",
    "status",
    "source",
    "is_synthetic",
]

PROJECT_SOURCE_TYPES = {"synthetic", "converted", "original"}

BT_CATEGORY_ALLOWLIST = {
    "Access/Permissions",
    "Ağ",
    "Azure",
    "Citrix",
    "Cloud/Infrastructure",
    "Data Analytics & Reporting",
    "Database",
    "Depolama",
    "Donanım",
    "E-posta",
    "Email/Communication",
    "Exchange",
    "Güvenlik",
    "Hardware",
    "IT Support",
    "Intune",
    "Kimlik & Erişim",
    "Network",
    "Network Infrastructure",
    "Release Management",
    "Security",
    "Security Operations",
    "Security+Azure",
    "Security+SharePoint",
    "Service Outages and Maintenance",
    "SharePoint",
    "Software",
    "Software Development",
    "Teams",
    "Technical Support",
    "Uygulama",
    "VMware",
    "Yazıcı",
    "Yazılım",
}
BT_CATEGORY_ALLOWLIST_NORMALIZED = {category.casefold() for category in BT_CATEGORY_ALLOWLIST}

KB_COLUMNS = [
    "document_id",
    "title",
    "content",
    "category",
    "source",
    "is_synthetic",
]


def fix_cp1254_mojibake(value: Any) -> str:
    """Fix cp1254/latin1 mojibake when Turkish labels were read through the wrong encoding."""
    text = "" if value is None else str(value).strip()
    if not text:
        return ""

    if any(char in text for char in "ðþýÝÐÞ"):
        try:
            return text.encode("latin1").decode("cp1254")
        except UnicodeError:
            return text

    return text


def safe_slug(value: str) -> str:
    """Return a stable, filesystem-friendly identifier fragment."""
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "item"


def relative_source(path: Path) -> str:
    """Return a stable project-relative source path."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def read_csv_safely(path: Path) -> pd.DataFrame:
    """Read a CSV file with a small encoding fallback chain."""
    encodings = ["utf-8-sig", "utf-8", "cp1254", "latin1"]
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                encoding=encoding,
                on_bad_lines="skip",
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Could not read {path} with supported encodings: {last_error}",
    )


def read_jsonl_safely(path: Path) -> pd.DataFrame:
    """Read a JSONL file into a DataFrame, skipping empty lines."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_number}: {exc}") from exc
            if isinstance(value, dict):
                records.append(value)

    return pd.DataFrame(records)


def series_from_first_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    default: str = "",
) -> pd.Series:
    """Return the first matching column as clean strings, or a default series."""
    lower_to_actual = {column.lower(): column for column in frame.columns}

    for candidate in candidates:
        actual = lower_to_actual.get(candidate.lower())
        if actual is not None:
            return frame[actual].fillna("").astype(str).str.strip()

    return pd.Series([default] * len(frame), index=frame.index, dtype="object")


def fill_empty(series: pd.Series, default: str) -> pd.Series:
    """Fill empty string-like values with a default."""
    cleaned = series.fillna("").astype(str).str.strip()
    return cleaned.mask(cleaned == "", default)


def infer_source_type(path: Path) -> str:
    """Classify a source file from its path."""
    parts = {part.lower() for part in path.parts}
    if "synthetic" in parts:
        return "synthetic"
    if "original" in parts:
        return "original"
    if "external" in parts:
        return "external"
    if "converted" in parts:
        return "converted"
    return "unknown"


def infer_is_synthetic(path: Path) -> bool:
    """Mark generated project data as synthetic."""
    source_type = infer_source_type(path)
    return source_type in {"synthetic", "converted"}


def infer_language(path: Path, frame: pd.DataFrame) -> pd.Series:
    """Infer language from a language column or from the source path."""
    language = series_from_first_column(frame, ["language", "lang"], default="")
    path_text = path.as_posix().lower()

    if "english" in path_text:
        language = fill_empty(language, "en")
    elif "german" in path_text:
        language = fill_empty(language, "de")
    elif "multilingual" in path_text:
        language = fill_empty(language, "multi")
    else:
        language = fill_empty(language, "tr")

    return language


def stable_created_at(path: Path, length: int) -> pd.Series:
    """Generate deterministic timestamp defaults for rows without dates."""
    seed = sum(ord(char) for char in relative_source(path)) % 365
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=seed)
    values = [
        (start + timedelta(minutes=index)).replace(tzinfo=None).isoformat(timespec="seconds")
        for index in range(length)
    ]
    return pd.Series(values, dtype="object")


def normalize_created_at(path: Path, frame: pd.DataFrame) -> pd.Series:
    """Normalize known date columns to ISO-like strings with safe defaults."""
    raw_dates = series_from_first_column(
        frame,
        ["created_at", "created_date", "created", "timestamp", "date"],
        default="",
    )
    defaults = stable_created_at(path, len(frame))
    parsed = pd.to_datetime(raw_dates, errors="coerce")
    formatted = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S")
    return formatted.fillna(defaults).mask(formatted.fillna("") == "", defaults)


def normalize_ticket_file(path: Path) -> pd.DataFrame:
    """Normalize one raw ticket CSV file to the project ticket schema."""
    frame = read_csv_safely(path)
    source = relative_source(path)
    source_slug = safe_slug(path.stem)
    source_type = infer_source_type(path)
    is_synthetic = infer_is_synthetic(path)

    ticket_id = series_from_first_column(
        frame,
        ["ticket_id", "id", "number", "case_id", "incident_id"],
        default="",
    )
    generated_ids = pd.Series(
        [f"{source_slug}-{index + 1:06d}" for index in range(len(frame))],
        index=frame.index,
        dtype="object",
    )
    ticket_id = ticket_id.mask(ticket_id == "", generated_ids)

    category = fill_empty(
        series_from_first_column(frame, ["category", "queue", "type", "product_service"], default=""),
        "Genel BT",
    )
    subcategory = fill_empty(
        series_from_first_column(frame, ["subcategory", "sub_category", "service", "tag_1"], default=""),
        "Genel",
    )
    short_description = fill_empty(
        series_from_first_column(frame, ["short_description", "subject", "title", "summary"], default=""),
        "BT destek kaydi",
    )
    description = fill_empty(
        series_from_first_column(frame, ["description", "body", "content", "details", "text"], default=""),
        short_description,
    )
    resolution = series_from_first_column(frame, ["resolution", "answer", "solution"], default="")
    priority = fill_empty(
        series_from_first_column(frame, ["priority", "urgency", "priority_score"], default=""),
        "medium",
    )
    status = series_from_first_column(frame, ["status", "state"], default="")
    status = status.mask((status == "") & (resolution != ""), "resolved")
    status = fill_empty(status, "open")

    normalized = pd.DataFrame(
        {
            "ticket_id": ticket_id,
            "created_at": normalize_created_at(path, frame),
            "category": category,
            "subcategory": subcategory,
            "short_description": short_description,
            "description": description,
            "resolution": resolution,
            "priority": priority,
            "status": status,
            "source": source,
            "is_synthetic": is_synthetic,
            "_language": infer_language(path, frame),
            "_source_type": source_type,
        }
    )

    return normalized


def iter_ticket_files() -> list[Path]:
    """Return ticket CSV files from the supported raw input directories."""
    files: list[Path] = []
    for directory in TICKET_INPUT_DIRS:
        if not directory.exists():
            continue
        files.extend(sorted(directory.rglob("*.csv")))
    return files


def make_unique_ids(values: pd.Series, fallback_prefix: str) -> pd.Series:
    """Ensure IDs are non-empty and unique while keeping existing values stable."""
    seen: dict[str, int] = {}
    output: list[str] = []

    for index, value in enumerate(values.fillna("").astype(str).str.strip()):
        base = value or f"{fallback_prefix}-{index + 1:06d}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        output.append(base if count == 0 else f"{base}-{count + 1}")

    return pd.Series(output, dtype="object")


def prepare_tickets() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load and normalize all supported ticket files."""
    frames = []
    for path in iter_ticket_files():
        normalized = normalize_ticket_file(path)
        if not normalized.empty:
            frames.append(normalized)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=TICKET_COLUMNS + ["_language", "_source_type"])

    combined["ticket_id"] = make_unique_ids(combined["ticket_id"], "ticket")
    combined = clean_ticket_strings(combined)
    combined, cleanup_stats = filter_tickets_for_bt_scope(combined)

    text = (
        combined["short_description"].fillna("").astype(str)
        + " "
        + combined["description"].fillna("").astype(str)
        + " Cozum: "
        + combined["resolution"].fillna("").astype(str)
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    parquet_frame = combined[TICKET_COLUMNS].copy()
    parquet_frame.insert(0, "id", combined["ticket_id"])
    parquet_frame.insert(1, "text", text)
    parquet_frame["language"] = combined["_language"]
    parquet_frame["source_type"] = combined["_source_type"]

    csv_frame = combined[TICKET_COLUMNS].copy()
    return csv_frame, parquet_frame, cleanup_stats


def clean_ticket_strings(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize string fields after all heterogeneous sources are merged."""
    cleaned = frame.copy()
    for column in [
        "category",
        "subcategory",
        "short_description",
        "description",
        "resolution",
        "priority",
        "status",
    ]:
        cleaned[column] = cleaned[column].map(fix_cp1254_mojibake)
    return cleaned


def is_bt_category(value: Any) -> bool:
    """Return whether a category is within the BT support assistant scope."""
    normalized = fix_cp1254_mojibake(value).casefold()
    return normalized in BT_CATEGORY_ALLOWLIST_NORMALIZED or normalized.startswith("it & technology/")


def ticket_dedupe_text(frame: pd.DataFrame) -> pd.Series:
    """Build the same text surface used by the index for duplicate detection."""
    return (
        frame["short_description"].fillna("").astype(str)
        + " "
        + frame["description"].fillna("").astype(str)
        + " Cozum: "
        + frame["resolution"].fillna("").astype(str)
    ).str.replace(r"\s+", " ", regex=True).str.strip()


def filter_tickets_for_bt_scope(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Keep BT-support scoped rows and remove exact duplicate external ticket text.

    Project-generated sources are preserved because retrieval/anomaly evaluation
    can refer to their stable IDs.
    """
    raw_count = len(frame)
    project_source_mask = frame["_source_type"].isin(PROJECT_SOURCE_TYPES)
    category_scope_mask = frame["category"].map(is_bt_category)
    scope_mask = project_source_mask | category_scope_mask

    scoped = frame[scope_mask].copy()
    scoped["_dedupe_text"] = ticket_dedupe_text(scoped)
    non_empty_mask = scoped["_dedupe_text"] != ""
    removed_empty_text = int((~non_empty_mask).sum())
    scoped = scoped[non_empty_mask].copy()

    protected = scoped["_source_type"].isin(PROJECT_SOURCE_TYPES)
    protected_rows = scoped[protected]
    external_rows = scoped[~protected]
    external_before_dedupe = len(external_rows)
    external_rows = external_rows.drop_duplicates(subset=["_dedupe_text"], keep="first")

    filtered = (
        pd.concat([protected_rows, external_rows], ignore_index=False)
        .sort_index()
        .drop(columns=["_dedupe_text"])
        .reset_index(drop=True)
    )

    cleanup_stats = {
        "raw_ticket_count": int(raw_count),
        "kept_ticket_count": int(len(filtered)),
        "removed_out_of_scope_count": int(raw_count - int(scope_mask.sum())),
        "removed_empty_text_count": removed_empty_text,
        "removed_duplicate_external_text_count": int(external_before_dedupe - len(external_rows)),
        "project_source_types_preserved": sorted(PROJECT_SOURCE_TYPES),
        "bt_category_allowlist": sorted(BT_CATEGORY_ALLOWLIST),
        "bt_category_prefixes": ["IT & Technology/"],
    }

    return filtered, cleanup_stats


def iter_kb_files() -> list[Path]:
    """Return supported KB CSV and JSONL files, excluding source text archives."""
    if not KB_INPUT_DIR.exists():
        return []

    files: list[Path] = []
    for path in sorted(KB_INPUT_DIR.rglob("*")):
        if not path.is_file():
            continue
        if "source_texts" in {part.lower() for part in path.parts}:
            continue
        if path.suffix.lower() in {".csv", ".jsonl"}:
            files.append(path)
    return files


def read_kb_file(path: Path) -> pd.DataFrame:
    """Read one KB source file."""
    if path.suffix.lower() == ".jsonl":
        return read_jsonl_safely(path)
    return read_csv_safely(path)


def normalize_kb_file(path: Path) -> pd.DataFrame:
    """Normalize one KB CSV/JSONL file to the project KB schema."""
    frame = read_kb_file(path)
    source = relative_source(path)
    source_slug = safe_slug(path.stem)
    is_synthetic = infer_is_synthetic(path)

    document_id = series_from_first_column(
        frame,
        ["document_id", "doc_id", "id", "kb_id"],
        default="",
    )
    generated_ids = pd.Series(
        [f"{source_slug}-{index + 1:06d}" for index in range(len(frame))],
        index=frame.index,
        dtype="object",
    )
    document_id = document_id.mask(document_id == "", generated_ids)

    title = fill_empty(
        series_from_first_column(frame, ["title", "subject", "name"], default=""),
        "BT bilgi bankasi dokumani",
    )
    content = fill_empty(
        series_from_first_column(frame, ["content", "text", "body", "description"], default=""),
        title,
    )
    category = fill_empty(
        series_from_first_column(frame, ["category", "doc_type", "type"], default=""),
        "Genel BT",
    )

    return pd.DataFrame(
        {
            "document_id": document_id,
            "title": title,
            "content": content,
            "category": category,
            "source": source,
            "is_synthetic": is_synthetic,
        }
    )


def prepare_kb_documents() -> pd.DataFrame:
    """Load and normalize all supported KB files."""
    frames = []
    for path in iter_kb_files():
        normalized = normalize_kb_file(path)
        if not normalized.empty:
            frames.append(normalized)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=KB_COLUMNS)

    combined["document_id"] = make_unique_ids(combined["document_id"], "kb")
    return combined[KB_COLUMNS].copy()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a document into overlapping character chunks."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + int(chunk_size * 0.75), end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)

    return chunks


def build_kb_chunks(
    kb_documents: pd.DataFrame,
    chunk_size: int,
    overlap: int,
) -> list[dict[str, Any]]:
    """Create JSONL-ready KB chunks compatible with existing index scripts."""
    chunks: list[dict[str, Any]] = []

    for _, row in kb_documents.iterrows():
        document_id = str(row["document_id"])
        document_chunks = chunk_text(str(row["content"]), chunk_size, overlap)
        for chunk_index, text in enumerate(document_chunks):
            chunk_id = f"{document_id}_chunk_{chunk_index:03d}"
            chunks.append(
                {
                    "id": chunk_id,
                    "doc_id": chunk_id,
                    "document_id": document_id,
                    "doc_type": "kb",
                    "title": str(row["title"]),
                    "category": str(row["category"]),
                    "text": text,
                    "content": text,
                    "source": str(row["source"]),
                    "source_pdf": str(row["source"]),
                    "page": 0,
                    "chunk_index": chunk_index,
                    "metadata": {
                        "document_id": document_id,
                        "source": str(row["source"]),
                        "is_synthetic": bool(row["is_synthetic"]),
                        "note": "Sentetik ve acik kaynak destekli test verisidir; gercek kurumsal veri degildir.",
                    },
                }
            )

    return chunks


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write JSONL records and return the number of written rows."""
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_summary(
    tickets: pd.DataFrame,
    ticket_parquet: pd.DataFrame,
    kb_documents: pd.DataFrame,
    kb_chunk_count: int,
    cleanup_stats: dict[str, Any],
) -> dict[str, Any]:
    """Create a compact data summary for audits and later pipeline checks."""
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Veri sentetik ve acik kaynak destekli test verisidir; gercek kurumsal veri degildir.",
        "tickets": {
            "total_ticket_count": int(len(tickets)),
            "category_distribution": tickets["category"].value_counts(dropna=False).to_dict(),
            "source_distribution": tickets["source"].value_counts(dropna=False).to_dict(),
            "synthetic_distribution": tickets["is_synthetic"].astype(str).value_counts(dropna=False).to_dict(),
            "source_type_distribution": ticket_parquet["source_type"].value_counts(dropna=False).to_dict(),
            "cleanup": cleanup_stats,
        },
        "kb": {
            "kb_document_count": int(len(kb_documents)),
            "kb_chunk_count": int(kb_chunk_count),
            "category_distribution": kb_documents["category"].value_counts(dropna=False).to_dict(),
            "source_distribution": kb_documents["source"].value_counts(dropna=False).to_dict(),
            "synthetic_distribution": kb_documents["is_synthetic"].astype(str).value_counts(dropna=False).to_dict(),
        },
        "outputs": {
            "tickets_csv": "data/processed/tickets.csv",
            "tickets_parquet": "data/processed/tickets.parquet",
            "kb_documents_csv": "data/processed/kb_documents.csv",
            "kb_chunks_jsonl": "data/processed/kb_chunks.jsonl",
            "data_summary_json": "data/processed/data_summary.json",
        },
    }


def prepare_data(chunk_size: int = 1200, overlap: int = 150) -> dict[str, Any]:
    """Prepare all processed data outputs and return the summary."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    tickets_csv, tickets_parquet, cleanup_stats = prepare_tickets()
    kb_documents = prepare_kb_documents()
    kb_chunks = build_kb_chunks(kb_documents, chunk_size=chunk_size, overlap=overlap)

    tickets_csv_path = PROCESSED_DIR / "tickets.csv"
    tickets_parquet_path = PROCESSED_DIR / "tickets.parquet"
    kb_documents_path = PROCESSED_DIR / "kb_documents.csv"
    kb_chunks_path = PROCESSED_DIR / "kb_chunks.jsonl"
    summary_path = PROCESSED_DIR / "data_summary.json"

    tickets_csv.to_csv(tickets_csv_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    try:
        tickets_parquet.to_parquet(tickets_parquet_path, index=False, engine="pyarrow")
    except ImportError:
        if not tickets_parquet_path.exists():
            raise
        print(
            f"Warning: pyarrow is not installed; keeping existing {tickets_parquet_path}. "
            "Install pyarrow to regenerate the parquet file."
        )
    kb_documents.to_csv(kb_documents_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    kb_chunk_count = write_jsonl(kb_chunks_path, kb_chunks)

    summary = build_summary(tickets_csv, tickets_parquet, kb_documents, kb_chunk_count, cleanup_stats)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data/processed files for indexing.")
    parser.add_argument("--chunk-size", type=int, default=1200, help="KB chunk size in characters.")
    parser.add_argument("--overlap", type=int, default=150, help="KB chunk overlap in characters.")
    args = parser.parse_args()

    summary = prepare_data(chunk_size=args.chunk_size, overlap=args.overlap)

    print("Processed data prepared.")
    print(f"Tickets: {summary['tickets']['total_ticket_count']}")
    print(f"KB documents: {summary['kb']['kb_document_count']}")
    print(f"KB chunks: {summary['kb']['kb_chunk_count']}")
    print("Outputs:")
    for output in summary["outputs"].values():
        print(f"- {output}")


if __name__ == "__main__":
    main()
