"""Normalize and anonymize ticket-system exports.

This tool is intentionally system-agnostic. It does not connect to a live
ticket platform and does not modify tickets. It reads CSV exports, maps them
to the project's standard schema, masks common PII patterns, and writes clean
CSV/JSONL outputs plus a quality report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml is in requirements.txt
    yaml = None


STANDARD_FIELDS = [
    "ticket_id",
    "created_at",
    "updated_at",
    "closed_at",
    "status",
    "category",
    "subcategory",
    "priority",
    "severity",
    "affected_service",
    "assignment_group",
    "channel",
    "location_type",
    "short_description",
    "description",
    "resolution_code",
    "action_taken",
    "root_cause",
    "resolution_status",
    "source_system",
]

REQUIRED_FIELDS = ["ticket_id", "created_at"]
TEXT_FIELDS = [
    "short_description",
    "description",
    "action_taken",
    "root_cause",
    "resolution_status",
]

DEFAULT_ALIASES = {
    "ticket_id": ["ticket_id", "id", "request_id", "incident_id", "kayit_no", "talep_no"],
    "created_at": ["created_at", "created", "open_time", "opened_at", "acilis_tarihi", "talep_tarihi"],
    "updated_at": ["updated_at", "updated", "last_update", "son_guncelleme"],
    "closed_at": ["closed_at", "resolved_at", "closed", "kapanis_tarihi", "cozum_tarihi"],
    "status": ["status", "state", "durum"],
    "category": ["category", "kategori", "main_category", "ana_kategori"],
    "subcategory": ["subcategory", "alt_kategori", "sub_category"],
    "priority": ["priority", "oncelik"],
    "severity": ["severity", "importance", "kritiklik", "onem"],
    "affected_service": ["affected_service", "service", "servis", "etkilenen_servis", "application"],
    "assignment_group": ["assignment_group", "group", "ekip", "atanan_grup"],
    "channel": ["channel", "kanal", "source"],
    "location_type": ["location_type", "lokasyon_tipi", "branch_type", "site_type"],
    "short_description": ["short_description", "summary", "subject", "baslik", "kisa_aciklama"],
    "description": ["description", "details", "aciklama", "talep_aciklamasi"],
    "resolution_code": ["resolution_code", "cozum_kodu", "closure_code"],
    "action_taken": ["action_taken", "resolution", "cozum", "cozum_notu", "closing_note"],
    "root_cause": ["root_cause", "kok_neden", "cause"],
    "resolution_status": ["resolution_status", "cozum_durumu", "close_status"],
}

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+?90[-.\s]?)?(?:0?5\d{2})[-.\s]?\d{3}[-.\s]?\d{2}[-.\s]?\d{2}\b"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "tckn": re.compile(r"\b[1-9]\d{10}\b"),
}


def load_config(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("pyyaml is required to read YAML config files")
        return yaml.safe_load(text) or {}
    return json.loads(text)


def normalize_name(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
    )


def build_column_map(fieldnames: list[str], config: dict[str, Any]) -> dict[str, str]:
    configured = config.get("column_mapping", {}) or {}
    normalized_inputs = {normalize_name(name): name for name in fieldnames}
    mapping: dict[str, str] = {}

    for standard, source in configured.items():
        if standard in STANDARD_FIELDS and source in fieldnames:
            mapping[standard] = source

    for standard, aliases in DEFAULT_ALIASES.items():
        if standard in mapping:
            continue
        candidates = [standard, *aliases]
        for candidate in candidates:
            source = normalized_inputs.get(normalize_name(candidate))
            if source:
                mapping[standard] = source
                break
    return mapping


def read_csv_rows(path: Path, encoding: str | None = None) -> tuple[list[dict[str, str]], list[str], str]:
    encodings = [encoding] if encoding else ["utf-8-sig", "utf-8", "cp1254", "latin1"]
    last_error: Exception | None = None
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, newline="") as handle:
                sample = handle.read(4096)
                handle.seek(0)
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                reader = csv.DictReader(handle, dialect=dialect)
                rows = [{k or "": v or "" for k, v in row.items()} for row in reader]
                return rows, list(reader.fieldnames or []), enc
        except Exception as exc:  # noqa: BLE001 - we try several encodings/dialects
            last_error = exc
    raise RuntimeError(f"Could not read {path}: {last_error}")


def mask_pii(text: str, counts: Counter[str]) -> str:
    result = text or ""
    replacements = {
        "email": "[EMAIL]",
        "phone": "[PHONE]",
        "ipv4": "[IP]",
        "tckn": "[TCKN]",
    }
    for key, pattern in PII_PATTERNS.items():
        result, num = pattern.subn(replacements[key], result)
        counts[key] += num
    return result


def stable_hash(value: str, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
    return digest[:12]


def normalize_row(
    row: dict[str, str],
    mapping: dict[str, str],
    source_system: str,
    pii_counts: Counter[str],
    hash_ticket_id: bool,
    hash_salt: str,
) -> dict[str, str]:
    normalized = {field: "" for field in STANDARD_FIELDS}
    for standard, source in mapping.items():
        normalized[standard] = (row.get(source) or "").strip()

    normalized["source_system"] = normalized["source_system"] or source_system

    if hash_ticket_id and normalized["ticket_id"]:
        normalized["ticket_id"] = f"TCK-{stable_hash(normalized['ticket_id'], hash_salt)}"

    for field in TEXT_FIELDS:
        normalized[field] = mask_pii(normalized.get(field, ""), pii_counts)

    if not normalized["short_description"] and normalized["description"]:
        normalized["short_description"] = normalized["description"][:120]

    return normalized


def collect_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.glob("*.csv") if p.is_file())


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STANDARD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_report(
    input_files: list[Path],
    rows: list[dict[str, str]],
    source_fieldnames: list[str],
    column_map: dict[str, str],
    pii_counts: Counter[str],
    encodings: dict[str, str],
) -> dict[str, Any]:
    missing_counts = {
        field: sum(1 for row in rows if not row.get(field))
        for field in [*REQUIRED_FIELDS, "category", "short_description", "description", "action_taken"]
    }
    date_values = [row["created_at"] for row in rows if row.get("created_at")]
    source_columns = sorted(set(source_fieldnames))
    mapped_sources = set(column_map.values())
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_files": [str(path) for path in input_files],
        "encodings": encodings,
        "rows_written": len(rows),
        "standard_fields": STANDARD_FIELDS,
        "required_fields": REQUIRED_FIELDS,
        "column_mapping": column_map,
        "unmapped_source_columns": [name for name in source_columns if name not in mapped_sources],
        "missing_field_counts": missing_counts,
        "category_distribution": Counter(row.get("category", "") or "[EMPTY]" for row in rows).most_common(20),
        "status_distribution": Counter(row.get("status", "") or "[EMPTY]" for row in rows).most_common(20),
        "created_at_min": min(date_values) if date_values else "",
        "created_at_max": max(date_values) if date_values else "",
        "pii_replacements": dict(pii_counts),
        "quality_notes": [
            "action_taken/resolution fields are important for RAG answer quality.",
            "created_at/category/subcategory fields are important for anomaly detection.",
            "This tool only reads exported files; it does not modify the source ticket system.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize and anonymize ticket exports")
    parser.add_argument("--input", required=True, help="CSV file or folder containing CSV exports")
    parser.add_argument("--output-dir", default="integrations/ticket_collector/outputs")
    parser.add_argument("--config", default="", help="Optional YAML/JSON column mapping config")
    parser.add_argument("--source-system", default="unknown_ticket_system")
    parser.add_argument("--encoding", default="", help="Optional explicit CSV encoding")
    parser.add_argument("--hash-ticket-id", action="store_true")
    parser.add_argument("--hash-salt", default="bt_support_ticket_collector")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(args.config)) if args.config else {}
    files = collect_input_files(input_path)
    if not files:
        raise SystemExit(f"No CSV files found at {input_path}")

    all_rows: list[dict[str, str]] = []
    all_fieldnames: list[str] = []
    pii_counts: Counter[str] = Counter()
    encodings: dict[str, str] = {}
    combined_map: dict[str, str] = {}

    for file_path in files:
        rows, fieldnames, used_encoding = read_csv_rows(file_path, args.encoding or None)
        encodings[str(file_path)] = used_encoding
        all_fieldnames.extend(fieldnames)
        column_map = build_column_map(fieldnames, config)
        combined_map.update(column_map)
        for row in rows:
            all_rows.append(
                normalize_row(
                    row=row,
                    mapping=column_map,
                    source_system=args.source_system,
                    pii_counts=pii_counts,
                    hash_ticket_id=args.hash_ticket_id,
                    hash_salt=args.hash_salt,
                )
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"tickets_standardized_{timestamp}.csv"
    jsonl_path = output_dir / f"tickets_standardized_{timestamp}.jsonl"
    report_path = output_dir / f"export_quality_report_{timestamp}.json"

    write_csv(csv_path, all_rows)
    write_jsonl(jsonl_path, all_rows)
    report = build_report(files, all_rows, all_fieldnames, combined_map, pii_counts, encodings)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Rows written: {len(all_rows)}")
    print(f"CSV: {csv_path}")
    print(f"JSONL: {jsonl_path}")
    print(f"Quality report: {report_path}")


if __name__ == "__main__":
    main()
