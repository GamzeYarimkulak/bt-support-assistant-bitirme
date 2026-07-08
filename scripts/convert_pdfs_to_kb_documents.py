"""Convert PDF knowledge-base documents into the project KB CSV schema."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SENSITIVE_PATTERNS = [
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE), "[EMAIL]"),
    (re.compile(r"\b(?:\+?90\s*)?(?:0\s*)?5\d{2}[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}\b"), "[PHONE]"),
    (re.compile(r"\b\d{3}[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}\b"), "[PHONE]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP]"),
    (re.compile(r"\b[1-9]\d{10}\b"), "[ID]"),
]


def import_pypdf() -> Any:
    try:
        import pypdf
    except ImportError as exc:
        raise SystemExit("pypdf is required. Install dependencies from requirements.txt.") from exc
    return pypdf


def stable_slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_value).strip("_").lower()
    return slug or "document"


def mask_sensitive_text(text: str) -> str:
    masked = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        masked = pattern.sub(replacement, masked)
    return masked


def extract_pdf_text(pdf_path: Path, max_pages: int | None = None) -> tuple[str, int]:
    pypdf = import_pypdf()
    page_texts: list[str] = []

    with pdf_path.open("rb") as handle:
        reader = pypdf.PdfReader(handle)
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                pass

        pages = reader.pages[:max_pages] if max_pages else reader.pages
        for page_number, page in enumerate(pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                page_texts.append(f"[Page {page_number}] {text}")

    return "\n".join(page_texts).strip(), len(reader.pages)


def load_manual_text(pdf_path: Path, manual_text_dir: Path | None) -> str:
    """Load manually prepared OCR text for scanned/image-only PDFs."""
    candidates = [pdf_path.with_suffix(".txt")]
    if manual_text_dir is not None:
        candidates.append(manual_text_dir / f"{stable_slug(pdf_path.stem)}.txt")
        candidates.append(manual_text_dir / f"{pdf_path.stem}.txt")

    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8").strip()
    return ""


def convert_pdfs(
    input_dir: Path,
    output_path: Path,
    category: str,
    min_chars: int,
    max_pages: int | None,
    mask_sensitive: bool,
    manual_text_dir: Path | None = None,
) -> dict[str, Any]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    manual_texts_used: list[str] = []

    for index, pdf_path in enumerate(pdf_files, start=1):
        try:
            content, page_count = extract_pdf_text(pdf_path, max_pages=max_pages)
        except Exception as exc:
            skipped.append({"file": pdf_path.name, "reason": str(exc)})
            continue

        if len(content) < min_chars:
            manual_content = load_manual_text(pdf_path, manual_text_dir)
            if manual_content:
                content = manual_content
                manual_texts_used.append(pdf_path.name)

        if mask_sensitive:
            content = mask_sensitive_text(content)

        if len(content) < min_chars:
            skipped.append({"file": pdf_path.name, "reason": f"text shorter than {min_chars} chars"})
            continue

        digest = hashlib.sha1(pdf_path.name.encode("utf-8")).hexdigest()[:8]
        document_id = f"ozdilek_kb_{index:04d}_{digest}"
        title = pdf_path.stem

        rows.append(
            {
                "document_id": document_id,
                "title": title,
                "content": content,
                "category": category,
                "source": str(pdf_path.relative_to(PROJECT_ROOT)) if pdf_path.is_relative_to(PROJECT_ROOT) else str(pdf_path),
                "is_synthetic": False,
                "page_count": page_count,
                "source_slug": stable_slug(pdf_path.stem),
            }
        )

    fieldnames = [
        "document_id",
        "title",
        "content",
        "category",
        "source",
        "is_synthetic",
        "page_count",
        "source_slug",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "output": str(output_path),
        "pdf_files": len(pdf_files),
        "documents_written": len(rows),
        "documents_skipped": len(skipped),
        "manual_texts_used": manual_texts_used,
        "skipped": skipped,
        "mask_sensitive": mask_sensitive,
    }

    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PDF files to KB documents CSV.")
    parser.add_argument("--input-dir", default="data/ozdilek_dokumanlar", help="Directory containing PDF files.")
    parser.add_argument(
        "--output",
        default="data/raw/kb/converted/ozdilek_kb_documents.csv",
        help="Output CSV path in the KB schema.",
    )
    parser.add_argument("--category", default="Ozdilek BT Dokumanlari", help="Category value for all documents.")
    parser.add_argument("--min-chars", type=int, default=80, help="Skip PDFs with less extracted text.")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional page limit per PDF.")
    parser.add_argument(
        "--manual-text-dir",
        default="data/raw/kb/manual_ocr",
        help="Directory with manually prepared OCR .txt files for scanned PDFs.",
    )
    parser.add_argument("--no-mask-sensitive", action="store_true", help="Disable basic email/phone/IP/id masking.")
    args = parser.parse_args()

    summary = convert_pdfs(
        input_dir=(PROJECT_ROOT / args.input_dir).resolve(),
        output_path=(PROJECT_ROOT / args.output).resolve(),
        category=args.category,
        min_chars=args.min_chars,
        max_pages=args.max_pages,
        mask_sensitive=not args.no_mask_sensitive,
        manual_text_dir=(PROJECT_ROOT / args.manual_text_dir).resolve() if args.manual_text_dir else None,
    )

    print("PDF KB conversion complete.")
    print(f"PDF files found: {summary['pdf_files']}")
    print(f"Documents written: {summary['documents_written']}")
    print(f"Documents skipped: {summary['documents_skipped']}")
    print(f"Output: {summary['output']}")
    print(f"Summary: {Path(summary['output']).with_suffix('.summary.json')}")


if __name__ == "__main__":
    sys.exit(main())
