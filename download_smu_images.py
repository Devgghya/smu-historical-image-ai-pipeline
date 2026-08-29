#!/usr/bin/env python3
"""Download image records from an SMU Digital Collections search.

The site is powered by CONTENTdm. This downloader uses the site's public JSON
search API to enumerate records and its public IIIF service to request images at
a chosen width, avoiding low-resolution browser thumbnails.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen


DEFAULT_SEARCH_URL = (
    "https://digitalcollections.smu.edu/digital/collection/eaa/"
    "search/searchterm/Ag2002.1407/page/1"
)
PAGE_SIZE = 100
USER_AGENT = "SMU-collection-downloader/1.0 (personal research downloader)"
PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class SearchSpec:
    base_url: str
    collection: str
    search_term: str
    field: str = "all"


@dataclass(frozen=True)
class Record:
    item_id: str
    title: str
    creator: str
    date: str
    part_of: str
    record_url: str
    image_url: str
    filename: str


@dataclass(frozen=True)
class Result:
    record: Record
    status: str
    byte_count: int
    error: str = ""


def parse_search_url(url: str) -> SearchSpec:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Search URL must be an absolute http(s) URL")

    parts = [unquote(part) for part in parsed.path.split("/") if part]

    def value_after(label: str) -> str | None:
        try:
            index = parts.index(label)
        except ValueError:
            return None
        return parts[index + 1] if index + 1 < len(parts) else None

    collection = value_after("collection")
    search_term = value_after("searchterm")
    field = value_after("field") or "all"
    if not collection or search_term is None:
        raise ValueError(
            "URL must contain /collection/<alias>/.../searchterm/<term>/"
        )

    return SearchSpec(
        base_url=f"{parsed.scheme}://{parsed.netloc}",
        collection=collection,
        search_term=search_term,
        field=field,
    )


def safe_slug(text: str, max_length: int = 90) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ascii_text).strip("_")
    return (slug or "untitled")[:max_length].rstrip("_")


def request_bytes(url: str, *, timeout: float, retries: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json,image/jpeg,image/*;q=0.9,*/*;q=0.1",
                },
            )
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            last_error = error
            retryable = not isinstance(error, HTTPError) or error.code in {
                408,
                429,
                500,
                502,
                503,
                504,
            }
            if attempt >= retries or not retryable:
                break
            time.sleep((2**attempt) + random.random())
    assert last_error is not None
    raise last_error


def request_json(url: str, *, timeout: float, retries: int) -> dict[str, Any]:
    return json.loads(request_bytes(url, timeout=timeout, retries=retries))


def search_api_url(spec: SearchSpec, page: int) -> str:
    return (
        f"{spec.base_url}/digital/api/search/collection/"
        f"{quote(spec.collection, safe='')}/searchterm/"
        f"{quote(spec.search_term, safe='')}/field/{quote(spec.field, safe='')}/"
        f"maxRecords/{PAGE_SIZE}/page/{page}"
    )


def image_api_url(spec: SearchSpec, item_id: str, width: int | None) -> str:
    size = "full" if width is None else f"{width},"
    identifier = quote(f"{spec.collection}:{item_id}", safe=":")
    return f"{spec.base_url}/iiif/2/{identifier}/full/{size}/0/default.jpg"


def record_page_url(spec: SearchSpec, item_id: str) -> str:
    return (
        f"{spec.base_url}/digital/collection/{quote(spec.collection, safe='')}/"
        f"id/{quote(item_id, safe='')}"
    )


def metadata_value(item: dict[str, Any], key: str) -> str:
    for entry in item.get("metadataFields", []):
        if entry.get("field") == key:
            return str(entry.get("value") or "")
    return ""


def fetch_records(
    spec: SearchSpec,
    *,
    width: int | None,
    timeout: float,
    retries: int,
    limit: int | None,
) -> tuple[list[Record], int]:
    items: list[dict[str, Any]] = []
    page = 1
    total = 0

    while True:
        payload = request_json(
            search_api_url(spec, page), timeout=timeout, retries=retries
        )
        total = int(payload.get("totalResults", 0))
        page_items = payload.get("items", [])
        if not isinstance(page_items, list):
            raise RuntimeError("The CONTENTdm API returned an unexpected items value")
        items.extend(page_items)

        target = min(total, limit) if limit is not None else total
        if not page_items or len(items) >= target:
            break
        page += 1

    if limit is not None:
        items = items[:limit]

    seen_ids: set[str] = set()
    records: list[Record] = []
    for item in items:
        item_id = str(item.get("itemId") or "").strip()
        if not item_id or item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        title = str(item.get("title") or "Untitled").strip()
        records.append(
            Record(
                item_id=item_id,
                title=title,
                creator=metadata_value(item, "creato"),
                date=metadata_value(item, "datea") or metadata_value(item, "date"),
                part_of=metadata_value(item, "part"),
                record_url=record_page_url(spec, item_id),
                image_url=image_api_url(spec, item_id, width),
                filename=f"{item_id}_{safe_slug(title)}.jpg",
            )
        )

    return records, total


def download_one(
    record: Record,
    image_dir: Path,
    *,
    timeout: float,
    retries: int,
    delay: float,
    overwrite: bool,
) -> Result:
    destination = image_dir / record.filename
    if destination.is_file() and destination.stat().st_size > 0 and not overwrite:
        return Result(record, "skipped", destination.stat().st_size)

    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        data = request_bytes(record.image_url, timeout=timeout, retries=retries)
        if not data.startswith(b"\xff\xd8\xff"):
            raise RuntimeError("server response was not a JPEG image")
        with temporary.open("wb") as handle:
            handle.write(data)
        os.replace(temporary, destination)
        return Result(record, "downloaded", len(data))
    except Exception as error:  # Keep processing the rest of the collection.
        temporary.unlink(missing_ok=True)
        return Result(record, "failed", 0, f"{type(error).__name__}: {error}")
    finally:
        if delay > 0:
            time.sleep(delay)


def write_manifest(path: Path, results: list[Result]) -> None:
    columns = [
        "item_id",
        "title",
        "creator",
        "date",
        "part_of",
        "record_url",
        "image_url",
        "filename",
        "status",
        "bytes",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            row = asdict(result.record)
            row.update(
                {
                    "status": result.status,
                    "bytes": result.byte_count,
                    "error": result.error,
                }
            )
            writer.writerow(row)


def write_catalog(
    path: Path,
    source_search_url: str,
    spec: SearchSpec,
    records: list[Record],
    total: int,
) -> None:
    catalog = {
        "source_search_url": source_search_url,
        "base_url": spec.base_url,
        "collection": spec.collection,
        "search_term": spec.search_term,
        "total_results_reported": total,
        "records_selected": len(records),
        "records": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download medium/full-resolution images from an SMU Digital "
            "Collections search via CONTENTdm and IIIF."
        )
    )
    parser.add_argument("--url", default=DEFAULT_SEARCH_URL, help="SMU search URL")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("smu_images"),
        help="Output directory (default: smu_images)",
    )
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument(
        "--width",
        type=int,
        default=1600,
        help="Requested JPEG width in pixels (default: 1600)",
    )
    size_group.add_argument(
        "--full-resolution",
        action="store_true",
        help="Request each original image's full pixel dimensions",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Concurrent downloads; keep modest for the library server (default: 3)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Polite delay per worker after each image (default: 0.25 seconds)",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--limit", type=int, help="Download only the first N records (useful for a test)"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="Fetch the record list and write metadata, but do not download images",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Run local AI scene classification and person counting after download",
    )
    parser.add_argument(
        "--classify-limit",
        type=int,
        help="Classify only the first N downloaded records (useful for a model test)",
    )
    parser.add_argument(
        "--classification-batch-size",
        type=int,
        default=4,
        help="AI inference batch size (default: 4)",
    )
    parser.add_argument(
        "--classification-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--person-threshold",
        type=float,
        default=0.40,
        help="Person detector confidence threshold from 0 to 1 (default: 0.40)",
    )
    parser.add_argument(
        "--categories",
        type=Path,
        help="Optional JSON object mapping custom category labels to CLIP prompts",
    )
    parser.add_argument(
        "--reclassify",
        action="store_true",
        help="Ignore matching cached AI results and classify again",
    )
    parser.add_argument(
        "--organize",
        action="store_true",
        help="Create by_category folders using hardlinks (copies if unsupported)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.width is not None and args.width < 100:
        raise SystemExit("--width must be at least 100")
    if args.workers < 1 or args.workers > 8:
        raise SystemExit("--workers must be between 1 and 8")
    if args.delay < 0 or args.timeout <= 0 or args.retries < 0:
        raise SystemExit("--delay/--retries cannot be negative; --timeout must be positive")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.classify_limit is not None and args.classify_limit < 1:
        raise SystemExit("--classify-limit must be positive")
    if args.classification_batch_size < 1:
        raise SystemExit("--classification-batch-size must be positive")
    if not 0 < args.person_threshold < 1:
        raise SystemExit("--person-threshold must be between 0 and 1")

    try:
        spec = parse_search_url(args.url)
    except ValueError as error:
        raise SystemExit(f"Invalid --url: {error}") from error

    requested_width = None if args.full_resolution else args.width
    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "images"
    image_dir.mkdir(exist_ok=True)

    print(
        f"Finding records for {spec.collection!r} / {spec.search_term!r}...",
        flush=True,
    )
    records, total = fetch_records(
        spec,
        width=requested_width,
        timeout=args.timeout,
        retries=args.retries,
        limit=args.limit,
    )
    write_catalog(args.output / "catalog.json", args.url, spec, records, total)
    print(f"Found {total} records; selected {len(records)}.", flush=True)

    if args.catalog_only:
        print(f"Catalog written to {args.output / 'catalog.json'}")
        return 0

    results: list[Result] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                download_one,
                record,
                image_dir,
                timeout=args.timeout,
                retries=args.retries,
                delay=args.delay,
                overwrite=args.overwrite,
            ): record
            for record in records
        }
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            completed += 1
            show_item = result.status != "skipped" or not args.classify
            if show_item:
                with PRINT_LOCK:
                    suffix = f" - {result.error}" if result.error else ""
                    print(
                        f"[{completed:03}/{len(records):03}] "
                        f"{result.status:10} {result.record.filename}{suffix}",
                        flush=True,
                    )

    order = {record.item_id: index for index, record in enumerate(records)}
    results.sort(key=lambda result: order[result.record.item_id])
    manifest_path = args.output / "manifest.csv"
    all_skipped = bool(results) and all(
        result.status == "skipped" for result in results
    )
    if all_skipped and manifest_path.is_file():
        print(f"Reusing unchanged manifest: {manifest_path}")
    else:
        write_manifest(manifest_path, results)

    counts = {
        status: sum(result.status == status for result in results)
        for status in ("downloaded", "skipped", "failed")
    }
    total_bytes = sum(result.byte_count for result in results)
    print(
        "Done: "
        f"{counts['downloaded']} downloaded, {counts['skipped']} resumed/skipped, "
        f"{counts['failed']} failed, {total_bytes / (1024 * 1024):.1f} MiB present."
    )
    print(f"Images:   {image_dir}")
    print(f"Manifest: {manifest_path}")
    if args.classify:
        from classify_smu_images import classify_dataset

        classified_manifest = args.output / "classified_manifest.csv"
        print("Starting local AI classification...", flush=True)
        try:
            classification_counts = classify_dataset(
                image_dir=image_dir,
                manifest_path=manifest_path,
                output_path=classified_manifest,
                cache_path=args.output / "classification_cache.jsonl",
                categories_path=args.categories,
                person_threshold=args.person_threshold,
                batch_size=args.classification_batch_size,
                device_name=args.classification_device,
                limit=args.classify_limit,
                reclassify=args.reclassify,
                organize_dir=(args.output / "by_category") if args.organize else None,
            )
        except Exception as error:
            print(f"Classification failed: {error}", file=sys.stderr)
            return 2
        print("Classification complete:")
        for key, value in sorted(classification_counts.items()):
            print(f"  {key}: {value}")
        summary_path = args.output / "classification_summary.json"
        summary_path.write_text(
            json.dumps(classification_counts, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Classified manifest: {classified_manifest}")
        print(f"Classification summary: {summary_path}")
        if args.organize:
            print(f"Category folders:    {args.output / 'by_category'}")

    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
