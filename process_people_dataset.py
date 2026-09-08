"""Create originals, FAL.ai upscales, and YOLO upper-body crops.

The pipeline is resumable and never edits source images. Credentials are read
only from the FAL_KEY environment variable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


PIPELINE_VERSION = "smu-people-upscale-v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
CSV_COLUMNS = [
    "original_filename", "original_path", "upscaled_path",
    "detected_person_count", "crop_paths_json", "detection_confidences_json",
    "detection_boxes_json", "source_width", "source_height", "upscaled_width",
    "upscaled_height", "person_model", "person_confidence_threshold",
    "upscale_model", "upscale_scale", "primary_category", "people_bucket", "status", "failures_json",
    "needs_review", "review_reasons_json",
]


@dataclass
class Detection:
    confidence: float
    box_xyxy: list[float]
    keypoints_xy: list[list[float]]
    keypoints_confidence: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preserve originals, upscale with FAL.ai, and crop every person."
    )
    parser.add_argument("--input", type=Path, default=Path("smu_images/images"))
    parser.add_argument("--output", type=Path, default=Path("smu_processed"))
    parser.add_argument("--person-model", default="yolo11m-pose.pt")
    parser.add_argument("--confidence", type=float, default=0.18)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--device", default=None, help="YOLO device, e.g. 0 or cpu")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--upscale-model", default="RealESRGAN_x4plus")
    parser.add_argument("--upscale-scale", type=float, default=2.0)
    parser.add_argument("--upscale-retries", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent FAL workers")
    parser.add_argument("--manifest-csv", type=Path, default=Path("smu_images/classified_manifest.csv"), help="Classified manifest path for category mapping")
    parser.add_argument("--crop-from-upscaled", action="store_true", default=True, help="Crop from upscaled images rather than originals")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-upscale", action="store_true")
    parser.add_argument("--skip-detection", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reprocess-detection", action="store_true")
    return parser.parse_args()


def relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    replace_with_retry(temporary, path)


def replace_with_retry(source: Path, destination: Path, attempts: int = 30) -> None:
    """Handle brief Windows/OneDrive locks while publishing a checkpoint."""
    for attempt in range(attempts):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.1 * (attempt + 1))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preserve_original(source: Path, destination: Path) -> None:
    """Copy once; never replace an existing preserved original."""
    if destination.exists():
        if source.stat().st_size != destination.stat().st_size or sha256(source) != sha256(destination):
            raise RuntimeError(f"preserved original already exists but differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def detection_config_id(args: argparse.Namespace) -> str:
    payload = json.dumps(
        {
            "pipeline": PIPELINE_VERSION, "model": args.person_model,
            "confidence": args.confidence, "iou": args.iou,
            "image_size": args.image_size,
        }, sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_detection_cache(path: Path, config_id: str) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("config_id") == config_id and item.get("filename"):
                cache[item["filename"]] = item
    return cache


def append_detection_cache(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def require_pillow() -> Any:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required; install requirements-processing.txt") from error
    return Image


def load_yolo(model_name: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Ultralytics is required; install requirements-processing.txt") from error
    return YOLO(model_name)


def tensor_rows(value: Any) -> list[Any]:
    if value is None:
        return []
    return value.detach().cpu().tolist()


def detect_people(model: Any, image_path: Path, args: argparse.Namespace) -> list[Detection]:
    kwargs: dict[str, Any] = {
        "source": str(image_path), "conf": args.confidence, "iou": args.iou,
        "imgsz": args.image_size, "verbose": False,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    results = model.predict(**kwargs)
    if not results:
        return []
    result = results[0]
    boxes = tensor_rows(getattr(getattr(result, "boxes", None), "xyxy", None))
    scores = tensor_rows(getattr(getattr(result, "boxes", None), "conf", None))
    keypoints = getattr(result, "keypoints", None)
    points = tensor_rows(getattr(keypoints, "xy", None))
    point_scores = tensor_rows(getattr(keypoints, "conf", None))
    detections: list[Detection] = []
    for index, box in enumerate(boxes):
        detections.append(Detection(
            confidence=float(scores[index]),
            box_xyxy=[round(float(value), 3) for value in box],
            keypoints_xy=[
                [round(float(coordinate), 3) for coordinate in point]
                for point in (points[index] if index < len(points) else [])
            ],
            keypoints_confidence=[
                round(float(value), 4)
                for value in (point_scores[index] if index < len(point_scores) else [])
            ],
        ))
    detections.sort(key=lambda detection: detection.box_xyxy[0])
    return detections


def visible_keypoint(detection: Detection, index: int, threshold: float = 0.25) -> bool:
    return (
        index < len(detection.keypoints_xy)
        and index < len(detection.keypoints_confidence)
        and detection.keypoints_confidence[index] >= threshold
        and detection.keypoints_xy[index] != [0.0, 0.0]
    )


def upper_body_square(
    detection: Detection, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    """Return an in-image square containing the face and upper torso."""
    x1, y1, x2, y2 = detection.box_xyxy
    box_width = max(2.0, x2 - x1)
    box_height = max(2.0, y2 - y1)
    face_ids = [index for index in range(5) if visible_keypoint(detection, index)]
    shoulder_ids = [index for index in (5, 6) if visible_keypoint(detection, index)]
    hip_ids = [index for index in (11, 12) if visible_keypoint(detection, index)]
    face_points = [detection.keypoints_xy[index] for index in face_ids]
    shoulder_points = [detection.keypoints_xy[index] for index in shoulder_ids]
    hip_points = [detection.keypoints_xy[index] for index in hip_ids]

    if shoulder_points:
        center_x = sum(point[0] for point in shoulder_points) / len(shoulder_points)
    elif face_points:
        center_x = sum(point[0] for point in face_points) / len(face_points)
    else:
        center_x = (x1 + x2) / 2.0

    top = max(0.0, y1 - 0.03 * box_height)
    if hip_points:
        bottom = sum(point[1] for point in hip_points) / len(hip_points) + 0.08 * box_height
    elif shoulder_points:
        shoulder_y = sum(point[1] for point in shoulder_points) / len(shoulder_points)
        bottom = max(y1 + 0.68 * box_height, shoulder_y + 0.38 * box_height)
    else:
        bottom = y1 + 0.68 * box_height
    bottom = min(float(image_height), max(bottom, top + 0.45 * box_height))

    if len(shoulder_points) == 2:
        shoulder_span = abs(shoulder_points[1][0] - shoulder_points[0][0])
        desired_width = max(1.9 * shoulder_span, 0.76 * box_width)
    else:
        desired_width = 0.88 * box_width
    side = max(desired_width, bottom - top) * 1.10
    side = min(side, float(image_width), float(image_height))
    side = max(2.0, side)
    center_y = (top + bottom) / 2.0
    left = min(max(0.0, center_x - side / 2.0), image_width - side)
    upper = min(max(0.0, center_y - side / 2.0), image_height - side)
    left_i, upper_i, side_i = int(round(left)), int(round(upper)), max(2, int(round(side)))
    if left_i + side_i > image_width:
        left_i = image_width - side_i
    if upper_i + side_i > image_height:
        upper_i = image_height - side_i
    return left_i, upper_i, left_i + side_i, upper_i + side_i


def save_crops(
    source: Path, detections: list[Detection], crops_dir: Path,
    crop_size: int, overwrite: bool,
) -> list[Path]:
    Image = require_pillow()
    outputs: list[Path] = []
    base_stem = source.stem[:-9] if source.stem.endswith("_upscaled") else source.stem
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        for index, detection in enumerate(detections, start=1):
            output = crops_dir / f"{base_stem}_person_{index:02d}.jpg"
            outputs.append(output)
            if output.exists() and not overwrite:
                continue
            cropped = image.crop(upper_body_square(detection, image.width, image.height)).resize(
                (crop_size, crop_size), resample=Image.Resampling.LANCZOS
            )
            temporary = output.with_suffix(output.suffix + ".part")
            cropped.save(temporary, format="JPEG", quality=95, subsampling=0, optimize=True)
            temporary.replace(output)
    return outputs


def scale_detection(detection: Detection, scale_x: float, scale_y: float) -> Detection:
    return Detection(
        confidence=detection.confidence,
        box_xyxy=[
            round(detection.box_xyxy[0] * scale_x, 3),
            round(detection.box_xyxy[1] * scale_y, 3),
            round(detection.box_xyxy[2] * scale_x, 3),
            round(detection.box_xyxy[3] * scale_y, 3),
        ],
        keypoints_xy=[
            [round(pt[0] * scale_x, 3), round(pt[1] * scale_y, 3)]
            for pt in detection.keypoints_xy
        ],
        keypoints_confidence=list(detection.keypoints_confidence),
    )


def load_category_map(csv_path: Path) -> dict[str, dict[str, str]]:
    category_map: dict[str, dict[str, str]] = {}
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                fn = row.get("filename")
                if fn:
                    category_map[fn] = {
                        "primary_category": row.get("primary_category") or "uncategorized",
                        "people_bucket": row.get("people_bucket") or "no_people",
                        "scene_category": row.get("scene_category") or "uncategorized",
                    }
    return category_map


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except Exception:
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass


def validated_dimensions(path: Path) -> tuple[int, int]:
    Image = require_pillow()
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size


def download_file(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "smu-dataset-pipeline/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    validated_dimensions(temporary)
    temporary.replace(destination)


def upscale_with_fal(
    source: Path, destination: Path, args: argparse.Namespace
) -> tuple[int, int]:
    if destination.exists() and not args.overwrite:
        return validated_dimensions(destination)
    if not os.environ.get("FAL_KEY"):
        raise RuntimeError("FAL_KEY is not set")
    try:
        import fal_client
    except ImportError as error:
        raise RuntimeError("fal-client is required; install requirements-processing.txt") from error
    last_error: Exception | None = None
    for attempt in range(1, args.upscale_retries + 1):
        try:
            source_url = fal_client.upload_file(source)
            result = fal_client.subscribe(
                "fal-ai/esrgan",
                arguments={
                    "image_url": source_url, "scale": args.upscale_scale,
                    "model": args.upscale_model, "face": False,
                    "output_format": "jpeg",
                },
                with_logs=False, client_timeout=900,
            )
            result_url = result.get("image", {}).get("url")
            if not result_url:
                raise RuntimeError("FAL response did not include image.url")
            download_file(result_url, destination)
            return validated_dimensions(destination)
        except Exception as error:
            last_error = error
            if attempt < args.upscale_retries:
                time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"FAL upscale failed after {args.upscale_retries} attempts: {last_error}")


def image_files(folder: Path) -> Iterable[Path]:
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def write_manifests(output: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = output / "manifest.csv"
    csv_temporary = csv_path.with_suffix(".csv.tmp")
    with csv_temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in CSV_COLUMNS} for row in rows)
    replace_with_retry(csv_temporary, csv_path)
    atomic_json(output / "manifest.json", {
        "pipeline_version": PIPELINE_VERSION,
        "image_count": len(rows),
        "people_detected": sum(int(row["detected_person_count"]) for row in rows),
        "successful": sum(row["status"] == "complete" for row in rows),
        "partial_or_failed": sum(row["status"] in {"partial", "failed"} for row in rows),
        "awaiting_upscale": sum(row["status"] == "awaiting_upscale" for row in rows),
        "needs_review": sum(bool(row.get("needs_review")) for row in rows),
        "images": rows,
    })


def process_single_image(
    source: Path,
    args: argparse.Namespace,
    output: Path,
    originals_dir: Path,
    upscaled_dir: Path,
    crops_dir: Path,
    config_id: str,
    cache: dict[str, dict[str, Any]],
    cache_lock: threading.Lock,
    cache_path: Path,
    model: Any,
    category_map: dict[str, dict[str, str]],
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    original_output = originals_dir / source.name
    upscaled_output = upscaled_dir / f"{source.stem}_upscaled.jpg"
    detections: list[Detection] = []
    crop_outputs: list[Path] = []
    source_width = source_height = upscaled_width = upscaled_height = 0

    try:
        source_width, source_height = validated_dimensions(source)
        preserve_original(source, original_output)
    except Exception as error:
        failures.append({"stage": "original", "error": str(error)})

    if not args.skip_detection and not failures:
        try:
            source_signature = f"{source.stat().st_size}:{source.stat().st_mtime_ns}"
            cached = None
            with cache_lock:
                cached = cache.get(source.name)
            if cached and cached.get("source_signature") == source_signature:
                detections = [Detection(**item) for item in cached["detections"]]
            else:
                detections = detect_people(model, source, args)
                cached = {
                    "filename": source.name,
                    "source_signature": source_signature,
                    "config_id": config_id,
                    "detections": [asdict(item) for item in detections],
                }
                with cache_lock:
                    append_detection_cache(cache_path, cached)
                    cache[source.name] = cached
        except Exception as error:
            failures.append({"stage": "person_detection", "error": str(error)})

    if not args.skip_upscale and not any(item["stage"] == "original" for item in failures):
        try:
            upscaled_width, upscaled_height = upscale_with_fal(source, upscaled_output, args)
        except Exception as error:
            failures.append({"stage": "upscale", "error": str(error)})

    category_info = category_map.get(source.name, {})
    cat = category_info.get("primary_category", "uncategorized")
    bucket = category_info.get("people_bucket", "no_people")
    if upscaled_output.exists():
        cat_upscaled_path = output / "upscaled_by_category" / cat / bucket / upscaled_output.name
        link_or_copy(upscaled_output, cat_upscaled_path)

    if not args.skip_detection and detections and not failures:
        try:
            if args.crop_from_upscaled and upscaled_output.exists() and source_width > 0 and source_height > 0:
                scale_x = upscaled_width / source_width
                scale_y = upscaled_height / source_height
                active_detections = [scale_detection(d, scale_x, scale_y) for d in detections]
                crop_source = upscaled_output
            else:
                active_detections = detections
                crop_source = source
            crop_outputs = save_crops(crop_source, active_detections, crops_dir, args.crop_size, args.overwrite)
            for crop_p in crop_outputs:
                cat_crop_p = output / "crops_by_category" / bucket / crop_p.name
                link_or_copy(crop_p, cat_crop_p)
        except Exception as error:
            failures.append({"stage": "cropping", "error": str(error)})

    if failures:
        status = "partial"
    elif args.skip_upscale:
        status = "awaiting_upscale"
    elif args.skip_detection:
        status = "detection_skipped"
    else:
        status = "complete"
    if failures and any(item["stage"] == "original" for item in failures):
        status = "failed"

    crop_paths = [relative_path(path, output) for path in crop_outputs]
    confidences = [round(item.confidence, 4) for item in detections]
    boxes = [item.box_xyxy for item in detections]
    review_reasons: list[str] = []
    for index, detection in enumerate(detections, start=1):
        if detection.confidence < 0.30:
            review_reasons.append(f"person_{index:02d}_low_confidence")
        if not any(visible_keypoint(detection, point) for point in range(5)):
            review_reasons.append(f"person_{index:02d}_face_keypoints_not_visible")

    return {
        "original_filename": source.name,
        "original_path": relative_path(original_output, output),
        "upscaled_path": relative_path(upscaled_output, output) if upscaled_output.exists() else "",
        "detected_person_count": len(detections),
        "crop_paths": crop_paths,
        "crop_paths_json": json.dumps(crop_paths),
        "detection_confidences": confidences,
        "detection_confidences_json": json.dumps(confidences),
        "detection_boxes": boxes,
        "detection_boxes_json": json.dumps(boxes),
        "source_width": source_width,
        "source_height": source_height,
        "upscaled_width": upscaled_width,
        "upscaled_height": upscaled_height,
        "person_model": args.person_model,
        "person_confidence_threshold": args.confidence,
        "upscale_model": "fal-ai/esrgan:" + args.upscale_model,
        "upscale_scale": args.upscale_scale,
        "primary_category": cat,
        "people_bucket": bucket,
        "status": status,
        "failures": failures,
        "failures_json": json.dumps(failures),
        "needs_review": bool(review_reasons),
        "review_reasons": review_reasons,
        "review_reasons_json": json.dumps(review_reasons),
    }


def main() -> int:
    args = parse_args()
    source_dir = args.input.resolve()
    output = args.output.resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {source_dir}")
    if args.crop_size < 64:
        raise SystemExit("--crop-size must be at least 64")
    if not 0 < args.confidence <= 1:
        raise SystemExit("--confidence must be between 0 and 1")
    if not args.skip_upscale and not os.environ.get("FAL_KEY"):
        raise SystemExit("FAL_KEY is not set. Set it in the environment, or use --skip-upscale.")

    originals_dir, upscaled_dir = output / "original", output / "upscaled"
    crops_dir = output / "upper_body_crops"
    for folder in (output, originals_dir, upscaled_dir, crops_dir):
        folder.mkdir(parents=True, exist_ok=True)
    files = list(image_files(source_dir))
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"No images found in {source_dir}")

    config_id = detection_config_id(args)
    cache_path = output / "detection_cache.jsonl"
    cache = {} if args.reprocess_detection else load_detection_cache(cache_path, config_id)
    model = None if args.skip_detection else load_yolo(args.person_model)
    category_map = load_category_map(args.manifest_csv)
    cache_lock = threading.Lock()
    manifest_lock = threading.Lock()
    rows: list[dict[str, Any]] = []
    completed_count = 0

    print(f"Processing {len(files)} images with {args.concurrency} concurrent workers...", flush=True)
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_file = {
            executor.submit(
                process_single_image,
                file,
                args,
                output,
                originals_dir,
                upscaled_dir,
                crops_dir,
                config_id,
                cache,
                cache_lock,
                cache_path,
                model,
                category_map,
            ): file
            for file in files
        }
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                row = future.result()
            except Exception as exc:
                print(f"[ERROR] {file.name} generated an exception: {exc}", flush=True)
                row = {
                    "original_filename": file.name,
                    "status": "failed",
                    "failures": [{"stage": "worker", "error": str(exc)}],
                    "failures_json": json.dumps([{"stage": "worker", "error": str(exc)}]),
                }
            with manifest_lock:
                rows.append(row)
                completed_count += 1
                status_icon = "OK" if row.get("status") in {"complete", "awaiting_upscale"} else "WARN"
                print(f"[{completed_count}/{len(files)}] [{status_icon}] {file.name}", flush=True)
                if completed_count % 10 == 0 or completed_count == len(files):
                    write_manifests(output, rows)

    # Sort rows by original filename so manifest order is stable
    rows.sort(key=lambda r: r.get("original_filename", ""))
    write_manifests(output, rows)

    print(
        f"Done: {len(rows)} originals, "
        f"{sum(int(row.get('detected_person_count', 0)) for row in rows)} person crops, "
        f"{sum(row.get('status') in {'partial', 'failed'} for row in rows)} partial/failed, "
        f"{sum(row.get('status') == 'awaiting_upscale' for row in rows)} awaiting upscale.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
