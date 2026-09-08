"""Colorize + Topaz Generative Restoration & Categorized Cropping Pipeline.

Workflow per image:
1. Border Crop: Remove album cardboard margins and cursive handwriting.
2. Pre-process: Clean grayscale + auto-contrast pre-pass.
3. AI Colorization: fal-ai/ddcolor (model_size='large').
4. Topaz Generative Upscale: topaz/upscale/image/generative.
5. High-res Upper Body Cropping: 1024x1024 square crops from the Topaz image.
6. Auto-organization into categorized folders (one_person, two_people, etc.).
"""

from __future__ import annotations

import argparse
import csv
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
from typing import Any

from PIL import Image, ImageOps

import fal_client
import process_people_dataset as ppd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colorize + Topaz Generative Restoration Pipeline.")
    parser.add_argument("--input", type=Path, default=Path("smu_images/images"))
    parser.add_argument("--output", type=Path, default=Path("smu_topaz_restored"))
    parser.add_argument("--manifest-csv", type=Path, default=Path("smu_images/classified_manifest.csv"))
    parser.add_argument("--detection-cache", type=Path, default=Path("smu_processed/detection_cache.jsonl"))
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrent workers for FAL")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--people-only", action="store_true", default=True, help="Process images containing people")
    parser.add_argument("--all", dest="people_only", action="store_false", help="Process all images including architecture/landscapes")
    parser.add_argument("--limit", type=int, help="Optional image count limit for testing")
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def get_photo_box(width: int, height: int) -> tuple[int, int, int, int]:
    """Return inner photo coordinates without cardboard mount and handwriting."""
    if height > width:
        # Portrait (e.g. Volume I: Costumes and Characters)
        return (
            int(width * 0.14),
            int(height * 0.105),
            int(width * 0.86),
            int(height * 0.805),
        )
    else:
        # Landscape (e.g. Volume II: Scenery and Public Buildings)
        return (
            int(width * 0.185),
            int(height * 0.145),
            int(width * 0.815),
            int(height * 0.825),
        )


def download_with_retry(url: str, destination: Path, retries: int = 3) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "smu-topaz-pipeline/1.0"})
            with urllib.request.urlopen(req, timeout=300) as resp, temporary.open("wb") as out:
                shutil.copyfileobj(resp, out)
            temporary.replace(destination)
            return
        except Exception as err:
            last_err = err
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url} -> {destination}: {last_err}")


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


def process_image(
    image_path: Path,
    meta: dict[str, Any],
    args: argparse.Namespace,
    dirs: dict[str, Path],
    cached_detections: list[dict[str, Any]],
) -> dict[str, Any]:
    stem = image_path.stem
    primary_cat = meta.get("primary_category", "people")
    bucket = meta.get("people_bucket", "one_person")
    
    cropped_orig_path = dirs["cropped_orig"] / f"{stem}_cropped.jpg"
    color_path = dirs["colorized"] / f"{stem}_color.png"
    topaz_path = dirs["topaz"] / f"{stem}_topaz.jpg"

    # Step 1: Border Crop
    with Image.open(image_path) as opened:
        orig_w, orig_h = opened.size
        crop_box = get_photo_box(orig_w, orig_h)
        crop_l, crop_t, crop_r, crop_b = crop_box
        crop_w, crop_h = crop_r - crop_l, crop_b - crop_t
        if not cropped_orig_path.exists():
            cropped_orig = opened.crop(crop_box)
            cropped_orig.save(cropped_orig_path, format="JPEG", quality=95)

    # Step 2: Colorization via fal-ai/ddcolor
    if not color_path.exists():
        # Pre-process: grayscale + auto-contrast
        with Image.open(cropped_orig_path) as c_img:
            gray = ImageOps.grayscale(c_img)
            enhanced = ImageOps.autocontrast(gray, cutoff=1)
            temp_prep = dirs["temp"] / f"{stem}_prep.jpg"
            enhanced.save(temp_prep, format="JPEG", quality=95)

        upload_url = fal_client.upload_file(str(temp_prep))
        res = fal_client.subscribe("fal-ai/ddcolor", arguments={"image_url": upload_url, "model_size": "large"})
        color_url = res.get("image", {}).get("url")
        if not color_url:
            raise RuntimeError(f"ddcolor returned no image url for {image_path.name}")
        download_with_retry(color_url, color_path, retries=args.retries)
        if temp_prep.exists():
            try:
                temp_prep.unlink()
            except Exception:
                pass

    # Step 3: Topaz Generative Upscale via topaz/upscale/image/generative
    if not topaz_path.exists():
        upload_color_url = fal_client.upload_file(str(color_path))
        topaz_res = fal_client.subscribe(
            "topaz/upscale/image/generative",
            arguments={"image_url": upload_color_url},
        )
        topaz_url = topaz_res.get("image", {}).get("url")
        if not topaz_url:
            raise RuntimeError(f"Topaz returned no image url for {image_path.name}")
        download_with_retry(topaz_url, topaz_path, retries=args.retries)

    # Organize full restored image
    cat_topaz_path = dirs["by_category"] / primary_cat / bucket / topaz_path.name
    link_or_copy(topaz_path, cat_topaz_path)

    # Step 4: High-Res Upper Body Crops (from Topaz output)
    crop_paths: list[str] = []
    if cached_detections:
        with Image.open(topaz_path) as top_img:
            topaz_w, topaz_h = top_img.size
            scale_x = topaz_w / crop_w
            scale_y = topaz_h / crop_h

            for idx, det_dict in enumerate(cached_detections, start=1):
                det = ppd.Detection(**det_dict)
                new_box = [
                    round(max(0, (det.box_xyxy[0] - crop_l) * scale_x), 2),
                    round(max(0, (det.box_xyxy[1] - crop_t) * scale_y), 2),
                    round(min(topaz_w, (det.box_xyxy[2] - crop_l) * scale_x), 2),
                    round(min(topaz_h, (det.box_xyxy[3] - crop_t) * scale_y), 2),
                ]
                new_kpts = [
                    [
                        round((pt[0] - crop_l) * scale_x, 2),
                        round((pt[1] - crop_t) * scale_y, 2),
                    ]
                    for pt in det.keypoints_xy
                ]
                mapped_det = ppd.Detection(det.confidence, new_box, new_kpts, det.keypoints_confidence)
                square = ppd.upper_body_square(mapped_det, topaz_w, topaz_h)

                crop_out = dirs["master_crops"] / f"{stem}_person_{idx:02d}.jpg"
                if not crop_out.exists():
                    person_cropped = top_img.crop(square).resize(
                        (args.crop_size, args.crop_size),
                        resample=Image.Resampling.LANCZOS,
                    )
                    temp_crop = crop_out.with_suffix(".part")
                    person_cropped.save(temp_crop, format="JPEG", quality=95, optimize=True)
                    temp_crop.replace(crop_out)

                cat_crop_out = dirs["crops_by_category"] / bucket / crop_out.name
                link_or_copy(crop_out, cat_crop_out)
                crop_paths.append(str(crop_out.relative_to(dirs["output"])))

    return {
        "filename": image_path.name,
        "primary_category": primary_cat,
        "people_bucket": bucket,
        "cropped_original": str(cropped_orig_path.relative_to(dirs["output"])),
        "colorized_path": str(color_path.relative_to(dirs["output"])),
        "topaz_path": str(topaz_path.relative_to(dirs["output"])),
        "person_crops_count": len(crop_paths),
        "person_crop_paths_json": json.dumps(crop_paths),
        "status": "complete",
    }


def main() -> int:
    args = parse_args()
    if not os.environ.get("FAL_KEY"):
        raise SystemExit("FAL_KEY environment variable is required.")

    output = args.output.resolve()
    dirs = {
        "output": output,
        "cropped_orig": output / "cropped_originals",
        "colorized": output / "colorized",
        "topaz": output / "full_restored",
        "by_category": output / "full_restored_by_category",
        "master_crops": output / "upper_body_crops",
        "crops_by_category": output / "crops_by_category",
        "temp": output / ".temp",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Load metadata
    manifest_map: dict[str, dict[str, Any]] = {}
    with args.manifest_csv.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("filename"):
                manifest_map[r["filename"]] = r

    # Load detection cache
    detections_by_file: dict[str, list[dict[str, Any]]] = {}
    if args.detection_cache.exists():
        with args.detection_cache.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    fn = item.get("filename")
                    if fn:
                        detections_by_file[fn] = item.get("detections", [])

    # Select target images
    all_files = sorted(args.input.glob("*.jpg"))
    targets: list[Path] = []
    for p in all_files:
        meta = manifest_map.get(p.name, {})
        contains_people = meta.get("contains_people") == "True" or meta.get("primary_category") == "people"
        if args.people_only and not contains_people:
            continue
        targets.append(p)

    if args.limit:
        targets = targets[:args.limit]

    print(f"Targeting {len(targets)} images (people_only={args.people_only}) with {args.concurrency} workers...", flush=True)

    manifest_rows: list[dict[str, Any]] = []
    manifest_lock = threading.Lock()
    completed = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_map = {
            executor.submit(
                process_image,
                p,
                manifest_map.get(p.name, {}),
                args,
                dirs,
                detections_by_file.get(p.name, []),
            ): p
            for p in targets
        }

        for future in as_completed(future_map):
            p = future_map[future]
            try:
                row = future.result()
            except Exception as err:
                print(f"[ERROR] {p.name}: {err}", flush=True)
                row = {
                    "filename": p.name,
                    "status": "failed",
                    "error": str(err),
                }
            with manifest_lock:
                manifest_rows.append(row)
                completed += 1
                status = "OK" if row.get("status") == "complete" else "ERR"
                crops_cnt = row.get("person_crops_count", 0)
                print(f"[{completed}/{len(targets)}] [{status}] {p.name} ({crops_cnt} crops)", flush=True)

                if completed % 5 == 0 or completed == len(targets):
                    manifest_csv_path = output / "manifest.csv"
                    with manifest_csv_path.open("w", newline="", encoding="utf-8-sig") as mf:
                        writer = csv.DictWriter(mf, fieldnames=[
                            "filename", "primary_category", "people_bucket",
                            "cropped_original", "colorized_path", "topaz_path",
                            "person_crops_count", "person_crop_paths_json", "status", "error"
                        ])
                        writer.writeheader()
                        writer.writerows(
                            {k: r.get(k, "") for k in [
                                "filename", "primary_category", "people_bucket",
                                "cropped_original", "colorized_path", "topaz_path",
                                "person_crops_count", "person_crop_paths_json", "status", "error"
                            ]}
                            for r in manifest_rows
                        )

    print(f"Pipeline completed! Successfully processed {sum(r.get('status') == 'complete' for r in manifest_rows)}/{len(targets)} images.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
