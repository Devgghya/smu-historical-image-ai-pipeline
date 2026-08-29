"""Local AI classification for the downloaded SMU image dataset.

CLIP assigns a broad visual category. A separate COCO object detector estimates
the number of visible people. Models are imported lazily so downloading images
still needs only the Python standard library.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


CLASSIFIER_VERSION = "smu-clip-person-v2"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_SCENE_PROMPTS = {
    "people": (
        "a historical photograph where one or more people occupy the main "
        "foreground, including a posed portrait, family, group, or crowd"
    ),
    "landscape": (
        "a historical photograph of a natural landscape, countryside, mountain, "
        "river, trees, fields, or open land"
    ),
    "architecture_monument": (
        "a historical photograph primarily showing a building, temple, mosque, "
        "church, palace, tomb, cave, ruin, or monument, with few or no people"
    ),
    "cityscape_settlement": (
        "a historical photograph of a city, street, town, market, village, "
        "harbour, or urban settlement"
    ),
    "transport_infrastructure": (
        "a historical photograph primarily showing a train, railway, bridge, "
        "ship, boat, road, station, or transport infrastructure"
    ),
    "object_artifact": (
        "a historical photograph primarily showing an object, artifact, cannon, "
        "cart, machine, equipment, or vehicle"
    ),
    "document_album_cover": (
        "an actual album cover, book cover, title page, map, drawing, or written "
        "document with little or no photographic scene"
    ),
}
CLASSIFICATION_COLUMNS = [
    "primary_category",
    "primary_category_score",
    "second_category",
    "second_category_score",
    "scene_category",
    "scene_category_score",
    "people_count_estimate",
    "people_bucket",
    "contains_people",
    "person_detector_max_score",
    "visual_people_score",
    "image_width",
    "image_height",
    "needs_review",
    "review_reason",
    "category_scores_json",
    "classifier_version",
    "scene_model",
    "person_model",
    "person_threshold",
    "config_id",
]


@dataclass(frozen=True)
class Classification:
    filename: str
    primary_category: str
    primary_category_score: float
    second_category: str
    second_category_score: float
    scene_category: str
    scene_category_score: float
    people_count_estimate: int
    people_bucket: str
    contains_people: bool
    person_detector_max_score: float
    visual_people_score: float
    image_width: int
    image_height: int
    needs_review: bool
    review_reason: str
    category_scores_json: str
    classifier_version: str
    scene_model: str
    person_model: str
    person_threshold: float
    config_id: str


def people_bucket(count: int, visual_people_category: bool = False) -> str:
    if count <= 0:
        return "unknown_people_count" if visual_people_category else "no_people"
    if count == 1:
        return "one_person"
    if count == 2:
        return "two_people"
    if count <= 5:
        return "group_3_to_5"
    return "group_6_plus"


def classification_config_id(
    scene_model: str, person_model: str, person_threshold: float, prompts: dict[str, str]
) -> str:
    payload = json.dumps(
        {
            "version": CLASSIFIER_VERSION,
            "scene_model": scene_model,
            "person_model": person_model,
            "person_threshold": person_threshold,
            "prompts": prompts,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_prompts(path: Path | None) -> dict[str, str]:
    if path is None:
        return dict(DEFAULT_SCENE_PROMPTS)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or len(data) < 2:
        raise ValueError("category JSON must be an object with at least two labels")
    prompts = {str(label).strip(): str(prompt).strip() for label, prompt in data.items()}
    if any(not label or not prompt for label, prompt in prompts.items()):
        raise ValueError("category labels and prompts cannot be empty")
    if "people" not in prompts:
        raise ValueError("category JSON must include a 'people' label")
    return prompts


def batched(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def feature_tensor(model_output: Any) -> Any:
    """Support both Transformers 4 tensor output and Transformers 5 model output."""
    return getattr(model_output, "pooler_output", model_output)


def require_ai_dependencies() -> tuple[Any, ...]:
    try:
        import torch
        from PIL import Image
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_fpn,
        )
        from torchvision.transforms.functional import pil_to_tensor
        from transformers import AutoProcessor, CLIPModel
    except ImportError as error:
        raise RuntimeError(
            "AI classification needs Pillow, PyTorch, torchvision, and "
            "Transformers. See README.md for the install command."
        ) from error
    return (
        torch,
        Image,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_fpn,
        pil_to_tensor,
        AutoProcessor,
        CLIPModel,
    )


def load_cache(path: Path, config_id: str) -> dict[str, Classification]:
    cached: dict[str, Classification] = {}
    if not path.exists():
        return cached
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                data = json.loads(line)
                if data.get("config_id") == config_id:
                    cached[data["filename"]] = Classification(**data)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return cached


def determine_review_reason(
    ordered_scores: list[tuple[str, float]], people_count: int
) -> str:
    reasons: list[str] = []
    top_label, top_score = ordered_scores[0]
    second_score = ordered_scores[1][1]
    people_score = dict(ordered_scores).get("people", 0.0)
    if top_score < 0.40:
        reasons.append("low_category_confidence")
    if top_score - second_score < 0.08:
        reasons.append("category_scores_close")
    if top_label == "people" and people_count == 0:
        reasons.append("visual_people_but_detector_found_none")
    if people_count > 0 and people_score < 0.10:
        reasons.append("detector_people_but_visual_people_score_low")
    return ";".join(reasons)


def build_classification(
    *,
    filename: str,
    scores: dict[str, float],
    person_scores: list[float],
    image_size: tuple[int, int],
    scene_model: str,
    person_model: str,
    person_threshold: float,
    config_id: str,
) -> Classification:
    visual_people_score = scores.get("people", 0.0)
    ranking_scores = dict(scores)
    if person_scores and "people" in ranking_scores:
        # Treat the detector as independent evidence. A fixed boost is enough to
        # resolve mounted historical portraits without overpowering a strong
        # landscape/architecture score when people are merely incidental.
        ranking_scores["people"] += 0.25
        score_total = sum(ranking_scores.values())
        ranking_scores = {
            label: score / score_total for label, score in ranking_scores.items()
        }
    ordered = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    scene_ordered = sorted(
        (
            (label, score)
            for label, score in scores.items()
            if label != "people"
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    scene_category, scene_score = scene_ordered[0]
    document_is_primary = (
        scene_category == "document_album_cover" and scene_score >= 0.65
    )
    if person_scores and not document_is_primary:
        primary = "people"
        primary_score = ranking_scores["people"]
        second, second_score = scene_category, scene_score
    else:
        primary, primary_score = ordered[0]
        second, second_score = ordered[1]
    people_count = len(person_scores)
    review_reason = determine_review_reason(ordered, people_count)
    return Classification(
        filename=filename,
        primary_category=primary,
        primary_category_score=round(primary_score, 6),
        second_category=second,
        second_category_score=round(second_score, 6),
        scene_category=scene_category,
        scene_category_score=round(scene_score, 6),
        people_count_estimate=people_count,
        people_bucket=people_bucket(people_count, primary == "people"),
        contains_people=people_count > 0 or primary == "people",
        person_detector_max_score=round(max(person_scores, default=0.0), 6),
        visual_people_score=round(visual_people_score, 6),
        image_width=image_size[0],
        image_height=image_size[1],
        needs_review=bool(review_reason),
        review_reason=review_reason,
        category_scores_json=json.dumps(ranking_scores, sort_keys=True),
        classifier_version=CLASSIFIER_VERSION,
        scene_model=scene_model,
        person_model=person_model,
        person_threshold=person_threshold,
        config_id=config_id,
    )


def write_classified_manifest(
    path: Path,
    manifest_rows: list[dict[str, str]],
    classifications: dict[str, Classification],
) -> None:
    original_columns = list(manifest_rows[0]) if manifest_rows else []
    fieldnames = original_columns + [
        column for column in CLASSIFICATION_COLUMNS if column not in original_columns
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            classification = classifications.get(row.get("filename", ""))
            if classification is None:
                continue
            data = dict(row)
            label_data = asdict(classification)
            label_data.pop("filename", None)
            data.update(label_data)
            writer.writerow(data)


def organize_classifications(
    image_dir: Path,
    output_dir: Path,
    classifications: dict[str, Classification],
) -> tuple[int, int]:
    linked = 0
    copied = 0
    for classification in classifications.values():
        source = image_dir / classification.filename
        if not source.is_file():
            continue
        destination_dir = (
            output_dir
            / classification.primary_category
            / classification.people_bucket
        )
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / classification.filename
        for stale_path in output_dir.glob(f"*/*/{classification.filename}"):
            if stale_path != destination and stale_path.is_file():
                stale_path.unlink()
        if destination.exists():
            continue
        try:
            os.link(source, destination)
            linked += 1
        except OSError:
            shutil.copy2(source, destination)
            copied += 1
    return linked, copied


def classify_dataset(
    *,
    image_dir: Path,
    manifest_path: Path,
    output_path: Path,
    cache_path: Path,
    categories_path: Path | None,
    scene_model: str = DEFAULT_CLIP_MODEL,
    person_threshold: float = 0.40,
    batch_size: int = 4,
    device_name: str = "auto",
    limit: int | None = None,
    reclassify: bool = False,
    organize_dir: Path | None = None,
) -> dict[str, int]:
    if not manifest_path.is_file():
        raise RuntimeError(f"manifest not found: {manifest_path}")
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest_rows = [
        row for row in manifest_rows if (image_dir / row.get("filename", "")).is_file()
    ]
    if limit is not None:
        manifest_rows = manifest_rows[:limit]
    if not manifest_rows:
        raise RuntimeError("no downloaded images from the manifest were found")

    prompts = load_prompts(categories_path)
    person_model_name = "fasterrcnn_mobilenet_v3_large_fpn_coco"
    config_id = classification_config_id(
        scene_model, person_model_name, person_threshold, prompts
    )
    cached = {} if reclassify else load_cache(cache_path, config_id)
    wanted_names = {row["filename"] for row in manifest_rows}
    classifications = {
        name: value for name, value in cached.items() if name in wanted_names
    }
    pending_rows = [
        row for row in manifest_rows if row["filename"] not in classifications
    ]

    if pending_rows:
        (
            torch,
            Image,
            DetectorWeights,
            detector_builder,
            pil_to_tensor,
            AutoProcessor,
            CLIPModel,
        ) = require_ai_dependencies()
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--classification-device cuda requested, but CUDA is unavailable")
        device = torch.device(device_name)

        print(f"Loading CLIP scene model {scene_model!r} on {device}...", flush=True)
        processor = AutoProcessor.from_pretrained(scene_model)
        clip_model = CLIPModel.from_pretrained(scene_model).to(device).eval()
        print("Loading person detector...", flush=True)
        detector = detector_builder(weights=DetectorWeights.DEFAULT).to(device).eval()

        labels = list(prompts)
        text_inputs = processor(
            text=list(prompts.values()), return_tensors="pt", padding=True
        )
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        with torch.inference_mode():
            text_features = feature_tensor(
                clip_model.get_text_features(**text_inputs)
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        completed = len(classifications)
        with cache_path.open("a", encoding="utf-8") as cache_handle:
            for batch_rows in batched(pending_rows, batch_size):
                images = []
                for row in batch_rows:
                    with Image.open(image_dir / row["filename"]) as image:
                        images.append(image.convert("RGB").copy())

                image_inputs = processor(images=images, return_tensors="pt")
                pixel_values = image_inputs["pixel_values"].to(device)
                detector_inputs = [
                    pil_to_tensor(image).float().div(255).to(device) for image in images
                ]
                with torch.inference_mode():
                    image_features = feature_tensor(
                        clip_model.get_image_features(pixel_values=pixel_values)
                    )
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )
                    probabilities = (100.0 * image_features @ text_features.T).softmax(
                        dim=-1
                    )
                    detections = detector(detector_inputs)

                probabilities_cpu = probabilities.cpu().tolist()
                for row, image, probability_row, detection in zip(
                    batch_rows, images, probabilities_cpu, detections
                ):
                    scores = {
                        label: round(float(score), 6)
                        for label, score in zip(labels, probability_row)
                    }
                    detected_people = [
                        float(score)
                        for label, score in zip(
                            detection["labels"].detach().cpu().tolist(),
                            detection["scores"].detach().cpu().tolist(),
                        )
                        if label == 1 and score >= person_threshold
                    ]
                    result = build_classification(
                        filename=row["filename"],
                        scores=scores,
                        person_scores=detected_people,
                        image_size=image.size,
                        scene_model=scene_model,
                        person_model=person_model_name,
                        person_threshold=person_threshold,
                        config_id=config_id,
                    )
                    classifications[result.filename] = result
                    cache_handle.write(json.dumps(asdict(result)) + "\n")
                    completed += 1
                    print(
                        f"[{completed:03}/{len(manifest_rows):03}] "
                        f"{result.primary_category:24} "
                        f"{result.people_bucket:20} {result.filename}",
                        flush=True,
                    )
                cache_handle.flush()

    write_classified_manifest(output_path, manifest_rows, classifications)
    linked = copied = 0
    if organize_dir is not None:
        linked, copied = organize_classifications(
            image_dir, organize_dir, classifications
        )

    counts: dict[str, int] = {
        "classified": len(classifications),
        "needs_review": sum(item.needs_review for item in classifications.values()),
        "organized_hardlinks": linked,
        "organized_copies": copied,
    }
    for classification in classifications.values():
        key = f"category:{classification.primary_category}"
        counts[key] = counts.get(key, 0) + 1
        bucket_key = f"people:{classification.people_bucket}"
        counts[bucket_key] = counts.get(bucket_key, 0) + 1
    return counts
