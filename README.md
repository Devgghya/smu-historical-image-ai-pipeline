# SMU Digital Collections image downloader

This script downloads the 296 records returned by the supplied SMU Digital
Collections search. It uses CONTENTdm's public JSON API to enumerate records and
the site's IIIF image service to obtain real JPEGs instead of small browser
thumbnails.

No Python packages or browser extensions are required. Python 3.10+ is enough.

## Run it

From PowerShell in this folder:

```powershell
py -3 .\download_smu_images.py
```

On Windows, `py -3` deliberately selects the normal Windows Python and its
certificate store. If your `python` command points to that installation too,
`python .\download_smu_images.py` works equally well.

The default request is **1600 pixels wide** (medium resolution). Files go to
`smu_images\images`, with `smu_images\manifest.csv` preserving titles, creators,
record links, image links, download status, and byte sizes. `catalog.json` stores
the discovered record list.

Useful alternatives:

```powershell
# First test only five records
py -3 .\download_smu_images.py --limit 5 --output test_download

# Choose another medium-size width
py -3 .\download_smu_images.py --width 2000

# Request original pixel dimensions (much larger)
py -3 .\download_smu_images.py --full-resolution

# Use a different compatible SMU search URL
py -3 .\download_smu_images.py --url "https://digitalcollections.smu.edu/digital/collection/eaa/search/searchterm/Ag2002.1407/page/1"
```

Interrupted runs are resumable: run the same command again and valid existing
files will be skipped. Failed or partial (`.part`) files are retried. Use
`--overwrite` only when you intentionally want to replace downloaded images.

## Local AI classification

The same command can classify downloaded images with two local models:

- **CLIP** assigns a broad visual category: `people`, `landscape`,
  `architecture_monument`, `cityscape_settlement`, `transport_infrastructure`,
  `object_artifact`, or `document_album_cover`.
- **Faster R-CNN (COCO)** estimates the visible-person count and assigns
  `no_people`, `one_person`, `two_people`, `group_3_to_5`, or `group_6_plus`.

This machine already has the required GPU-enabled packages in Python 3.11. Run:

```powershell
py -3.11 .\download_smu_images.py --classify --organize
```

The first classification run downloads the pretrained model weights and caches
them in the normal Hugging Face/PyTorch cache. Results are written to:

- `smu_images\classified_manifest.csv` — training-ready metadata and labels.
- `smu_images\classification_summary.json` — saved category/count totals.
- `smu_images\classification_cache.jsonl` — resumable per-image AI results.
- `smu_images\by_category\<category>\<people_bucket>` — organized hardlinks to
  the original images. Hardlinks normally consume no second copy of image data.

To test the models on only eight images first:

```powershell
py -3.11 .\download_smu_images.py --classify --classify-limit 8
```

For another machine, install the optional dependencies with:

```powershell
py -3.11 -m pip install -r .\requirements-classification.txt
```

Useful controls include `--classification-device cpu`,
`--classification-batch-size 2`, `--person-threshold 0.5`, and `--reclassify`.
Categories can be customized with `--categories labels.json`; the file must be
a JSON object mapping category names to descriptive CLIP prompts and must retain
a `people` category.

Person counts are estimates, especially for faded 19th-century photographs,
montages, occluded figures, and distant crowds. The classified manifest includes
both model scores and a `needs_review`/`review_reason` field so low-confidence or
model-disagreement cases can be checked before training. Labels are multi-axis:
an image with detected people uses `primary_category=people`, while
`scene_category` independently describes its setting (for example landscape or
architecture). This avoids losing either signal in mixed scenes.

## Rights and attribution

The SMU record metadata asks users to cite **DeGolyer Library, Southern
Methodist University** as the source. It also directs users seeking
high-resolution files to contact the library. Public access to an image does not
by itself establish permission for every AI-training or redistribution use, so
review the rights statement and your intended use. The generated manifest keeps
the source record URL beside every image to make attribution and review easier.

## Upscaling and upper-body crops

`process_people_dataset.py` creates a training-ready dataset without changing
the downloaded source files. It uses YOLO pose detection so each detected person
gets an independently framed, square upper-body crop. Full-image upscaling uses
FAL.ai ESRGAN at 2x; face restoration is deliberately disabled to reduce the
risk of inventing historical facial details.

Set the FAL key in the environment (never add it to this folder), then run:

```powershell
$env:FAL_KEY = Read-Host "FAL.ai key"
py -3.11 .\process_people_dataset.py
Remove-Item Env:FAL_KEY
```

The output is resumable and organized as:

```text
smu_processed/
  original/original_filename.jpg
  upscaled/original_filename_upscaled.jpg
  upper_body_crops/original_filename_person_01.jpg
  manifest.csv
  manifest.json
  detection_cache.jsonl
```

For a local test that does not call the paid FAL API:

```powershell
py -3.11 .\process_people_dataset.py --limit 5 --skip-upscale --output smu_processed_test
```

Useful controls include `--device cpu`, `--confidence 0.25`, `--crop-size 1024`,
`--overwrite`, and `--reprocess-detection`. Existing originals are checked but
never overwritten. The CSV and JSON manifests contain person counts, confidence
scores, boxes, crop paths, upscale paths, and stage-specific failure details.
Low-confidence detections and crops without visible face keypoints are marked
`needs_review` in both manifests instead of being silently treated as certain.
