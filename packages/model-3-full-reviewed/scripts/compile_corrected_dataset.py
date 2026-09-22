from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO = Path(r"D:\Actual Project\lcd-data-extractor")
OUTPUT = Path(r"D:\Actual Project\Indali_Lcd-Data-Extractor\packages\model-3-full-reviewed\data\corrected_dataset")
VALIDATION_FRACTION = 0.15
RANDOM_SEED = 20260921

ANNOTATION_ROOTS = [
    REPO / "pilot_90" / "calibration",
    REPO / "pilot_90" / "train",
    REPO / "pilot_90" / "validation",
    REPO / "next_100",
    REPO / "additional_100",
    REPO / "remaining_376",
]

TEST_ROOT = REPO / "pilot_90" / "test"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Sample:
    key: str
    group: str
    source_image: Path
    image_name: str
    annotations: list[dict]
    rec_rows: list[tuple[Path, str]]
    image_sha256: str
    split: str = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_component(text: str) -> str:
    allowed = []
    for character in text:
        if character.isalnum() or character in {"-", "_"}:
            allowed.append(character)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_")


def relative_group(label_file: Path) -> str:
    relative = label_file.parent.relative_to(REPO)
    parts = [part for part in relative.parts if part.lower() != "images"]
    return "__".join(safe_component(part) for part in parts)


def resolve_image(label_file: Path, stored_path: str) -> Path:
    name = Path(stored_path.replace("\\", "/")).name
    candidates = [
        label_file.parent / name,
        label_file.parent / "images" / name,
        label_file.parent.parent / "images" / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find image {name!r} referenced by {label_file}"
    )


def read_recognition_exports(image_folder: Path) -> dict[str, tuple[Path, str]]:
    rec_file = image_folder / "rec_gt.txt"
    crop_folder = image_folder / "crop_img"

    if not rec_file.is_file():
        raise FileNotFoundError(f"Missing recognition labels: {rec_file}")
    if not crop_folder.is_dir():
        raise FileNotFoundError(f"Missing recognition crops: {crop_folder}")

    rows: dict[str, tuple[Path, str]] = {}
    for line_number, line in enumerate(
        rec_file.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid rec_gt row: {rec_file}, line {line_number}")
        path_text, transcription = parts
        crop_name = Path(path_text.replace("\\", "/")).name
        crop_path = crop_folder / crop_name
        if not crop_path.is_file():
            raise FileNotFoundError(f"Missing crop: {crop_path}")
        if not transcription:
            raise ValueError(f"Empty transcription: {rec_file}, line {line_number}")
        if crop_name in rows:
            raise ValueError(f"Duplicate crop reference in {rec_file}: {crop_name}")
        rows[crop_name] = (crop_path, transcription)
    return rows


def source_prefix_from_crop(crop_name: str) -> str:
    marker = "_crop_"
    if marker in crop_name:
        return crop_name.split(marker, 1)[0]
    return Path(crop_name).stem


def discover_samples() -> list[Sample]:
    samples: list[Sample] = []
    seen_keys: set[str] = set()

    for root in ANNOTATION_ROOTS:
        if not root.exists():
            continue

        for label_file in sorted(root.rglob("Label.txt")):
            if label_file.stat().st_size == 0:
                continue

            group = relative_group(label_file)
            rec_exports = read_recognition_exports(label_file.parent)

            for line_number, line in enumerate(
                label_file.read_text(encoding="utf-8-sig").splitlines(), start=1
            ):
                if not line.strip():
                    continue

                parts = line.split("\t", 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid Label.txt row: {label_file}, line {line_number}")

                stored_path, annotation_text = parts
                source_image = resolve_image(label_file, stored_path)
                annotations = json.loads(annotation_text)
                if not isinstance(annotations, list):
                    raise ValueError(f"Annotation is not a list: {label_file}, line {line_number}")

                image_stem = source_image.stem
                matching_rec_rows = [
                    value
                    for crop_name, value in rec_exports.items()
                    if source_prefix_from_crop(crop_name) == image_stem
                ]

                expected_regions = len(annotations)
                if len(matching_rec_rows) != expected_regions:
                    raise RuntimeError(
                        f"Region/crop mismatch for {source_image}: "
                        f"Label regions={expected_regions}, recognition crops={len(matching_rec_rows)}"
                    )

                key = f"{group}__{source_image.name}"
                if key in seen_keys:
                    raise RuntimeError(f"Duplicate discovered sample key: {key}")
                seen_keys.add(key)

                samples.append(
                    Sample(
                        key=key,
                        group=group,
                        source_image=source_image,
                        image_name=source_image.name,
                        annotations=annotations,
                        rec_rows=sorted(matching_rec_rows, key=lambda item: item[0].name),
                        image_sha256=sha256_file(source_image),
                    )
                )

    return samples


def load_test_identities() -> tuple[set[str], set[str]]:
    names: set[str] = set()
    hashes: set[str] = set()
    if not TEST_ROOT.exists():
        return names, hashes

    for path in TEST_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            names.add(path.name.lower())
            hashes.add(sha256_file(path))
    return names, hashes


def remove_duplicates_and_test_leakage(samples: list[Sample]) -> list[Sample]:
    test_names, test_hashes = load_test_identities()
    kept: list[Sample] = []
    seen_hashes: dict[str, Sample] = {}

    for sample in samples:
        if sample.image_name.lower() in test_names or sample.image_sha256 in test_hashes:
            raise RuntimeError(
                f"Sealed test leakage detected in annotated development data: {sample.source_image}"
            )

        if sample.image_sha256 in seen_hashes:
            first = seen_hashes[sample.image_sha256]
            raise RuntimeError(
                "Duplicate image content detected:\n"
                f"  {first.source_image}\n"
                f"  {sample.source_image}"
            )

        seen_hashes[sample.image_sha256] = sample
        kept.append(sample)

    return kept


def assign_grouped_split(samples: list[Sample]) -> None:
    rng = random.Random(RANDOM_SEED)
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group].append(sample)

    target_validation = round(len(samples) * VALIDATION_FRACTION)

    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for group, items in groups.items():
        exact = len(items) * VALIDATION_FRACTION
        base = int(exact)
        if len(items) >= 5 and base == 0:
            base = 1
        base = min(base, max(0, len(items) - 1))
        allocations[group] = base
        remainders.append((exact - int(exact), group))

    current = sum(allocations.values())
    remainders.sort(reverse=True)

    while current < target_validation:
        changed = False
        for _, group in remainders:
            if allocations[group] < len(groups[group]) - 1:
                allocations[group] += 1
                current += 1
                changed = True
                if current == target_validation:
                    break
        if not changed:
            break

    while current > target_validation:
        changed = False
        for _, group in reversed(remainders):
            minimum = 1 if len(groups[group]) >= 5 else 0
            if allocations[group] > minimum:
                allocations[group] -= 1
                current -= 1
                changed = True
                if current == target_validation:
                    break
        if not changed:
            break

    for group, items in groups.items():
        shuffled = sorted(items, key=lambda item: item.key)
        rng.shuffle(shuffled)
        validation_count = allocations[group]
        for index, sample in enumerate(shuffled):
            sample.split = "validation" if index < validation_count else "train"


def validate_polygon_points(sample: Sample) -> None:
    from PIL import Image

    with Image.open(sample.source_image) as image:
        width, height = image.size

    for region_number, region in enumerate(sample.annotations, start=1):
        points = region.get("points", [])
        if len(points) != 4:
            raise ValueError(
                f"{sample.source_image}, region {region_number}: expected 4 points"
            )
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(
                    f"{sample.source_image}, region {region_number}: invalid point {point!r}"
                )
            x, y = point
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"{sample.source_image}, region {region_number}: "
                    f"point {point!r} outside {width}x{height}"
                )


def copy_dataset(samples: list[Sample]) -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    detector_labels: dict[str, list[str]] = {"train": [], "validation": []}
    recognizer_labels: dict[str, list[str]] = {"train": [], "validation": []}
    manifest_rows: list[dict[str, str | int]] = []

    for sample in sorted(samples, key=lambda item: (item.split, item.key)):
        validate_polygon_points(sample)

        detector_image_dir = OUTPUT / "detector" / sample.split / "images"
        recognition_crop_dir = OUTPUT / "recognizer" / sample.split / "crop_img"
        detector_image_dir.mkdir(parents=True, exist_ok=True)
        recognition_crop_dir.mkdir(parents=True, exist_ok=True)

        prefixed_image_name = f"{sample.group}__{sample.image_name}"
        destination_image = detector_image_dir / prefixed_image_name
        shutil.copy2(sample.source_image, destination_image)

        detector_labels[sample.split].append(
            "images/"
            + prefixed_image_name
            + "\t"
            + json.dumps(sample.annotations, ensure_ascii=False)
        )

        for crop_index, (source_crop, transcription) in enumerate(sample.rec_rows):
            crop_suffix = source_crop.suffix.lower()
            crop_name = (
                f"{sample.group}__{sample.source_image.stem}__crop_{crop_index}"
                f"{crop_suffix}"
            )
            destination_crop = recognition_crop_dir / crop_name
            shutil.copy2(source_crop, destination_crop)
            recognizer_labels[sample.split].append(
                "crop_img/" + crop_name + "\t" + transcription
            )

        manifest_rows.append(
            {
                "sample_key": sample.key,
                "split": sample.split,
                "group": sample.group,
                "source_image": str(sample.source_image),
                "compiled_image": prefixed_image_name,
                "regions": len(sample.annotations),
                "sha256": sample.image_sha256,
            }
        )

    for split in ("train", "validation"):
        detector_root = OUTPUT / "detector" / split
        recognizer_root = OUTPUT / "recognizer" / split
        detector_root.mkdir(parents=True, exist_ok=True)
        recognizer_root.mkdir(parents=True, exist_ok=True)

        (detector_root / "Label.txt").write_text(
            "\n".join(detector_labels[split]) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (recognizer_root / "rec_gt.txt").write_text(
            "\n".join(recognizer_labels[split]) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    with (OUTPUT / "split_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    (OUTPUT / "split_settings.json").write_text(
        json.dumps(
            {
                "random_seed": RANDOM_SEED,
                "validation_fraction": VALIDATION_FRACTION,
                "test_split": "pilot_90/test remains sealed and was not compiled",
                "source_image_count": len(samples),
                "train_image_count": sum(item.split == "train" for item in samples),
                "validation_image_count": sum(
                    item.split == "validation" for item in samples
                ),
                "train_crop_count": len(recognizer_labels["train"]),
                "validation_crop_count": len(recognizer_labels["validation"]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def final_check(samples: list[Sample]) -> None:
    train_images = [sample for sample in samples if sample.split == "train"]
    validation_images = [sample for sample in samples if sample.split == "validation"]

    train_keys = {sample.key for sample in train_images}
    validation_keys = {sample.key for sample in validation_images}
    if train_keys & validation_keys:
        raise RuntimeError("Train/validation sample overlap detected")

    for split, items in (("train", train_images), ("validation", validation_images)):
        detector_images = list((OUTPUT / "detector" / split / "images").iterdir())
        detector_rows = [
            line
            for line in (OUTPUT / "detector" / split / "Label.txt").read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if line.strip()
        ]
        rec_crops = list((OUTPUT / "recognizer" / split / "crop_img").iterdir())
        rec_rows = [
            line
            for line in (OUTPUT / "recognizer" / split / "rec_gt.txt").read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if line.strip()
        ]

        if len(detector_images) != len(items) or len(detector_rows) != len(items):
            raise RuntimeError(f"Detector count mismatch in {split}")
        if len(rec_crops) != len(rec_rows):
            raise RuntimeError(f"Recognizer crop/label mismatch in {split}")

    print()
    print("Final compiled dataset")
    print("Source images:", len(samples))
    print("Training images:", len(train_images))
    print("Validation images:", len(validation_images))
    print(
        "Training recognition crops:",
        sum(len(sample.rec_rows) for sample in train_images),
    )
    print(
        "Validation recognition crops:",
        sum(len(sample.rec_rows) for sample in validation_images),
    )
    print("Sealed test split: unchanged")
    print("Output:", OUTPUT)
    print("Compilation and split validation passed.")


def main() -> None:
    samples = discover_samples()
    if not samples:
        raise RuntimeError("No annotated samples were discovered")

    samples = remove_duplicates_and_test_leakage(samples)
    assign_grouped_split(samples)
    copy_dataset(samples)
    final_check(samples)


if __name__ == "__main__":
    main()
