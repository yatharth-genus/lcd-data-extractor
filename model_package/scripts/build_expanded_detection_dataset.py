from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image


REPO = Path(r"D:\Actual Project\lcd-data-extractor")
OUTPUT_ROOT = REPO / "detection_expanded_240"

TRAIN_ROOT = OUTPUT_ROOT / "train"
TRAIN_IMAGES = TRAIN_ROOT / "images"
TRAIN_LABEL = TRAIN_ROOT / "Label.txt"

VALIDATION_ROOT = OUTPUT_ROOT / "validation"
VALIDATION_IMAGES = VALIDATION_ROOT / "images"
VALIDATION_LABEL = VALIDATION_ROOT / "Label.txt"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def read_label_file(label_file: Path) -> list[tuple[str, list[dict]]]:
    records: list[tuple[str, list[dict]]] = []

    if not label_file.is_file():
        raise FileNotFoundError(f"Missing label file: {label_file}")

    for line_number, line in enumerate(
        label_file.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        parts = line.split("\t", 1)
        if len(parts) != 2:
            raise ValueError(
                f"{label_file}, line {line_number}: expected a tab-separated "
                "image path and JSON annotation"
            )

        image_path, annotation_text = parts

        try:
            annotations = json.loads(annotation_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{label_file}, line {line_number}: invalid JSON: {error}"
            ) from error

        if not isinstance(annotations, list):
            raise ValueError(
                f"{label_file}, line {line_number}: annotation must be a list"
            )

        records.append((image_path.replace("\\", "/"), annotations))

    return records


def find_source_image(source_root: Path, stored_path: str) -> Path:
    normalized = Path(stored_path.replace("\\", "/"))

    candidates = [
        source_root / normalized,
        source_root / "images" / normalized.name,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve image {stored_path!r} below {source_root}"
    )


def clamp_points_to_image(
    annotations: list[dict],
    width: int,
    height: int,
    image_name: str,
) -> tuple[list[dict], int]:
    changed_points = 0

    for region_number, region in enumerate(annotations, start=1):
        if not isinstance(region, dict):
            raise ValueError(
                f"{image_name}, region {region_number}: region must be a dictionary"
            )

        points = region.get("points", [])
        if len(points) != 4:
            raise ValueError(
                f"{image_name}, region {region_number}: expected 4 points, "
                f"found {len(points)}"
            )

        repaired_points: list[list[int]] = []

        for point_number, point in enumerate(points, start=1):
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(
                    f"{image_name}, region {region_number}, point {point_number}: "
                    f"invalid point {point!r}"
                )

            x, y = point
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise ValueError(
                    f"{image_name}, region {region_number}, point {point_number}: "
                    f"coordinates must be numeric, found {point!r}"
                )

            repaired_x = max(0, min(int(round(x)), width - 1))
            repaired_y = max(0, min(int(round(y)), height - 1))

            if repaired_x != x or repaired_y != y:
                changed_points += 1
                print(
                    f"Boundary repair: {image_name}, region {region_number}, "
                    f"point {point_number}: {point!r} -> "
                    f"[{repaired_x}, {repaired_y}]"
                )

            repaired_points.append([repaired_x, repaired_y])

        region["points"] = repaired_points

    return annotations, changed_points


def copy_selected_records(
    source_root: Path,
    label_file: Path,
    destination_images: Path,
    output_records: list[str],
    selected_indexes: set[int] | None = None,
) -> tuple[int, int]:
    records = read_label_file(label_file)
    copied = 0
    repaired_points = 0

    for index, (stored_image_path, annotations) in enumerate(records):
        if selected_indexes is not None and index not in selected_indexes:
            continue

        source_image = find_source_image(source_root, stored_image_path)
        destination_image = destination_images / source_image.name

        if destination_image.exists():
            raise RuntimeError(
                f"Duplicate image filename in expanded split: {source_image.name}"
            )

        with Image.open(source_image) as image:
            width, height = image.size

        annotations, changed = clamp_points_to_image(
            annotations,
            width,
            height,
            source_image.name,
        )
        repaired_points += changed

        destination_images.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, destination_image)

        output_records.append(
            "images/"
            + source_image.name
            + "\t"
            + json.dumps(annotations, ensure_ascii=False)
        )
        copied += 1

    return copied, repaired_points


def validate_split(
    split_name: str,
    split_root: Path,
    label_file: Path,
    expected_images: int,
) -> None:
    records = read_label_file(label_file)
    image_files = [
        path
        for path in (split_root / "images").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    errors: list[str] = []
    region_count = 0

    if len(records) != expected_images:
        errors.append(
            f"Expected {expected_images} label rows, found {len(records)}"
        )

    if len(image_files) != expected_images:
        errors.append(
            f"Expected {expected_images} image files, found {len(image_files)}"
        )

    label_names: set[str] = set()

    for stored_image_path, annotations in records:
        image_path = split_root / Path(stored_image_path)
        image_name = image_path.name

        if image_name in label_names:
            errors.append(f"Duplicate label image name: {image_name}")
        label_names.add(image_name)

        if not image_path.is_file():
            errors.append(f"Missing image: {image_path}")
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        for region_number, region in enumerate(annotations, start=1):
            region_count += 1
            points = region.get("points", [])

            if len(points) != 4:
                errors.append(
                    f"{image_name}, region {region_number}: "
                    f"expected 4 points, found {len(points)}"
                )
                continue

            for point in points:
                if not isinstance(point, list) or len(point) != 2:
                    errors.append(
                        f"{image_name}, region {region_number}: invalid point {point!r}"
                    )
                    continue

                x, y = point
                if not (0 <= x < width and 0 <= y < height):
                    errors.append(
                        f"{image_name}, region {region_number}: point {point!r} "
                        f"outside {width}x{height}"
                    )

    image_names = {path.name for path in image_files}
    missing_in_labels = sorted(image_names - label_names)
    missing_on_disk = sorted(label_names - image_names)

    for name in missing_in_labels:
        errors.append(f"Image has no label row: {name}")
    for name in missing_on_disk:
        errors.append(f"Label row has no image file: {name}")

    print()
    print(f"Split: {split_name}")
    print(f"Images: {len(image_files)}")
    print(f"Label rows: {len(records)}")
    print(f"Text regions: {region_count}")
    print(f"Errors: {len(errors)}")

    for error in errors[:30]:
        print("  " + error)

    if errors:
        raise RuntimeError(f"{split_name} validation failed")


def main() -> None:
    required_label_files = [
        REPO / "pilot_90" / "train" / "images" / "Label.txt",
        REPO / "pilot_90" / "validation" / "images" / "Label.txt",
    ]

    for expansion_name in ("next_100", "additional_100"):
        for batch_number in range(1, 6):
            required_label_files.append(
                REPO
                / expansion_name
                / f"batch_{batch_number:02d}"
                / "images"
                / "Label.txt"
            )

    missing = [path for path in required_label_files if not path.is_file()]
    if missing:
        print("Required Label.txt files are missing:")
        for path in missing:
            print("  " + str(path))
        raise SystemExit(1)

    if OUTPUT_ROOT.exists():
        print(f"Removing previous output: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
    VALIDATION_IMAGES.mkdir(parents=True, exist_ok=True)

    training_records: list[str] = []
    validation_records: list[str] = []
    total_boundary_repairs = 0

    original_train_count, repairs = copy_selected_records(
        source_root=REPO / "pilot_90" / "train",
        label_file=REPO / "pilot_90" / "train" / "images" / "Label.txt",
        destination_images=TRAIN_IMAGES,
        output_records=training_records,
    )
    total_boundary_repairs += repairs

    original_validation_count, repairs = copy_selected_records(
        source_root=REPO / "pilot_90" / "validation",
        label_file=(
            REPO / "pilot_90" / "validation" / "images" / "Label.txt"
        ),
        destination_images=VALIDATION_IMAGES,
        output_records=validation_records,
    )
    total_boundary_repairs += repairs

    expansion_train_count = 0
    expansion_validation_count = 0

    train_indexes = set(range(18))
    validation_indexes = {18, 19}

    for expansion_name in ("next_100", "additional_100"):
        expansion_root = REPO / expansion_name

        for batch_number in range(1, 6):
            batch_root = expansion_root / f"batch_{batch_number:02d}"
            batch_label = batch_root / "images" / "Label.txt"

            copied, repairs = copy_selected_records(
                source_root=batch_root,
                label_file=batch_label,
                destination_images=TRAIN_IMAGES,
                output_records=training_records,
                selected_indexes=train_indexes,
            )
            expansion_train_count += copied
            total_boundary_repairs += repairs

            copied, repairs = copy_selected_records(
                source_root=batch_root,
                label_file=batch_label,
                destination_images=VALIDATION_IMAGES,
                output_records=validation_records,
                selected_indexes=validation_indexes,
            )
            expansion_validation_count += copied
            total_boundary_repairs += repairs

    TRAIN_LABEL.write_text(
        "\n".join(training_records) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    VALIDATION_LABEL.write_text(
        "\n".join(validation_records) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print()
    print("Dataset creation summary")
    print(f"Original training images: {original_train_count}")
    print(f"Expansion training images: {expansion_train_count}")
    print(f"Total training images: {len(training_records)}")
    print(f"Original validation images: {original_validation_count}")
    print(f"Expansion validation images: {expansion_validation_count}")
    print(f"Total validation images: {len(validation_records)}")
    print(f"Boundary coordinates repaired: {total_boundary_repairs}")

    if len(training_records) != 240:
        raise RuntimeError(
            f"Expected 240 training images, found {len(training_records)}"
        )

    if len(validation_records) != 28:
        raise RuntimeError(
            f"Expected 28 validation images, found {len(validation_records)}"
        )

    validate_split(
        split_name="training",
        split_root=TRAIN_ROOT,
        label_file=TRAIN_LABEL,
        expected_images=240,
    )
    validate_split(
        split_name="validation",
        split_root=VALIDATION_ROOT,
        label_file=VALIDATION_LABEL,
        expected_images=28,
    )

    print()
    print("Expanded detection dataset created successfully.")
    print(f"Output: {OUTPUT_ROOT}")
    print("The original pilot test split was not read or modified.")


if __name__ == "__main__":
    main()

