from contextlib import asynccontextmanager
from pathlib import Path
import os
import json
import re
from time import perf_counter
from typing import Literal
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .icon_detector import IconDetector
from .pipeline_registry import DEFAULT_PIPELINE, public_pipeline_catalog, resolve_pipeline
from .v6_ocr_engine import V6OcrEngine


API_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = API_ROOT.parent
PROJECT_ROOT = PACKAGE_ROOT
ICON_PACKAGE = PACKAGE_ROOT / "models" / "icon_detector"
HYBRID_PACKAGE = PACKAGE_ROOT / "models" / "general_v3"
MODEL4_PACKAGE = PACKAGE_ROOT / "models" / "experimental_v6"
RESULTS_FOLDER = PACKAGE_ROOT / "results" / "rendered"
RESULTS_ROOT = PACKAGE_ROOT / "results"
PPOCRV6_ROOT = Path(os.environ.get("PPOCRV6_ROOT", str(PACKAGE_ROOT / "runtime" / "PaddleOCR-3.7")))
PPOCRV6_PYTHON = Path(os.environ.get("PPOCRV6_PYTHON", str(PACKAGE_ROOT / ".venv-v6" / "Scripts" / "python.exe")))
PPOCRV6_DETECTOR_MODEL_PATH = MODEL4_PACKAGE / "models" / "detector" / "inference"
PPOCRV6_RECOGNIZER_MODEL_PATH = MODEL4_PACKAGE / "models" / "recognizer" / "inference"
PPOCRV6_CHARACTER_DICTIONARY_PATH = MODEL4_PACKAGE / "models" / "recognizer" / "character_dict.txt"


ICON_MODEL_PATH = ICON_PACKAGE / "model" / "saved_model.xml"
ICON_LABELS_PATH = ICON_PACKAGE / "dm.json"
ICON_TRANSFORMS_PATH = ICON_PACKAGE / "model_meta" / "transforms.yaml"
ICON_TRAIN_CONFIG_PATH = ICON_PACKAGE / "model_meta" / "train.yaml"
OCR_DETECTOR_MODEL_PATH = HYBRID_PACKAGE / "models" / "detector" / "inference"
OCR_RECOGNIZER_MODEL_PATH = HYBRID_PACKAGE / "models" / "recognizer" / "inference"
OCR_CHARACTER_DICTIONARY_PATH = HYBRID_PACKAGE / "models" / "recognizer" / "character_dict.txt"

ICON_THRESHOLD = 0.38
ICON_MASK_PADDING = 2
ICON_MASK_FILL_VALUE = 0


class ModelStatus(BaseModel):
    loaded: bool
    model_name: str
    runtime: str
    device: str


class HealthCheckResponse(BaseModel):
    status: str
    icon_detector: ModelStatus
    ocr_detector: ModelStatus
    ocr_recognizer: ModelStatus


class ImageInformation(BaseModel):
    filename: str
    width: int
    height: int


class IconOccurrence(BaseModel):
    id: str
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: list[int]
    detection_order: int = Field(ge=1)
    masked_before_ocr: bool


class TextOccurrence(BaseModel):
    id: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    polygon: list[list[int]]
    reading_order: int = Field(ge=1)


class ProcessingTiming(BaseModel):
    icon_detection_ms: float
    icon_masking_ms: float
    ocr_detection_and_recognition_ms: float
    rendering_ms: float
    total_ms: float


class RenderedResult(BaseModel):
    available: bool
    url: str | None


class SavedResults(BaseModel):
    run_name: str
    masked_url: str
    rendered_url: str
    prediction_url: str


class DevelopmentPredictionResponse(BaseModel):
    request_id: str
    mode: str
    status: str
    image: ImageInformation
    text_occurrences: list[TextOccurrence]
    icon_occurrences: list[IconOccurrence]
    timing: ProcessingTiming
    rendered_result: RenderedResult
    saved_results: SavedResults
    warnings: list[str]
    requested_meter_family: str = DEFAULT_PIPELINE
    resolved_pipeline: str = DEFAULT_PIPELINE
    fallback_used: bool = False
    fallback_reason: str | None = None
    pipeline: dict = Field(default_factory=dict)


def mask_detected_icons(image, icon_occurrences):
    started = perf_counter()
    masked = image.copy()
    height, width = masked.shape[:2]

    for occurrence in icon_occurrences:
        x1, y1, x2, y2 = occurrence["box"]
        x1 = max(0, x1 - ICON_MASK_PADDING)
        y1 = max(0, y1 - ICON_MASK_PADDING)
        x2 = min(width - 1, x2 + ICON_MASK_PADDING)
        y2 = min(height - 1, y2 + ICON_MASK_PADDING)
        cv2.rectangle(
            masked,
            (x1, y1),
            (x2, y2),
            (ICON_MASK_FILL_VALUE,) * 3,
            thickness=-1,
        )

    return masked, (perf_counter() - started) * 1000


def render_combined_result(original, icons, texts):
    started = perf_counter()
    rendered = original.copy()

    for occurrence in icons:
        x1, y1, x2, y2 = occurrence["box"]
        label = (
            f"{occurrence['id']} {occurrence['class_name']} "
            f"{occurrence['confidence']:.3f}"
        )
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(
            rendered, label, (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1, cv2.LINE_AA,
        )

    for occurrence in texts:
        polygon = np.asarray(occurrence["polygon"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(rendered, [polygon], True, (0, 200, 0), 2)
        x, y = occurrence["polygon"][0]
        label = (
            f"{occurrence['id']} {occurrence['value']} "
            f"{occurrence['confidence']:.3f}"
        )
        cv2.putText(
            rendered, label, (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 0), 1, cv2.LINE_AA,
        )

    return rendered, (perf_counter() - started) * 1000




def strengthen_v6_icon_masking(image, masked_image, icon_occurrences):
    """Apply a second, padded LCD-background fill for PP-OCRv6.

    The normal masking stage remains unchanged. This additional pass is used
    only by experimental_v6 because PP-OCRv6 is more sensitive to remaining
    icon edges and segments.
    """
    strengthened = masked_image.copy()
    image_height, image_width = image.shape[:2]

    for icon in icon_occurrences:
        box = icon.get("box")
        if not box or len(box) != 4:
            continue

        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)

        pad_x = max(5, int(round(box_width * 0.25)))
        pad_y = max(5, int(round(box_height * 0.30)))

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(image_width - 1, x2 + pad_x)
        y2 = min(image_height - 1, y2 + pad_y)

        sample_pad = max(4, min(16, int(round(max(box_width, box_height) * 0.15))))
        sx1 = max(0, x1 - sample_pad)
        sy1 = max(0, y1 - sample_pad)
        sx2 = min(image_width - 1, x2 + sample_pad)
        sy2 = min(image_height - 1, y2 + sample_pad)

        sample = image[sy1:sy2 + 1, sx1:sx2 + 1]
        if sample.size == 0:
            fill_color = (0, 0, 0)
        else:
            border_pixels = []
            if y1 > sy1:
                border_pixels.append(sample[: max(1, y1 - sy1), :, :].reshape(-1, 3))
            if sy2 > y2:
                border_pixels.append(sample[-max(1, sy2 - y2) :, :, :].reshape(-1, 3))
            if x1 > sx1:
                border_pixels.append(sample[:, : max(1, x1 - sx1), :].reshape(-1, 3))
            if sx2 > x2:
                border_pixels.append(sample[:, -max(1, sx2 - x2) :, :].reshape(-1, 3))

            if border_pixels:
                pixels = np.concatenate(border_pixels, axis=0)
            else:
                pixels = sample.reshape(-1, 3)

            fill_color = tuple(int(value) for value in np.median(pixels, axis=0))

        cv2.rectangle(
            strengthened,
            (x1, y1),
            (x2, y2),
            fill_color,
            thickness=-1,
        )

    return strengthened




def sanitize_result_name(value: str, fallback: str = "result") -> str:
    value = value.replace("\\", "/").strip("/")
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return value or fallback


def allocate_unique_directory(parent: Path, requested_name: str) -> Path:
    base_name = sanitize_result_name(requested_name)
    candidate = parent / base_name
    number = 2
    while candidate.exists():
        candidate = parent / f"{base_name}_{number:03d}"
        number += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def derive_result_location(uploaded_filename: str | None):
    normalized = (uploaded_filename or "uploaded_image").replace("\\", "/").strip("/")
    parts = [part for part in normalized.split("/") if part]
    filename = parts[-1] if parts else "uploaded_image"
    stem = sanitize_result_name(Path(filename).stem, "image")

    if len(parts) > 1:
        folder_name = sanitize_result_name(parts[0], "folder_batch")
        run_directory = RESULTS_ROOT / "folder" / folder_name
        run_directory.mkdir(parents=True, exist_ok=True)
        masked_directory = run_directory / "masked"
        rendered_directory = run_directory / "rendered"
        prediction_directory = run_directory / "predictions"
        for directory in (masked_directory, rendered_directory, prediction_directory):
            directory.mkdir(parents=True, exist_ok=True)
        return {
            "mode": "folder",
            "run_name": folder_name,
            "stem": stem,
            "masked_path": masked_directory / f"{stem}_masked.png",
            "rendered_path": rendered_directory / f"{stem}_rendered.png",
            "prediction_path": prediction_directory / f"{stem}_prediction.json",
        }

    item_directory = allocate_unique_directory(RESULTS_ROOT / "single", stem)
    item_stem = item_directory.name
    return {
        "mode": "single",
        "run_name": item_stem,
        "stem": stem,
        "masked_path": item_directory / f"{stem}_masked.png",
        "rendered_path": item_directory / f"{stem}_rendered.png",
        "prediction_path": item_directory / f"{stem}_prediction.json",
    }


def result_url(path: Path) -> str:
    relative = path.resolve().relative_to(RESULTS_ROOT.resolve())
    return "/results/" + "/".join(relative.parts)


def create_general_v3_engine():
    """Create the historical general_v3 engine only when explicitly requested."""
    try:
        from .ocr_engine import OcrEngine
    except ModuleNotFoundError as error:
        if error.name in {"paddleocr", "paddle"}:
            raise RuntimeError(
                "The general_v3 pipeline is unavailable because its PaddleOCR "
                "runtime is not installed in the API environment. Start or select "
                "experimental_v6, which uses the isolated .venv-v6 worker."
            ) from error
        raise

    return OcrEngine(
        detector_model_path=OCR_DETECTOR_MODEL_PATH,
        recognizer_model_path=OCR_RECOGNIZER_MODEL_PATH,
        character_dictionary_path=OCR_CHARACTER_DICTIONARY_PATH,
    )


def get_pipeline_engine(app, routing):
    engine_name = routing["definition"].engine_state_name
    engine = getattr(app.state, engine_name, None)

    if engine_name == "ocr_engine_general_v3" and engine is None:
        with app.state.general_v3_initialization_lock:
            engine = getattr(app.state, engine_name, None)
            if engine is None:
                engine = create_general_v3_engine()
                setattr(app.state, engine_name, engine)

    if engine is None:
        raise RuntimeError(
            f"OCR engine '{engine_name}' is not available for pipeline "
            f"'{routing['resolved_pipeline']}'."
        )

    return engine


def run_complete_pipeline(app, image, meter_family=DEFAULT_PIPELINE):
    routing = resolve_pipeline(meter_family)
    engine = get_pipeline_engine(app, routing)
    raw_icons, icon_ms = app.state.icon_detector.predict(image)
    icons = [
        {
            "id": f"icon_{index:03d}",
            "class_id": item["class_id"],
            "class_name": item["class_name"],
            "confidence": item["confidence"],
            "box": item["box"],
            "detection_order": index,
            "masked_before_ocr": True,
        }
        for index, item in enumerate(raw_icons, start=1)
    ]

    masked, masking_ms = mask_detected_icons(image, icons)
    if routing["resolved_pipeline"] == "experimental_v6":
        masked = strengthen_v6_icon_masking(image, masked, icons)
        debug_mask_path = PACKAGE_ROOT / "results" / "debug" / "debug_last_masked_v6.png"
        debug_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_mask_path), masked)
    raw_texts, ocr_ms = engine.predict(masked)
    texts = [
        {
            "id": f"text_{index:03d}",
            "value": item["value"],
            "confidence": item["confidence"],
            "polygon": item["polygon"],
            "reading_order": index,
        }
        for index, item in enumerate(raw_texts, start=1)
    ]

    return {
        "masked_image": masked,
        "icons": icons,
        "texts": texts,
        "icon_ms": icon_ms,
        "masking_ms": masking_ms,
        "ocr_ms": ocr_ms,
        "routing": routing,
    }


@asynccontextmanager
async def application_lifespan(app: FastAPI):
    required_files = [
        ICON_MODEL_PATH,
        ICON_LABELS_PATH,
        ICON_TRANSFORMS_PATH,
        ICON_TRAIN_CONFIG_PATH,
        PPOCRV6_PYTHON,
        PPOCRV6_CHARACTER_DICTIONARY_PATH,
        API_ROOT / "v6_ocr_worker.py",
    ]
    required_folders = [
        PPOCRV6_ROOT,
        PPOCRV6_DETECTOR_MODEL_PATH,
        PPOCRV6_RECOGNIZER_MODEL_PATH,
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    missing += [str(path) for path in required_folders if not path.is_dir()]
    if missing:
        raise RuntimeError("Required model paths are missing: " + "; ".join(missing))

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    app.state.icon_detector = IconDetector(
        model_path=ICON_MODEL_PATH,
        labels_path=ICON_LABELS_PATH,
        transforms_path=ICON_TRANSFORMS_PATH,
        train_config_path=ICON_TRAIN_CONFIG_PATH,
        threshold=ICON_THRESHOLD,
        device="CPU",
    )
    # general_v3 is a historical fallback. Do not import or initialize its
    # PaddleOCR runtime unless that pipeline is explicitly requested.
    import threading
    app.state.general_v3_initialization_lock = threading.Lock()
    app.state.ocr_engine_general_v3 = None
    app.state.ocr_engine_experimental_v6 = V6OcrEngine(
        python_executable=PPOCRV6_PYTHON,
        paddleocr_root=PPOCRV6_ROOT,
        detector_model_path=PPOCRV6_DETECTOR_MODEL_PATH,
        recognizer_model_path=PPOCRV6_RECOGNIZER_MODEL_PATH,
        character_dictionary_path=PPOCRV6_CHARACTER_DICTIONARY_PATH,
        worker_program=API_ROOT / "v6_ocr_worker.py",
    )
    app.state.models_loaded = True
    yield
    app.state.models_loaded = False
    app.state.icon_detector = None
    if getattr(app.state, "ocr_engine_experimental_v6", None):
        app.state.ocr_engine_experimental_v6.close()
    app.state.ocr_engine_general_v3 = None
    app.state.ocr_engine_experimental_v6 = None


app = FastAPI(
    title="LCD Understanding API",
    description=(
        "Development API for extracting duplicate-preserving text and "
        "icon occurrences from utility-meter LCD images."
    ),
    version="0.5.0",
    lifespan=application_lifespan,
    docs_url=None,
)


@app.get("/", include_in_schema=False)
def redirect_to_visual_tester():
    return RedirectResponse(url="/test", status_code=307)


@app.get("/test", include_in_schema=False)
def visual_model_tester():
    page = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LCD Model Visual Tester</title>
    <style>
        :root {
            color-scheme: light;
            font-family: Inter, "Segoe UI", Arial, sans-serif;
            background: #f4f7fb;
            color: #152238;
        }
        * { box-sizing: border-box; }
        body { margin: 0; min-height: 100vh; background: #f4f7fb; }
        header {
            display: flex; align-items: center; justify-content: space-between;
            gap: 20px; padding: 18px 28px; color: white;
            background: linear-gradient(120deg, #0f4c81, #0f6b78);
            box-shadow: 0 2px 12px rgba(16, 41, 70, 0.18);
        }
        header h1 { margin: 0; font-size: 23px; }
        header p { margin: 4px 0 0; opacity: 0.88; font-size: 13px; }
        nav a {
            display: inline-block; padding: 9px 13px; border: 1px solid rgba(255,255,255,.45);
            border-radius: 8px; color: white; text-decoration: none; font-size: 13px;
        }
        main { max-width: 1500px; margin: 0 auto; padding: 24px; }
        .toolbar, .panel {
            background: white; border: 1px solid #dfe7f1; border-radius: 12px;
            box-shadow: 0 5px 18px rgba(25, 51, 85, 0.07);
        }
        .toolbar {
            display: flex; flex-wrap: wrap; align-items: center; gap: 12px; padding: 16px;
            margin-bottom: 18px;
        }
        select { padding: 10px; border: 1px solid #c9d6e5; border-radius: 8px; background: white; }
        input[type=file] {
            flex: 1; min-width: 280px; padding: 10px; border: 1px solid #c9d6e5;
            border-radius: 8px; background: #fbfdff;
        }
        button {
            border: 0; border-radius: 8px; padding: 11px 18px; cursor: pointer;
            color: white; background: #0f6b78; font-weight: 650;
        }
        button:hover { background: #0c5964; }
        button:disabled { background: #8fa9ad; cursor: wait; }
        #status { margin-left: auto; font-size: 13px; color: #506176; }
        .image-grid {
            display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px;
        }
        .panel { overflow: hidden; }
        .panel-title {
            padding: 12px 16px; font-weight: 700; color: #17365d;
            border-bottom: 1px solid #e4ebf3; background: #f9fbfd;
        }
        .image-stage {
            min-height: 360px; height: 48vh; display: flex; align-items: center;
            justify-content: center; padding: 14px; background: white;
        }
        .image-stage img { max-width: 100%; max-height: 100%; object-fit: contain; }
        .placeholder { color: #8090a4; text-align: center; line-height: 1.5; }
        .data-panel { margin-top: 18px; padding: 18px; }
        .data-panel h2 { margin: 0 0 14px; font-size: 19px; color: #17365d; }
        .summary-grid {
            display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px; margin-bottom: 17px;
        }
        .summary-card { background: #f4f8fc; border-radius: 9px; padding: 12px; }
        .summary-card strong { display: block; font-size: 20px; color: #0f6b78; }
        .summary-card span { color: #647489; font-size: 12px; }
        .results-grid { display: grid; grid-template-columns: 1.35fr 1fr; gap: 18px; }
        .result-section h3 { margin: 0 0 10px; color: #17365d; }
        .occurrence-list { display: grid; gap: 8px; }
        .occurrence {
            display: grid; grid-template-columns: 75px minmax(80px, 1fr) 100px;
            gap: 10px; align-items: center; padding: 10px 12px;
            background: white; border: 1px solid #e0e7ef; border-radius: 8px;
        }
        .occurrence .id { font-family: Consolas, monospace; color: #66758a; font-size: 12px; }
        .occurrence .value { font-weight: 700; overflow-wrap: anywhere; }
        .occurrence .confidence { text-align: right; color: #0f6b78; font-variant-numeric: tabular-nums; }
        .empty { color: #7a899c; padding: 12px; border: 1px dashed #cbd6e3; border-radius: 8px; }
        .timing { margin-top: 18px; padding-top: 14px; border-top: 1px solid #e2e9f1; }
        .timing pre {
            margin: 8px 0 0; padding: 12px; overflow: auto; border-radius: 8px;
            background: #102238; color: #e8f1fa; font-size: 12px;
        }
        .error { color: #b42318 !important; }
        @media (max-width: 900px) {
            .image-grid, .results-grid { grid-template-columns: 1fr; }
            .summary-grid { grid-template-columns: 1fr; }
            #status { width: 100%; margin-left: 0; }
            .image-stage { height: 380px; }
        }
    </style>
</head>
<body>
<header>
    <div>
        <h1>LCD Model Visual Tester</h1>
        <p>Icon detector + icon masking + selectable OCR pipeline</p>
    </div>
    <nav><a href="/docs/" target="_blank">Open API Docs</a></nav>
</header>
<main>
    <section class="toolbar">
        <label for="meterFamily"><strong>Meter family</strong></label>
        <select id="meterFamily" data-default="__DEFAULT_PIPELINE__">
            <option value="general_v3">General v3</option>
            <option value="experimental_v6">Experimental v6</option>
        </select>
        <input id="imageInput" type="file" accept="image/png,image/jpeg,image/bmp,image/webp">
        <button id="runButton" type="button">Run Prediction</button>
        <span id="status">Choose an LCD image to begin.</span>
    </section>

    <section class="image-grid">
        <article class="panel">
            <div class="panel-title">Original image</div>
            <div class="image-stage">
                <div id="originalPlaceholder" class="placeholder">The selected image will appear here.</div>
                <img id="originalImage" alt="Original uploaded LCD" hidden>
            </div>
        </article>
        <article class="panel">
            <div class="panel-title">Final rendered prediction</div>
            <div class="image-stage">
                <div id="renderedPlaceholder" class="placeholder">The combined icon and OCR visualization will appear here.</div>
                <img id="renderedImage" alt="Rendered LCD prediction" hidden>
            </div>
        </article>
    </section>

    <section class="panel data-panel">
        <h2>Extracted data</h2>
        <div class="summary-grid">
            <div class="summary-card"><strong id="textCount">0</strong><span>Text occurrences</span></div>
            <div class="summary-card"><strong id="iconCount">0</strong><span>Icon occurrences</span></div>
            <div class="summary-card"><strong id="totalTime">0 ms</strong><span>Total processing time</span></div>
        </div>
        <div class="results-grid">
            <div class="result-section">
                <h3>Recognized text</h3>
                <div id="textResults" class="occurrence-list"><div class="empty">No prediction has been run.</div></div>
            </div>
            <div class="result-section">
                <h3>Detected icons</h3>
                <div id="iconResults" class="occurrence-list"><div class="empty">No prediction has been run.</div></div>
            </div>
        </div>
        <div class="timing">
            <strong>Detailed timing</strong>
            <pre id="timingResults">Waiting for a prediction...</pre>
        </div>
    </section>
</main>
<script>
    const imageInput = document.getElementById("imageInput");
    const runButton = document.getElementById("runButton");
    const statusNode = document.getElementById("status");
    const originalImage = document.getElementById("originalImage");
    const renderedImage = document.getElementById("renderedImage");
    const originalPlaceholder = document.getElementById("originalPlaceholder");
    const renderedPlaceholder = document.getElementById("renderedPlaceholder");

    let originalObjectUrl = null;
    document.getElementById("meterFamily").value = document.getElementById("meterFamily").dataset.default;

    function setStatus(message, isError = false) {
        statusNode.textContent = message;
        statusNode.classList.toggle("error", isError);
    }

    function showOriginal(file) {
        if (originalObjectUrl) URL.revokeObjectURL(originalObjectUrl);
        originalObjectUrl = URL.createObjectURL(file);
        originalImage.src = originalObjectUrl;
        originalImage.hidden = false;
        originalPlaceholder.hidden = true;
    }

    function occurrenceRow(id, value, confidence) {
        const row = document.createElement("div");
        row.className = "occurrence";
        const idNode = document.createElement("span");
        idNode.className = "id";
        idNode.textContent = id;
        const valueNode = document.createElement("span");
        valueNode.className = "value";
        valueNode.textContent = value;
        const confidenceNode = document.createElement("span");
        confidenceNode.className = "confidence";
        confidenceNode.textContent = Number(confidence).toFixed(4);
        row.append(idNode, valueNode, confidenceNode);
        return row;
    }

    function fillResults(containerId, items, valueField) {
        const container = document.getElementById(containerId);
        container.replaceChildren();
        if (!items.length) {
            const empty = document.createElement("div");
            empty.className = "empty";
            empty.textContent = "No occurrences detected.";
            container.appendChild(empty);
            return;
        }
        for (const item of items) {
            container.appendChild(occurrenceRow(item.id, item[valueField], item.confidence));
        }
    }

    imageInput.addEventListener("change", () => {
        const file = imageInput.files[0];
        if (!file) return;
        showOriginal(file);
        renderedImage.hidden = true;
        renderedImage.removeAttribute("src");
        renderedPlaceholder.hidden = false;
        renderedPlaceholder.textContent = "Press Run Prediction to generate the final visualization.";
        setStatus(`Ready: ${file.name}`);
    });

    runButton.addEventListener("click", async () => {
        const file = imageInput.files[0];
        if (!file) {
            setStatus("Please choose an image first.", true);
            return;
        }

        runButton.disabled = true;
        setStatus("Running icon detection, masking, OCR, and rendering...");
        renderedImage.hidden = true;
        renderedPlaceholder.hidden = false;
        renderedPlaceholder.textContent = "Processing...";

        try {
            const formData = new FormData();
            formData.append("lcd_image", file);
            formData.append("meter_family", document.getElementById("meterFamily").value);
            const response = await fetch("/predict", { method: "POST", body: formData });
            const payload = await response.json();
            if (!response.ok) {
                const detail = payload.detail || payload;
                throw new Error(detail.message || JSON.stringify(detail));
            }

            fillResults("textResults", payload.text_occurrences, "value");
            fillResults("iconResults", payload.icon_occurrences, "class_name");
            document.getElementById("textCount").textContent = payload.text_occurrences.length;
            document.getElementById("iconCount").textContent = payload.icon_occurrences.length;
            document.getElementById("totalTime").textContent = `${payload.timing.total_ms.toFixed(1)} ms`;
            document.getElementById("timingResults").textContent = JSON.stringify(payload.timing, null, 2);

            renderedImage.src = `${payload.rendered_result.url}?cache=${Date.now()}`;
            renderedImage.hidden = false;
            renderedPlaceholder.hidden = true;
            setStatus(`Completed successfully. Request ID: ${payload.request_id}`);
        } catch (error) {
            renderedPlaceholder.textContent = "Prediction failed. Review the message above and the server console.";
            setStatus(`Prediction failed: ${error.message}`, true);
        } finally {
            runButton.disabled = false;
        }
    });
</script>
</body>
</html>
    """
    page = page.replace('__DEFAULT_PIPELINE__', DEFAULT_PIPELINE)
    return HTMLResponse(content=page, status_code=200)


@app.get("/docs", include_in_schema=False)
def redirect_docs_without_slash():
    return RedirectResponse(url="/docs/", status_code=307)


@app.get("/docs/", include_in_schema=False)
def custom_swagger_documentation():
    swagger_response = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
    )
    swagger_html = swagger_response.body.decode("utf-8")

    # Before a browser refresh/navigation, remove Swagger's URL fragment.
    # The refreshed page therefore returns to the clean /docs/ view.
    refresh_script = """
    <script>
        window.addEventListener("beforeunload", function () {
            if (window.location.hash) {
                history.replaceState(null, "", "/docs/");
            }
        });
    </script>
    """
    swagger_html = swagger_html.replace(
        "</body>",
        refresh_script + "</body>",
    )
    return HTMLResponse(content=swagger_html, status_code=200)


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Check service and model health",
    tags=["Service"],
)
def check_service_health(request: Request):
    loaded = bool(getattr(request.app.state, "models_loaded", False))
    common = {"loaded": loaded, "device": "CPU"}
    return {
        "status": "healthy" if loaded else "unhealthy",
        "icon_detector": {
            **common,
            "model_name": "company_openvino_icon_detector",
            "runtime": "OpenVINO",
        },
        "ocr_detector": {
            **common,
            "model_name": "PP-OCRv6_small_det_lcd",
            "runtime": "PaddleOCR 3.7 isolated worker",
        },
        "ocr_recognizer": {
            **common,
            "model_name": "PP-OCRv6_small_rec_lcd",
            "runtime": "PaddleOCR 3.7 isolated worker",
        },
    }


@app.post(
    "/predict",
    response_model=DevelopmentPredictionResponse,
    summary="Extract text and icons from an LCD image",
    description=(
        "Runs icon detection, masks detected icons, runs the selected meter-family "
        "OCR pipeline, preserves duplicates, and creates named result artifacts."
    ),
    tags=["Prediction"],
    operation_id="extract_lcd_text_and_icons",
)
async def extract_lcd_text_and_icons(
    request: Request,
    lcd_image: UploadFile = File(..., description="LCD image to process."),
    meter_family: Literal["general_v3", "experimental_v6"] = Form(DEFAULT_PIPELINE, description="Select the OCR pipeline."),
):
    supported_types = {"image/png", "image/jpeg", "image/bmp", "image/webp"}
    content_type = lcd_image.content_type or ""
    if content_type not in supported_types:
        raise HTTPException(
            status_code=415,
            detail={
                "code": "UNSUPPORTED_FILE_TYPE",
                "message": "Supported formats are PNG, JPG, JPEG, BMP, and WebP.",
            },
        )

    image_bytes = await lcd_image.read()
    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail={"code": "EMPTY_FILE", "message": "The uploaded file is empty."},
        )

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_IMAGE",
                "message": "The uploaded file could not be decoded as an image.",
            },
        )

    request_id = str(uuid.uuid4())
    total_started = perf_counter()
    prediction = await run_in_threadpool(run_complete_pipeline, request.app, image, meter_family)
    rendered, rendering_ms = await run_in_threadpool(
        render_combined_result,
        image,
        prediction["icons"],
        prediction["texts"],
    )
    result_location = derive_result_location(lcd_image.filename)
    masked_path = result_location["masked_path"]
    rendered_path = result_location["rendered_path"]
    prediction_path = result_location["prediction_path"]
    rendered_filename = rendered_path.name
    cv2.imwrite(str(masked_path), prediction["masked_image"])
    if not cv2.imwrite(str(rendered_path), rendered):
        raise HTTPException(
            status_code=500,
            detail={
                "code": "RENDERING_FAILURE",
                "message": "Prediction succeeded, but the rendered image could not be saved.",
            },
        )

    height, width = image.shape[:2]
    response_payload = {
        "request_id": request_id,
        "mode": "development",
        "status": "success",
        "image": {
            "filename": Path(lcd_image.filename or "uploaded_image").name,
            "width": width,
            "height": height,
        },
        "text_occurrences": prediction["texts"],
        "icon_occurrences": prediction["icons"],
        "timing": {
            "icon_detection_ms": prediction["icon_ms"],
            "icon_masking_ms": prediction["masking_ms"],
            "ocr_detection_and_recognition_ms": prediction["ocr_ms"],
            "rendering_ms": rendering_ms,
            "total_ms": (perf_counter() - total_started) * 1000,
        },
        "rendered_result": {
            "available": True,
            "url": result_url(rendered_path),
        },
        "saved_results": {
            "run_name": result_location["run_name"],
            "masked_url": result_url(masked_path),
            "rendered_url": result_url(rendered_path),
            "prediction_url": result_url(prediction_path),
        },
        "warnings": (["Unknown meter_family; general_v3 fallback used."] if prediction["routing"]["fallback_used"] else []),
        "requested_meter_family": prediction["routing"]["requested_meter_family"],
        "resolved_pipeline": prediction["routing"]["resolved_pipeline"],
        "fallback_used": prediction["routing"]["fallback_used"],
        "fallback_reason": prediction["routing"]["fallback_reason"],
        "pipeline": {"icon_detector": prediction["routing"]["definition"].icon_detector, "icon_masking": True, "text_detector": prediction["routing"]["definition"].text_detector, "text_recognizer": prediction["routing"]["definition"].text_recognizer},
    }


    serializable_payload = json.loads(
        json.dumps(response_payload, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value))
    )
    prediction_path.write_text(
        json.dumps(serializable_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return response_payload


@app.get(
    "/results/{result_path:path}",
    summary="Open a saved result file",
    tags=["Results"],
    include_in_schema=False,
)
def get_named_result(result_path: str):
    requested = (RESULTS_ROOT / result_path).resolve()
    root = RESULTS_ROOT.resolve()
    try:
        requested.relative_to(root)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="Result not found") from error

    if not requested.is_file():
        raise HTTPException(status_code=404, detail="Result not found")

    media_type = "application/json" if requested.suffix.lower() == ".json" else "image/png"
    return FileResponse(requested, media_type=media_type, filename=requested.name)

@app.get(
    "/results/{request_id}.png",
    summary="Download a rendered development result",
    tags=["Results"],
)
def get_rendered_result(request_id: str):
    try:
        uuid.UUID(request_id)
    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail={"code": "RESULT_NOT_FOUND", "message": "The rendered result does not exist."},
        ) from error

    result_path = RESULTS_FOLDER / f"{request_id}.png"
    if not result_path.is_file():
        raise HTTPException(
            status_code=404,
            detail={"code": "RESULT_NOT_FOUND", "message": "The rendered result does not exist."},
        )

    return FileResponse(result_path, media_type="image/png", filename=result_path.name)


# FOLDER_PREDICTION_SUPPORT_V1
FOLDER_SUPPORTED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/bmp", "image/webp"}

class FolderPredictionFailure(BaseModel):
    filename: str
    code: str
    message: str

class FolderPredictionResponse(BaseModel):
    batch_id: str
    status: str
    total_files_received: int
    images_processed: int
    images_failed: int
    total_processing_time_ms: float
    results: list[dict]
    failures: list[FolderPredictionFailure]

def _response_to_dict(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return value
    raise TypeError("Unsupported single-image prediction response type.")

@app.post(
    "/predict/folder",
    response_model=FolderPredictionResponse,
    summary="Process a folder containing LCD images",
    description=(
        "Accepts one or more images. Each image receives fresh icon "
        "detection, icon masking, the selected meter-family OCR pipeline, "
        "and named result artifacts. Processing is sequential."
    ),
    tags=["Prediction"],
    operation_id="predict_lcd_image_folder",
)
async def predict_lcd_image_folder(
    request: Request,
    lcd_images: list[UploadFile] = File(..., description="Select one or more LCD images."),
    meter_family: Literal["general_v3", "experimental_v6"] = Form(DEFAULT_PIPELINE, description="Select one pipeline for the complete folder request."),
):
    received_count = len(lcd_images)
    if received_count < 1:
        raise HTTPException(status_code=400, detail={
            "code": "EMPTY_BATCH", "message": "Select at least one image.",
            "minimum_images": 1, "received_files": 0,
        })

    batch_id = str(uuid.uuid4())
    batch_started = perf_counter()
    successful_results = []
    failures = []

    for upload in lcd_images:
        original_name = upload.filename or "uploaded_image"
        filename = Path(original_name.replace("\\", "/")).name
        content_type = upload.content_type or ""
        if content_type not in FOLDER_SUPPORTED_CONTENT_TYPES:
            failures.append({
                "filename": filename,
                "code": "UNSUPPORTED_FILE_TYPE",
                "message": "Supported formats are PNG, JPG, JPEG, BMP, and WebP.",
            })
            continue
        try:
            prediction = await extract_lcd_text_and_icons(request=request, lcd_image=upload, meter_family=meter_family)
            prediction_data = _response_to_dict(prediction)
            prediction_data["source_relative_path"] = original_name
            successful_results.append(prediction_data)
        except HTTPException as error:
            detail = error.detail
            if isinstance(detail, dict):
                code = str(detail.get("code", "PREDICTION_FAILED"))
                message = str(detail.get("message", detail))
            else:
                code, message = "PREDICTION_FAILED", str(detail)
            failures.append({"filename": filename, "code": code, "message": message})
        except Exception as error:
            failures.append({"filename": filename, "code": "INTERNAL_ERROR", "message": str(error)})

    elapsed_ms = (perf_counter() - batch_started) * 1000
    processed_count = len(successful_results)
    failed_count = len(failures)
    status = "success" if processed_count == received_count else ("partial_success" if processed_count else "failed")
    return {
        "batch_id": batch_id,
        "status": status,
        "total_files_received": received_count,
        "images_processed": processed_count,
        "images_failed": failed_count,
        "total_processing_time_ms": elapsed_ms,
        "results": successful_results,
        "failures": failures,
    }

FOLDER_TEST_PAGE = r'''<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1"
    >

    <title>LCD Folder Tester</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f4f7fb;
            color: #172033;
        }

        header {
            padding: 18px 24px;
            background: white;
            border-bottom: 1px solid #dce3ec;
        }

        main {
            max-width: 1500px;
            margin: auto;
            padding: 24px;
        }

        .panel,
        .card {
            margin-bottom: 18px;
            padding: 18px;
            background: white;
            border: 1px solid #dce3ec;
            border-radius: 14px;
            box-shadow: 0 4px 18px rgba(28, 44, 70, 0.06);
        }

        .controls {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            align-items: center;
        }

        button,
        a.btn {
            padding: 10px 16px;
            border: 0;
            border-radius: 9px;
            background: #1769e0;
            color: white;
            font-weight: 700;
            text-decoration: none;
            cursor: pointer;
        }

        button:disabled {
            opacity: 0.55;
            cursor: not-allowed;
        }

        .muted {
            color: #657386;
        }

        .warning {
            color: #b42318;
            font-weight: 700;
        }

        .success {
            color: #087a55;
            font-weight: 700;
        }

        .progress-shell {
            position: relative;
            height: 28px;
            margin-top: 16px;
            overflow: hidden;
            background: #e7edf5;
            border: 1px solid #d4dce8;
            border-radius: 14px;
        }

        .progress-bar {
            width: 0%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(90deg, #1769e0, #0f9d8a);
            color: white;
            font-size: 13px;
            font-weight: 700;
            white-space: nowrap;
            transition: width 0.25s ease;
        }

        .progress-details {
            margin-top: 10px;
            line-height: 1.6;
        }

        .summary {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
        }

        .metric {
            padding: 12px;
            background: #f7f9fc;
            border-radius: 10px;
        }

        .metric strong {
            display: block;
            margin-top: 5px;
            color: #0f6b78;
            font-size: 22px;
        }

        .images,
        .data {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .box {
            padding: 10px;
            background: white;
            border: 1px solid #dce3ec;
            border-radius: 10px;
        }

        .box img {
            width: 100%;
            max-height: 520px;
            object-fit: contain;
        }

        .occ {
            padding: 8px 0;
            border-bottom: 1px solid #edf1f6;
        }

        .result-links {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 12px;
        }

        .result-links a {
            color: #1769e0;
            font-weight: 700;
        }

        @media (max-width: 900px) {
            .images,
            .data {
                grid-template-columns: 1fr;
            }

            .summary {
                grid-template-columns: 1fr 1fr;
            }
        }
    </style>
</head>

<body>
    <header>
        <h1>LCD Folder Prediction Tester</h1>

        <p>
            Select a folder containing supported LCD images.
            Images are processed sequentially with live progress.
        </p>
    </header>

    <main>
        <section class="panel">
            <div class="controls">
                <label for="folderMeterFamily">
                    <strong>Meter family</strong>
                </label>

                <select
                    id="folderMeterFamily"
                    data-default="__DEFAULT_PIPELINE__"
                >
                    <option value="general_v3">
                        General v3
                    </option>

                    <option value="experimental_v6">
                        Experimental v6
                    </option>
                </select>

                <input
                    id="files"
                    type="file"
                    webkitdirectory
                    multiple
                    accept="image/png,image/jpeg,image/bmp,image/webp"
                >

                <button id="run" disabled>
                    Run Folder Prediction
                </button>

                <a class="btn" href="/test">
                    Single Image
                </a>

                <a class="btn" href="/docs/">
                    API Docs
                </a>
            </div>

            <p id="message" class="muted">
                Select one or more supported images.
                There is no application-level image-count limit.
            </p>

            <div class="progress-shell">
                <div id="progressBar" class="progress-bar">
                    0%
                </div>
            </div>

            <div id="progress" class="progress-details muted">
                No folder prediction is running.
            </div>
        </section>

        <section id="summary" class="panel" hidden>
            <div class="summary">
                <div class="metric">
                    Selected
                    <strong id="received">0</strong>
                </div>

                <div class="metric">
                    Successful
                    <strong id="processed">0</strong>
                </div>

                <div class="metric">
                    Failed
                    <strong id="failed">0</strong>
                </div>

                <div class="metric">
                    Total Time
                    <strong id="time">0 ms</strong>
                </div>
            </div>
        </section>

        <section id="results"></section>
    </main>

    <script>
        document.getElementById(
            "folderMeterFamily"
        ).value = "__DEFAULT_PIPELINE__";

        const input = document.getElementById("files");
        const run = document.getElementById("run");
        const message = document.getElementById("message");
        const progress = document.getElementById("progress");
        const progressBar = document.getElementById("progressBar");
        const results = document.getElementById("results");
        const summary = document.getElementById("summary");
        const received = document.getElementById("received");
        const processed = document.getElementById("processed");
        const failed = document.getElementById("failed");
        const time = document.getElementById("time");
        const meterFamily = document.getElementById(
            "folderMeterFamily"
        );

        const allowedMimeTypes = new Set([
            "image/png",
            "image/jpeg",
            "image/bmp",
            "image/webp"
        ]);

        const allowedExtensions = new Set([
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".webp"
        ]);

        let selected = [];
        let running = false;

        function escapeHtml(value) {
            return String(value).replace(
                /[&<>'"]/g,
                function (character) {
                    const replacements = {
                        "&": "&amp;",
                        "<": "&lt;",
                        ">": "&gt;",
                        "'": "&#39;",
                        '"': "&quot;"
                    };

                    return replacements[character];
                }
            );
        }

        function isSupportedImage(file) {
            if (allowedMimeTypes.has(file.type)) {
                return true;
            }

            const lowerName = file.name.toLowerCase();

            for (const extension of allowedExtensions) {
                if (lowerName.endsWith(extension)) {
                    return true;
                }
            }

            return false;
        }

        function occurrenceRows(items, kind) {
            if (!items || items.length === 0) {
                return '<div class="muted">None detected</div>';
            }

            return items.map(function (item) {
                const value = kind === "text"
                    ? item.value
                    : item.class_name;

                return `
                    <div class="occ">
                        <strong>
                            ${escapeHtml(item.id)}:
                            ${escapeHtml(value)}
                        </strong>
                        <br>
                        <span class="muted">
                            Confidence
                            ${Number(item.confidence).toFixed(4)}
                        </span>
                    </div>
                `;
            }).join("");
        }

        function getRenderedUrl(item) {
            if (
                item.rendered_result &&
                item.rendered_result.url
            ) {
                return item.rendered_result.url;
            }

            if (
                item.saved_results &&
                item.saved_results.rendered_url
            ) {
                return item.saved_results.rendered_url;
            }

            return "";
        }

        function getSavedLinks(item) {
            if (!item.saved_results) {
                return "";
            }

            const links = [];

            if (item.saved_results.masked_url) {
                links.push(`
                    <a
                        href="${escapeHtml(
                            item.saved_results.masked_url
                        )}"
                        target="_blank"
                    >
                        Open Masked Image
                    </a>
                `);
            }

            if (item.saved_results.rendered_url) {
                links.push(`
                    <a
                        href="${escapeHtml(
                            item.saved_results.rendered_url
                        )}"
                        target="_blank"
                    >
                        Open Rendered Image
                    </a>
                `);
            }

            if (item.saved_results.prediction_url) {
                links.push(`
                    <a
                        href="${escapeHtml(
                            item.saved_results.prediction_url
                        )}"
                        target="_blank"
                    >
                        Open Prediction JSON
                    </a>
                `);
            }

            if (links.length === 0) {
                return "";
            }

            return `
                <div class="result-links">
                    ${links.join("")}
                </div>
            `;
        }

        function appendSuccessfulResult(item, previewUrl) {
            const name = (
                item.image &&
                item.image.filename
            )
                ? item.image.filename
                : "uploaded_image";

            const renderedUrl = getRenderedUrl(item);

            const textItems = item.text_occurrences || [];
            const iconItems = item.icon_occurrences || [];

            const card = document.createElement("article");
            card.className = "card";

            card.innerHTML = `
                <h2>${escapeHtml(name)}</h2>

                <div class="images">
                    <div class="box">
                        <h3>Original</h3>
                        <img
                            src="${escapeHtml(previewUrl)}"
                            alt="Original ${escapeHtml(name)}"
                        >
                    </div>

                    <div class="box">
                        <h3>Rendered Result</h3>

                        ${
                            renderedUrl
                                ? `
                                    <img
                                        src="${escapeHtml(renderedUrl)}"
                                        alt="Rendered ${escapeHtml(name)}"
                                    >
                                `
                                : `
                                    <div class="warning">
                                        Rendered result URL was not returned.
                                    </div>
                                `
                        }
                    </div>
                </div>

                <div class="data">
                    <div class="box">
                        <h3>
                            Text (${textItems.length})
                        </h3>

                        ${occurrenceRows(textItems, "text")}
                    </div>

                    <div class="box">
                        <h3>
                            Icons (${iconItems.length})
                        </h3>

                        ${occurrenceRows(iconItems, "icon")}
                    </div>
                </div>

                ${getSavedLinks(item)}
            `;

            results.appendChild(card);
        }

        function appendFailure(displayName, errorMessage) {
            const card = document.createElement("section");
            card.className = "panel";

            card.innerHTML = `
                <h2>
                    Failed:
                    ${escapeHtml(displayName)}
                </h2>

                <div class="warning">
                    ${escapeHtml(errorMessage)}
                </div>
            `;

            results.appendChild(card);
        }

        function updateProgress(
            completed,
            total,
            succeeded,
            failedCount,
            currentName,
            startedAt
        ) {
            const percentage = total > 0
                ? Math.round((completed / total) * 100)
                : 0;

            progressBar.style.width = `${percentage}%`;
            progressBar.textContent = `${percentage}%`;

            received.textContent = String(total);
            processed.textContent = String(succeeded);
            failed.textContent = String(failedCount);

            time.textContent = (
                performance.now() - startedAt
            ).toFixed(1) + " ms";

            progress.innerHTML = `
                Completed
                <strong>${completed} of ${total}</strong>
                <br>

                ${
                    currentName
                        ? `
                            Last file:
                            ${escapeHtml(currentName)}
                            <br>
                        `
                        : ""
                }

                Successful: ${succeeded}
                |
                Failed: ${failedCount}
            `;
        }

        input.addEventListener("change", function () {
            const allFiles = Array.from(input.files);

            selected = allFiles.filter(isSupportedImage);

            const ignoredCount = (
                allFiles.length - selected.length
            );

            if (selected.length === 0) {
                message.textContent =
                    "No supported images were selected.";

                run.disabled = true;
                return;
            }

            message.textContent =
                `Selected ${selected.length} supported image(s)` +
                (
                    ignoredCount > 0
                        ? ` and ignored ${ignoredCount} unsupported file(s).`
                        : "."
                );

            run.disabled = false;
        });

        run.addEventListener("click", async function () {
            if (running) {
                return;
            }

            if (selected.length < 1) {
                message.innerHTML = `
                    <span class="warning">
                        Select at least one supported image.
                    </span>
                `;

                return;
            }

            running = true;
            run.disabled = true;
            input.disabled = true;
            meterFamily.disabled = true;

            results.innerHTML = "";
            summary.hidden = false;

            const total = selected.length;
            const startedAt = performance.now();
            const selectedMeterFamily = meterFamily.value;

            let completed = 0;
            let succeeded = 0;
            let failedCount = 0;

            received.textContent = String(total);
            processed.textContent = "0";
            failed.textContent = "0";
            time.textContent = "0 ms";

            progressBar.style.width = "0%";
            progressBar.textContent = "0%";

            for (const file of selected) {
                const relativeName = (
                    file.webkitRelativePath ||
                    file.name
                );

                progress.innerHTML = `
                    Processing
                    <strong>
                        ${completed + 1} of ${total}
                    </strong>
                    <br>

                    Current file:
                    ${escapeHtml(relativeName)}
                    <br>

                    Successful: ${succeeded}
                    |
                    Failed: ${failedCount}
                `;

                const previewUrl = URL.createObjectURL(file);
                const formData = new FormData();

                formData.append(
                    "meter_family",
                    selectedMeterFamily
                );

                formData.append(
                    "lcd_images",
                    file,
                    relativeName
                );

                try {
                    const response = await fetch(
                        "/predict/folder",
                        {
                            method: "POST",
                            body: formData
                        }
                    );

                    let payload;

                    try {
                        payload = await response.json();
                    } catch (jsonError) {
                        throw new Error(
                            `Server returned HTTP ${response.status} ` +
                            "with a non-JSON response."
                        );
                    }

                    if (!response.ok) {
                        const detail = payload.detail;

                        let serverMessage;

                        if (
                            detail &&
                            typeof detail === "object"
                        ) {
                            serverMessage = (
                                detail.message ||
                                JSON.stringify(detail)
                            );
                        } else {
                            serverMessage = (
                                detail ||
                                JSON.stringify(payload)
                            );
                        }

                        throw new Error(serverMessage);
                    }

                    if (
                        payload.results &&
                        payload.results.length > 0
                    ) {
                        appendSuccessfulResult(
                            payload.results[0],
                            previewUrl
                        );

                        succeeded += 1;
                    } else {
                        let failureMessage =
                            "Prediction returned no result.";

                        if (
                            payload.failures &&
                            payload.failures.length > 0
                        ) {
                            failureMessage = (
                                payload.failures[0].message ||
                                failureMessage
                            );
                        }

                        throw new Error(failureMessage);
                    }
                } catch (error) {
                    failedCount += 1;

                    appendFailure(
                        relativeName,
                        error.message || String(error)
                    );
                }

                completed += 1;

                updateProgress(
                    completed,
                    total,
                    succeeded,
                    failedCount,
                    relativeName,
                    startedAt
                );
            }

            progressBar.style.width = "100%";
            progressBar.textContent = "100%";

            progress.innerHTML = `
                <span class="success">
                    Folder prediction completed.
                </span>
                <br>

                Processed ${completed} of ${total} files.
                <br>

                Successful: ${succeeded}
                |
                Failed: ${failedCount}
            `;

            time.textContent = (
                performance.now() - startedAt
            ).toFixed(1) + " ms";

            running = false;
            run.disabled = false;
            input.disabled = false;
            meterFamily.disabled = false;
        });
    </script>
</body>
</html>'''

@app.get("/folder-test", response_class=HTMLResponse, include_in_schema=False)
def get_folder_test_page():
    return HTMLResponse(content=FOLDER_TEST_PAGE.replace('__DEFAULT_PIPELINE__', DEFAULT_PIPELINE))
