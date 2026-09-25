from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.icon_detector import IconDetector
from app.ocr_engine import OcrEngine


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FOLDER = PACKAGE_ROOT / "results"

ICON_MODEL_PATH = PACKAGE_ROOT / "models" / "icon_detector" / "saved_model.xml"
ICON_LABELS_PATH = PACKAGE_ROOT / "models" / "icon_detector" / "dm.json"
ICON_TRANSFORMS_PATH = PACKAGE_ROOT / "models" / "icon_detector" / "transforms.yaml"
ICON_TRAIN_CONFIG_PATH = PACKAGE_ROOT / "models" / "icon_detector" / "train.yaml"
OCR_DETECTOR_MODEL_PATH = PACKAGE_ROOT / "models" / "ocr_detector"
OCR_RECOGNIZER_MODEL_PATH = PACKAGE_ROOT / "models" / "ocr_recognizer"
OCR_CHARACTER_DICTIONARY_PATH = PACKAGE_ROOT / "models" / "ocr_recognizer" / "character_dict.txt"

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


class DevelopmentPredictionResponse(BaseModel):
    request_id: str
    mode: str
    status: str
    image: ImageInformation
    text_occurrences: list[TextOccurrence]
    icon_occurrences: list[IconOccurrence]
    timing: ProcessingTiming
    rendered_result: RenderedResult
    warnings: list[str]


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


def run_complete_pipeline(app, image):
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
    raw_texts, ocr_ms = app.state.ocr_engine.predict(masked)
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
        "icons": icons,
        "texts": texts,
        "icon_ms": icon_ms,
        "masking_ms": masking_ms,
        "ocr_ms": ocr_ms,
    }


@asynccontextmanager
async def application_lifespan(app: FastAPI):
    required_files = [
        ICON_MODEL_PATH,
        ICON_LABELS_PATH,
        ICON_TRANSFORMS_PATH,
        ICON_TRAIN_CONFIG_PATH,
        OCR_CHARACTER_DICTIONARY_PATH,
    ]
    required_folders = [OCR_DETECTOR_MODEL_PATH, OCR_RECOGNIZER_MODEL_PATH]
    missing = [str(path) for path in required_files if not path.is_file()]
    missing += [str(path) for path in required_folders if not path.is_dir()]
    if missing:
        raise RuntimeError("Required model paths are missing: " + "; ".join(missing))

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    app.state.icon_detector = IconDetector(
        model_path=ICON_MODEL_PATH,
        labels_path=ICON_LABELS_PATH,
        transforms_path=ICON_TRANSFORMS_PATH,
        train_config_path=ICON_TRAIN_CONFIG_PATH,
        threshold=ICON_THRESHOLD,
        device="CPU",
    )
    app.state.ocr_engine = OcrEngine(
        detector_model_path=OCR_DETECTOR_MODEL_PATH,
        recognizer_model_path=OCR_RECOGNIZER_MODEL_PATH,
        character_dictionary_path=OCR_CHARACTER_DICTIONARY_PATH,
    )
    app.state.models_loaded = True
    yield
    app.state.models_loaded = False
    app.state.icon_detector = None
    app.state.ocr_engine = None


app = FastAPI(
    title="LCD Understanding API",
    description=(
        "Development API for extracting duplicate-preserving text and "
        "icon occurrences from utility-meter LCD images."
    ),
    version="0.4.1",
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
        <p>Icon detector + icon masking + Model 2 detector + Model 3 recognizer</p>
    </div>
    <nav><a href="/docs/" target="_blank">Open API Docs</a></nav>
</header>
<main>
    <section class="toolbar">
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
            "model_name": "det_lcd_v2_expanded_240",
            "runtime": "Paddle Inference",
        },
        "ocr_recognizer": {
            **common,
            "model_name": "rec_lcd_v3_full_reviewed",
            "runtime": "Paddle Inference",
        },
    }


@app.post(
    "/predict",
    response_model=DevelopmentPredictionResponse,
    summary="Extract text and icons from an LCD image",
    description=(
        "Runs icon detection, hides detected icons, runs the finalized hybrid "
        "OCR pipeline, preserves duplicates, and creates a rendered result."
    ),
    tags=["Prediction"],
    operation_id="extract_lcd_text_and_icons",
)
async def extract_lcd_text_and_icons(
    request: Request,
    lcd_image: UploadFile = File(..., description="LCD image to process."),
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
    prediction = await run_in_threadpool(run_complete_pipeline, request.app, image)
    rendered, rendering_ms = await run_in_threadpool(
        render_combined_result,
        image,
        prediction["icons"],
        prediction["texts"],
    )

    rendered_filename = f"{request_id}.png"
    rendered_path = RESULTS_FOLDER / rendered_filename
    if not cv2.imwrite(str(rendered_path), rendered):
        raise HTTPException(
            status_code=500,
            detail={
                "code": "RENDERING_FAILURE",
                "message": "Prediction succeeded, but the rendered image could not be saved.",
            },
        )

    height, width = image.shape[:2]
    return {
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
            "url": f"/results/{rendered_filename}",
        },
        "warnings": [],
    }


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
MAX_FOLDER_IMAGES = 20
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
    summary="Process a folder containing up to 20 LCD images",
    description=(
        "Accepts between 1 and 20 images. Each image receives fresh icon "
        "detection, icon masking, Model 2 text detection, Model 3 text "
        "recognition, and a rendered result. Processing is sequential."
    ),
    tags=["Prediction"],
    operation_id="predict_lcd_image_folder",
)
async def predict_lcd_image_folder(
    request: Request,
    lcd_images: list[UploadFile] = File(..., description="Select 1 to 20 LCD images."),
):
    received_count = len(lcd_images)
    if received_count < 1:
        raise HTTPException(status_code=400, detail={
            "code": "EMPTY_BATCH", "message": "Select at least one image.",
            "minimum_images": 1, "received_files": 0,
        })
    if received_count > MAX_FOLDER_IMAGES:
        raise HTTPException(status_code=400, detail={
            "code": "BATCH_LIMIT_EXCEEDED",
            "message": "A folder prediction request can contain a maximum of 20 images.",
            "maximum_images": MAX_FOLDER_IMAGES,
            "received_files": received_count,
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
            prediction = await extract_lcd_text_and_icons(request=request, lcd_image=upload)
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
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LCD Folder Tester</title>
<style>
*{box-sizing:border-box}body{margin:0;font-family:Arial,sans-serif;background:#f4f7fb;color:#172033}header{background:white;border-bottom:1px solid #dce3ec;padding:18px 24px}main{max-width:1500px;margin:auto;padding:24px}.panel,.card{background:white;border:1px solid #dce3ec;border-radius:14px;padding:18px;margin-bottom:18px;box-shadow:0 4px 18px rgba(28,44,70,.06)}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center}button,a.btn{border:0;border-radius:9px;padding:10px 16px;background:#1769e0;color:white;font-weight:700;text-decoration:none;cursor:pointer}button:disabled{opacity:.55}.muted{color:#657386}.warning{color:#b42318;font-weight:700}.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:#f7f9fc;border-radius:10px;padding:12px}.metric strong{display:block;font-size:22px}.images,.data{display:grid;grid-template-columns:1fr 1fr;gap:16px}.box{border:1px solid #dce3ec;border-radius:10px;padding:10px;background:white}.box img{width:100%;max-height:520px;object-fit:contain}.occ{padding:8px 0;border-bottom:1px solid #edf1f6}@media(max-width:900px){.images,.data{grid-template-columns:1fr}.summary{grid-template-columns:1fr 1fr}}
</style></head><body>
<header><h1>LCD Folder Prediction Tester</h1><p>Select a folder containing 1 to 20 supported images.</p></header>
<main><section class="panel"><div class="controls"><input id="files" type="file" webkitdirectory multiple accept="image/png,image/jpeg,image/bmp,image/webp"><button id="run">Run Folder Prediction</button><a class="btn" href="/test">Single Image</a><a class="btn" href="/docs/">API Docs</a></div><p id="message" class="muted">Maximum 20 images per folder request.</p><p id="progress"></p></section>
<section id="summary" class="panel" hidden><div class="summary"><div class="metric">Received<strong id="received">0</strong></div><div class="metric">Processed<strong id="processed">0</strong></div><div class="metric">Failed<strong id="failed">0</strong></div><div class="metric">Total Time<strong id="time">0 ms</strong></div></div></section><section id="results"></section></main>
<script>
const input=document.getElementById('files'),run=document.getElementById('run'),message=document.getElementById('message'),progress=document.getElementById('progress'),results=document.getElementById('results'),summary=document.getElementById('summary');
const allowed=new Set(['image/png','image/jpeg','image/bmp','image/webp']);let selected=[];
const esc=v=>String(v).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
input.onchange=()=>{selected=Array.from(input.files).filter(f=>allowed.has(f.type));if(selected.length>20){message.innerHTML=`<span class="warning">Selected ${selected.length} supported images. Maximum is 20.</span>`;run.disabled=true}else{message.textContent=selected.length?`Selected ${selected.length} supported image(s). Maximum is 20.`:'No supported images selected.';run.disabled=false}};
function rows(items,kind){if(!items||!items.length)return'<div class="muted">None detected</div>';return items.map(x=>`<div class="occ"><strong>${esc(x.id)}: ${esc(kind==='text'?x.value:x.class_name)}</strong><br><span class="muted">Confidence ${Number(x.confidence).toFixed(4)}</span></div>`).join('')}
run.onclick=async()=>{if(selected.length<1||selected.length>20){message.innerHTML='<span class="warning">Select between 1 and 20 supported images.</span>';return}run.disabled=true;results.innerHTML='';summary.hidden=true;progress.textContent=`Processing ${selected.length} image(s)...`;const form=new FormData(),previews=new Map();selected.forEach(file=>{form.append('lcd_images',file,file.webkitRelativePath||file.name);previews.set(file.name,URL.createObjectURL(file))});try{const response=await fetch('/predict/folder',{method:'POST',body:form}),payload=await response.json();if(!response.ok)throw new Error(payload.detail?.message||JSON.stringify(payload.detail));received.textContent=payload.total_files_received;processed.textContent=payload.images_processed;failed.textContent=payload.images_failed;time.textContent=`${Number(payload.total_processing_time_ms).toFixed(1)} ms`;summary.hidden=false;progress.textContent=`Completed ${payload.images_processed} of ${payload.total_files_received}.`;payload.results.forEach(item=>{const name=item.image.filename,card=document.createElement('article');card.className='card';card.innerHTML=`<h2>${esc(name)}</h2><div class="images"><div class="box"><h3>Original</h3><img src="${previews.get(name)||''}"></div><div class="box"><h3>Rendered Result</h3><img src="${item.rendered_result.url}"></div></div><div class="data"><div class="box"><h3>Text (${item.text_occurrences.length})</h3>${rows(item.text_occurrences,'text')}</div><div class="box"><h3>Icons (${item.icon_occurrences.length})</h3>${rows(item.icon_occurrences,'icon')}</div></div>`;results.appendChild(card)});if(payload.failures.length){const card=document.createElement('section');card.className='panel';card.innerHTML='<h2>Failures</h2>'+payload.failures.map(x=>`<div class="occ"><strong>${esc(x.filename)}</strong>: ${esc(x.message)}</div>`).join('');results.appendChild(card)}}catch(error){progress.innerHTML=`<span class="warning">${esc(error.message)}</span>`}finally{run.disabled=false}};
</script></body></html>'''

@app.get("/folder-test", response_class=HTMLResponse, include_in_schema=False)
def get_folder_test_page():
    return HTMLResponse(content=FOLDER_TEST_PAGE)


