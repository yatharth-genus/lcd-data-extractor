# Meter Family API with PP-OCRv6

Self-contained API package following the same layout as the earlier live icon-masked hybrid package.

## Selected pipeline

The default is `experimental_v6`:

```text
LCD image
→ OpenVINO icon detector
→ strengthened icon masking
→ fine-tuned PP-OCRv6 small detector
→ fine-tuned PP-OCRv6 small recognizer
→ JSON response and rendered image
```

`general_v3` remains packaged as a historical fallback and registry example.

## Package layout

```text
packages/meter-family-api/
├── app/
├── models/
│   ├── icon_detector/
│   ├── general_v3/
│   └── experimental_v6/
├── requirements/
├── results/
├── runtime/
├── setup.ps1
├── run_api.ps1
├── validate_package.ps1
└── README.md
```

## Clone and download models

```powershell
git clone --branch feature/meter-family-api-v6 --single-branch <COMPANY_REPOSITORY_URL> lcd-data-extractor
git lfs pull
cd .\lcd-data-extractor\packages\meter-family-api
```

## Validate package

```powershell
powershell -ExecutionPolicy Bypass -File .alidate_package.ps1
```

## One-time setup

Internet access is required for dependency installation and cloning PaddleOCR 3.7.

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

## Start the API with PP-OCRv6

```powershell
powershell -ExecutionPolicy Bypass -File .
un_api.ps1 -MeterFamily experimental_v6
```

Open:

- Visual tester: http://127.0.0.1:8000/test
- Folder tester: http://127.0.0.1:8000/folder-test
- Swagger: http://127.0.0.1:8000/docs/
- Health: http://127.0.0.1:8000/health

The single-image tester, folder tester, and Swagger interface include a meter-family dropdown.

## Folder limit

The HTTP folder endpoint accepts between 1 and 20 images per request. This limit does not apply to offline model evaluation.

## Results

Rendered API images are saved under:

```text
packages/meter-family-api/results/rendered
```

The latest PP-OCRv6 masked debug image is saved under:

```text
packages/meter-family-api/results/debug/debug_last_masked_v6.png
```

The API response contains text occurrences, icon occurrences, confidences, coordinates, timings, routing metadata, and the rendered-result URL. Duplicate values are preserved.

## Runtime notes

- The API environment and PP-OCRv6 environment remain separate to avoid PaddlePaddle version conflicts.
- `run_api.ps1` starts the FastAPI process and points the persistent PP-OCRv6 worker to the packaged runtime.
- No machine-specific absolute paths are required.
- Model binaries use Git LFS.
