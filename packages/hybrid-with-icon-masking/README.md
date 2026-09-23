# Hybrid OCR with Pre-OCR Icon Masking

## Components

OCR detector: Model 2, det_lcd_v2_expanded_240

OCR recognizer: Model 3, rec_lcd_v3_full_reviewed

Icon detector: existing company OpenVINO icon detector

## Method

The icon detector runs first on each original image. Every detected icon region is filled with black pixels, with two pixels of padding. The hybrid OCR pipeline then runs on the masked image.

## Visual outcome

The 97-image end-to-end run no longer reports the trusted detected icons as OCR text. The rendered OCR outputs are included under results/ocr_visual_results.

## Controlled detector evaluation on the same 97 images

Unmasked control:

Precision: 0.5115384615384615
Recall: 0.7169811320754716
Hmean: 0.5970819304152638
FPS: 0.6987833929373587

Icon-masked run:

Precision: 0.4649910233393178
Recall: 0.6981132075471698
Hmean: 0.5581896551724138
FPS: 0.6942090873756062

Change:

Precision: -4.65 percentage points
Recall: -1.89 percentage points
Hmean: -3.89 percentage points

## Decision

The masked pipeline is retained as a functional application-level baseline because it removes icon outputs from OCR and appears visually cleaner for the intended use. The lower standalone text-detection metrics are documented and must not be hidden.

Future quality improvement should focus on larger and more representative OCR training and testing data, hard negatives, inactive segments, dense layouts, and accurate annotations. Both unmasked and icon-masked modes must remain available for comparison.

## Reproduction

The runner uses the existing icon detection JSON and then invokes the packaged Model 2 detector and Model 3 recognizer. Paths can be overridden through command-line arguments.
