# Native Model 3 versus Hybrid Pipeline

## Input set

Both pipelines were tested on the corrected Model 3 validation images.

Input images: 97

## Native Model 3

Detector:

det_lcd_v3_full_reviewed

Recognizer:

rec_lcd_v3_full_reviewed

Detector standalone Hmean:

0.5275181723779855

Rendered files:

98

Detector-generated crops:

413

## Hybrid pipeline

Detector:

det_lcd_v2_expanded_240

Recognizer:

rec_lcd_v3_full_reviewed

Detector standalone Hmean:

0.6863468634686346

Rendered files:

98

Detector-generated crops:

393

## Initial recommendation

The hybrid pipeline is the leading candidate because Model 2 has the stronger
standalone detector and Model 3 has the stronger standalone recognizer.

Final selection requires visual side-by-side review and, ideally, manual
end-to-end scoring against corrected annotations.
