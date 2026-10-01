from dataclasses import asdict, dataclass
import os

DEFAULT_PIPELINE = os.environ.get("LCD_METER_FAMILY", "general_v3").strip() or "general_v3"
if DEFAULT_PIPELINE not in {"general_v3", "experimental_v6"}:
    raise RuntimeError("LCD_METER_FAMILY must be general_v3 or experimental_v6")

@dataclass(frozen=True)
class PipelineDefinition:
    pipeline_id: str
    icon_detector: str
    icon_masking: bool
    text_detector: str
    text_recognizer: str
    engine_state_name: str

PIPELINES = {
    "general_v3": PipelineDefinition("general_v3", "company_openvino_icon_detector", True, "det_lcd_v2_expanded_240", "rec_lcd_v3_full_reviewed", "ocr_engine_general_v3"),
    "experimental_v6": PipelineDefinition("experimental_v6", "company_openvino_icon_detector", True, "PP-OCRv6_small_det_lcd", "PP-OCRv6_small_rec_lcd", "ocr_engine_experimental_v6"),
}

def resolve_pipeline(requested):
    requested = (requested or DEFAULT_PIPELINE).strip() or DEFAULT_PIPELINE
    if requested in PIPELINES:
        return {"requested_meter_family": requested, "resolved_pipeline": requested, "fallback_used": False, "fallback_reason": None, "definition": PIPELINES[requested]}
    return {"requested_meter_family": requested, "resolved_pipeline": DEFAULT_PIPELINE, "fallback_used": True, "fallback_reason": "unknown_meter_family", "definition": PIPELINES[DEFAULT_PIPELINE]}

def public_pipeline_catalog():
    return {name: {k: v for k, v in asdict(item).items() if k != "engine_state_name"} for name, item in PIPELINES.items()}
