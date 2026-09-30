from pathlib import Path
from time import perf_counter
import json

import cv2
import numpy as np
import openvino as ov
import yaml


class IconDetector:
    def __init__(
        self,
        model_path: Path,
        labels_path: Path,
        transforms_path: Path,
        train_config_path: Path,
        threshold: float = 0.38,
        device: str = "CPU",
    ):
        self.threshold = threshold
        self.class_names = self._load_class_names(labels_path)
        self.preprocessing = self._load_preprocessing(
            transforms_path,
            train_config_path,
        )

        core = ov.Core()
        model = core.read_model(model_path)
        self.compiled_model = core.compile_model(model, device)
        self.input_name = self.compiled_model.input(0).get_any_name()
        self._warm_up()

    @staticmethod
    def _load_class_names(labels_path: Path) -> dict[int, str]:
        data = json.loads(labels_path.read_text(encoding="utf-8"))
        return {
            int(class_id): class_information["name"]
            for class_id, class_information in data.items()
        }

    @staticmethod
    def _load_preprocessing(
        transforms_path: Path,
        train_config_path: Path,
    ) -> dict:
        transforms = yaml.safe_load(
            transforms_path.read_text(encoding="utf-8")
        )
        train_config = yaml.safe_load(
            train_config_path.read_text(encoding="utf-8")
        )

        validation_transforms = transforms["valid"]
        resize_settings = next(
            item["RescaleWithPadding"]
            for item in validation_transforms
            if "RescaleWithPadding" in item
        )
        normalization_settings = next(
            item["NormalizeMeanStd"]
            for item in validation_transforms
            if "NormalizeMeanStd" in item
        )

        return {
            "width": int(resize_settings["width"]),
            "height": int(resize_settings["height"]),
            "mean": np.asarray(
                normalization_settings["mean"],
                dtype=np.float32,
            ),
            "std": np.asarray(
                normalization_settings["std"],
                dtype=np.float32,
            ),
            "metadata_threshold": float(
                train_config["model"].get("score_threshold", 0.38)
            ),
        }

    def _warm_up(self) -> None:
        warmup_input = np.zeros(
            (
                1,
                3,
                self.preprocessing["height"],
                self.preprocessing["width"],
            ),
            dtype=np.float32,
        )
        self.compiled_model({self.input_name: warmup_input})

    def _prepare_image(self, image_bgr: np.ndarray):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        source_height, source_width = image_rgb.shape[:2]
        target_width = self.preprocessing["width"]
        target_height = self.preprocessing["height"]

        scale = min(
            target_width / source_width,
            target_height / source_height,
        )
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))

        resized_image = cv2.resize(
            image_rgb,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        padded_image = np.zeros(
            (target_height, target_width, 3),
            dtype=np.uint8,
        )
        padding_left = (target_width - resized_width) // 2
        padding_top = (target_height - resized_height) // 2
        padded_image[
            padding_top:padding_top + resized_height,
            padding_left:padding_left + resized_width,
        ] = resized_image

        normalized_image = (
            padded_image.astype(np.float32)
            - self.preprocessing["mean"]
        ) / self.preprocessing["std"]

        input_tensor = np.transpose(
            normalized_image,
            (2, 0, 1),
        )[None].astype(np.float32, copy=False)

        geometry = {
            "source_width": source_width,
            "source_height": source_height,
            "scale": scale,
            "padding_left": padding_left,
            "padding_top": padding_top,
        }
        return input_tensor, geometry

    @staticmethod
    def _restore_box(model_box, geometry) -> list[int]:
        scale = geometry["scale"]
        padding_left = geometry["padding_left"]
        padding_top = geometry["padding_top"]
        source_width = geometry["source_width"]
        source_height = geometry["source_height"]

        x_min = int(round((float(model_box[0]) - padding_left) / scale))
        y_min = int(round((float(model_box[1]) - padding_top) / scale))
        x_max = int(round((float(model_box[2]) - padding_left) / scale))
        y_max = int(round((float(model_box[3]) - padding_top) / scale))

        x_min = max(0, min(x_min, source_width - 1))
        y_min = max(0, min(y_min, source_height - 1))
        x_max = max(0, min(x_max, source_width - 1))
        y_max = max(0, min(y_max, source_height - 1))

        return [
            min(x_min, x_max),
            min(y_min, y_max),
            max(x_min, x_max),
            max(y_min, y_max),
        ]

    def predict(self, image_bgr: np.ndarray):
        input_tensor, geometry = self._prepare_image(image_bgr)

        inference_started = perf_counter()
        raw_outputs = self.compiled_model(
            {self.input_name: input_tensor}
        )
        inference_time_ms = (
            perf_counter() - inference_started
        ) * 1000

        output_map = {
            output_port.get_any_name(): np.asarray(output_value)
            for output_port, output_value in raw_outputs.items()
        }

        detections = []
        for raw_detection, raw_label in zip(
            output_map["dets"][0],
            output_map["labels"][0],
        ):
            confidence = float(raw_detection[4])
            class_id = int(raw_label)

            if confidence < self.threshold or class_id == 0:
                continue

            box = self._restore_box(raw_detection[:4], geometry)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self.class_names.get(
                        class_id,
                        str(class_id),
                    ),
                    "confidence": confidence,
                    "box": box,
                }
            )

        detections.sort(
            key=lambda item: item["confidence"],
            reverse=True,
        )
        return detections, inference_time_ms
