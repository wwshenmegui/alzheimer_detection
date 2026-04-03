from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_FORMATS = {"PNG", "JPEG", "JPG"}
DEFAULT_MIN_IMAGE_SIZE = (128, 128)
DEFAULT_ASPECT_RATIO_RANGE = (0.75, 1.4)
DEFAULT_MIN_STDDEV = 0.03
DEFAULT_MIN_CENTER_BORDER_DIFF = 0.02
DEFAULT_DUPLICATE_HASH_DISTANCE = 6


@dataclass
class ValidationFeedback:
    passed: bool
    error_code: str | None = None
    message: str | None = None
    user_action: str | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["details"] = payload.get("details") or {}
        return payload


class InputValidationError(Exception):
    def __init__(self, feedback: ValidationFeedback):
        super().__init__(feedback.message or "Input validation failed.")
        self.feedback = feedback


def compute_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_average_hash(image: Image.Image, hash_size: int = 8) -> str:
    grayscale = image.convert("L").resize((hash_size, hash_size))
    pixels = np.asarray(grayscale, dtype=np.float32)
    threshold = float(pixels.mean())
    bits = (pixels >= threshold).astype(np.uint8).flatten().tolist()
    return "".join(str(bit) for bit in bits)


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("Hash strings must have equal length.")
    return sum(char_left != char_right for char_left, char_right in zip(left, right))


def extract_patient_id(file_path: Path, filename_regex: str | None = None) -> str | None:
    if not filename_regex:
        return None
    match = re.search(filename_regex, file_path.name)
    if not match:
        return None
    if match.groupdict().get("patient_id"):
        return str(match.group("patient_id"))
    return str(match.group(1)) if match.groups() else None


def inspect_image(image: Image.Image) -> dict[str, Any]:
    width, height = image.size
    image_format = image.format or image.get_format_mimetype() or "UNKNOWN"
    grayscale = image.convert("L")
    array = np.asarray(grayscale, dtype=np.float32) / 255.0
    return {
        "width": int(width),
        "height": int(height),
        "mode": image.mode,
        "image_format": str(image_format),
        "mean_intensity": float(array.mean()),
        "std_intensity": float(array.std()),
        "center_mean_intensity": float(_center_crop(array).mean()),
        "border_mean_intensity": float(_border_region(array).mean()),
        "average_hash": compute_average_hash(image),
    }


def inspect_image_path(file_path: Path) -> dict[str, Any]:
    with Image.open(file_path) as image:
        metadata = inspect_image(image)
    metadata["file_size_bytes"] = int(file_path.stat().st_size)
    metadata["sha256"] = compute_sha256(file_path)
    return metadata


def validate_mri_image_metadata(
    metadata: dict[str, Any],
    *,
    min_image_size: tuple[int, int] = DEFAULT_MIN_IMAGE_SIZE,
    aspect_ratio_range: tuple[float, float] = DEFAULT_ASPECT_RATIO_RANGE,
    min_stddev: float = DEFAULT_MIN_STDDEV,
    min_center_border_diff: float = DEFAULT_MIN_CENTER_BORDER_DIFF,
) -> ValidationFeedback:
    width = int(metadata["width"])
    height = int(metadata["height"])
    image_format = str(metadata.get("image_format", "UNKNOWN")).upper()
    std_intensity = float(metadata.get("std_intensity", 0.0))
    center_mean = float(metadata.get("center_mean_intensity", 0.0))
    border_mean = float(metadata.get("border_mean_intensity", 0.0))
    aspect_ratio = width / height if height else 0.0

    if image_format not in SUPPORTED_IMAGE_FORMATS:
        return ValidationFeedback(
            passed=False,
            error_code="unsupported_file_type",
            message=f"Unsupported image format: {image_format}.",
            user_action="Please upload a PNG, JPG, or JPEG brain MRI image.",
            details={"image_format": image_format},
        )

    min_width, min_height = min_image_size
    if width < min_width or height < min_height:
        return ValidationFeedback(
            passed=False,
            error_code="image_too_small",
            message=f"Image too small. Uploaded image is {width} x {height}.",
            user_action=f"Please upload an MRI image at least {min_width} x {min_height} pixels.",
            details={
                "uploaded_width": width,
                "uploaded_height": height,
                "minimum_width": min_width,
                "minimum_height": min_height,
            },
        )

    min_ratio, max_ratio = aspect_ratio_range
    if not (min_ratio <= aspect_ratio <= max_ratio):
        return ValidationFeedback(
            passed=False,
            error_code="wrong_aspect_ratio",
            message=f"Unexpected image aspect ratio: {aspect_ratio:.2f}.",
            user_action="Please upload the original MRI image without stretching or unusual cropping.",
            details={"aspect_ratio": aspect_ratio, "minimum_ratio": min_ratio, "maximum_ratio": max_ratio},
        )

    if std_intensity < min_stddev:
        return ValidationFeedback(
            passed=False,
            error_code="low_signal_image",
            message="The uploaded image appears too blank or too uniform.",
            user_action="Please upload a clearer brain MRI image with visible anatomical structure.",
            details={"std_intensity": std_intensity, "minimum_std_intensity": min_stddev},
        )

    if center_mean - border_mean < min_center_border_diff:
        return ValidationFeedback(
            passed=False,
            error_code="non_mri_like_image",
            message="The uploaded image does not appear to be a brain MRI.",
            user_action="Please upload a valid brain MRI image in PNG, JPG, or JPEG format.",
            details={
                "center_mean_intensity": center_mean,
                "border_mean_intensity": border_mean,
                "minimum_center_border_diff": min_center_border_diff,
            },
        )

    return ValidationFeedback(passed=True, details={"width": width, "height": height})


def validate_mri_image_bytes(
    image_bytes: bytes,
    *,
    min_image_size: tuple[int, int] = DEFAULT_MIN_IMAGE_SIZE,
    aspect_ratio_range: tuple[float, float] = DEFAULT_ASPECT_RATIO_RANGE,
    min_stddev: float = DEFAULT_MIN_STDDEV,
    min_center_border_diff: float = DEFAULT_MIN_CENTER_BORDER_DIFF,
) -> tuple[ValidationFeedback, dict[str, Any] | None]:
    if not image_bytes:
        return (
            ValidationFeedback(
                passed=False,
                error_code="empty_upload",
                message="Uploaded file is empty.",
                user_action="Please choose an MRI image file and try again.",
                details={},
            ),
            None,
        )

    try:
        from io import BytesIO

        with Image.open(BytesIO(image_bytes)) as image:
            metadata = inspect_image(image)
    except UnidentifiedImageError:
        return (
            ValidationFeedback(
                passed=False,
                error_code="corrupted_image",
                message="The uploaded file could not be read as a valid image.",
                user_action="Please upload a valid MRI image in PNG, JPG, or JPEG format.",
                details={},
            ),
            None,
        )
    except OSError:
        return (
            ValidationFeedback(
                passed=False,
                error_code="corrupted_image",
                message="The uploaded file appears to be corrupted.",
                user_action="Please re-export the MRI image and upload it again.",
                details={},
            ),
            None,
        )

    return (
        validate_mri_image_metadata(
            metadata,
            min_image_size=min_image_size,
            aspect_ratio_range=aspect_ratio_range,
            min_stddev=min_stddev,
            min_center_border_diff=min_center_border_diff,
        ),
        metadata,
    )


def assign_duplicate_groups(records: list[dict[str, Any]], max_hash_distance: int = DEFAULT_DUPLICATE_HASH_DISTANCE) -> list[dict[str, Any]]:
    sha_groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        sha_groups.setdefault(str(record.get("sha256", "")), []).append(index)

    for group_number, indices in enumerate((group for group in sha_groups.values() if len(group) > 1), start=1):
        for index in indices:
            records[index]["duplicate_group_id"] = f"exact_duplicate_{group_number:05d}"

    near_group_number = 1
    for index, record in enumerate(records):
        if record.get("duplicate_group_id"):
            continue
        for other_index in range(index + 1, len(records)):
            other = records[other_index]
            if other.get("duplicate_group_id"):
                continue
            if record.get("label_name") != other.get("label_name"):
                continue
            distance = hamming_distance(str(record.get("average_hash", "")), str(other.get("average_hash", "")))
            if distance <= max_hash_distance:
                group_id = f"near_duplicate_{near_group_number:05d}"
                records[index]["duplicate_group_id"] = group_id
                other["duplicate_group_id"] = group_id
        if record.get("duplicate_group_id", "").startswith("near_duplicate_"):
            near_group_number += 1

    for record in records:
        record["duplicate_group_id"] = record.get("duplicate_group_id") or ""
        record["group_id"] = record.get("patient_id") or record["duplicate_group_id"] or str(record["sample_id"])
    return records


def summarize_duplicates(records: list[dict[str, Any]]) -> dict[str, Any]:
    exact_groups = {record["duplicate_group_id"] for record in records if str(record.get("duplicate_group_id", "")).startswith("exact_duplicate_")}
    near_groups = {record["duplicate_group_id"] for record in records if str(record.get("duplicate_group_id", "")).startswith("near_duplicate_")}
    return {
        "exact_duplicate_groups": len(exact_groups),
        "near_duplicate_groups": len(near_groups),
        "duplicate_rows": sum(1 for record in records if record.get("duplicate_group_id")),
    }


def _center_crop(array: np.ndarray) -> np.ndarray:
    height, width = array.shape
    height_margin = max(height // 4, 1)
    width_margin = max(width // 4, 1)
    return array[height_margin : height - height_margin, width_margin : width - width_margin]


def _border_region(array: np.ndarray) -> np.ndarray:
    height, width = array.shape
    border = max(min(height, width) // 8, 1)
    top = array[:border, :]
    bottom = array[-border:, :]
    left = array[:, :border]
    right = array[:, -border:]
    return np.concatenate([top.flatten(), bottom.flatten(), left.flatten(), right.flatten()])