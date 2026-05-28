import base64
import re
from typing import Any


DEFAULT_SIZE = "1024x1024"
DEFAULT_QUALITY = "medium"
DEFAULT_OUTPUT_FORMAT = "jpeg"
DEFAULT_OUTPUT_COMPRESSION = 85
DEFAULT_BACKGROUND = "opaque"

MODEL = "gpt-image-2"

QUALITY_VALUES = {"low", "medium", "high", "auto"}
OUTPUT_FORMAT_VALUES = {"auto", "png", "jpeg", "webp"}
BACKGROUND_VALUES = {"auto", "opaque"}
MODERATION_VALUES = {"auto", "low"}

SIZE_PATTERN = re.compile(r"^(\d+)x(\d+)$")
MIN_PIXELS = 655_360
MAX_PIXELS = 8_294_400
MAX_EDGE = 3_840
MAX_RATIO = 3


class ParameterError(ValueError):
    """Raised when a tool parameter cannot be sent to the OpenAI image API."""


def build_edit_args(parameters: dict[str, Any], *, include_image: bool = True) -> dict[str, Any]:
    prompt = _required_string(parameters.get("prompt"), "prompt")
    args: dict[str, Any] = {
        "model": MODEL,
        "prompt": prompt,
    }

    if include_image and not parameters.get("image"):
        raise ParameterError("Input image file is required.")

    size = _string_choice(parameters.get("size", DEFAULT_SIZE), "size")
    if size != "auto":
        _validate_size(size)
        args["size"] = size

    quality = _string_choice(parameters.get("quality", DEFAULT_QUALITY), "quality")
    if quality not in QUALITY_VALUES:
        raise ParameterError("Invalid quality. Choose low, medium, high, or auto.")
    if quality != "auto":
        args["quality"] = quality

    output_format = _string_choice(
        parameters.get("output_format", DEFAULT_OUTPUT_FORMAT),
        "output_format",
    )
    if output_format not in OUTPUT_FORMAT_VALUES:
        raise ParameterError("Invalid output_format. Choose auto, png, jpeg, or webp.")
    if output_format != "auto":
        args["output_format"] = output_format

    output_compression = parameters.get("output_compression", DEFAULT_OUTPUT_COMPRESSION)
    if output_compression not in (None, ""):
        compression = _bounded_int(
            output_compression,
            "output_compression",
            minimum=0,
            maximum=100,
        )
        if output_format in {"jpeg", "webp"}:
            args["output_compression"] = compression

    background = _string_choice(parameters.get("background", DEFAULT_BACKGROUND), "background")
    if background == "transparent":
        raise ParameterError("gpt-image-2 does not support transparent background.")
    if background not in BACKGROUND_VALUES:
        raise ParameterError("Invalid background. Choose auto or opaque.")
    if background != "auto":
        args["background"] = background

    moderation = _string_choice(parameters.get("moderation", "auto"), "moderation")
    if moderation not in MODERATION_VALUES:
        raise ParameterError("Invalid moderation. Choose auto or low.")
    if moderation != "auto":
        args["moderation"] = moderation

    n = _bounded_int(parameters.get("n", 1), "n", minimum=1, maximum=10)
    args["n"] = n

    stream = _to_bool(parameters.get("stream", False), "stream")
    if stream:
        args["stream"] = True

    user = parameters.get("user")
    if user not in (None, ""):
        if not isinstance(user, str):
            raise ParameterError("Invalid user. Provide a string identifier.")
        user = user.strip()
        if user:
            args["user"] = user

    return args


def collect_streamed_final_images(
    stream: Any,
    edit_args: dict[str, Any],
    *,
    elapsed_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    final_usage: dict[str, Any] = {}
    partial_image_count = 0
    final_images: list[dict[str, Any]] = []

    for event in stream:
        payload = extract_event_payload(event)
        event_type = payload.get("type")

        if payload.get("usage"):
            final_usage = usage_to_dict(payload["usage"])

        if event_type == "image_edit.partial_image":
            partial_image_count += 1
            continue
        if event_type != "image_edit.completed":
            continue

        b64_json = payload.get("b64_json")
        if not b64_json:
            continue

        mime_type, blob = decode_image_payload(b64_json)
        mime_type = mime_from_output_format(edit_args, payload, mime_type)
        metadata: dict[str, Any] = {
            "mime_type": mime_type,
            "model": MODEL,
            "operation": "edit",
            "stream": True,
            "elapsed_seconds": elapsed_seconds,
        }
        if final_usage:
            metadata["token_usage"] = final_usage

        final_images.append({"blob": blob, "metadata": metadata})

    summary: dict[str, Any] = {
        "model": MODEL,
        "operation": "edit",
        "stream": True,
        "image_count": len(final_images),
        "partial_image_count": partial_image_count,
        "elapsed_seconds": elapsed_seconds,
    }
    if final_usage:
        summary["token_usage"] = final_usage

    return final_images, summary


def decode_image_payload(base64_image: str) -> tuple[str, bytes]:
    if not isinstance(base64_image, str) or not base64_image:
        raise ParameterError("Image payload is empty.")

    if not base64_image.startswith("data:image"):
        return "image/png", base64.b64decode(base64_image)

    try:
        mime_type = base64_image.split(";", 1)[0].split(":", 1)[1]
        image_data_base64 = base64_image.split(",", 1)[1]
    except IndexError as error:
        raise ParameterError("Invalid image data URL.") from error

    return mime_type, base64.b64decode(image_data_base64)


def extract_event_payload(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        raw_payload = event
    elif hasattr(event, "model_dump"):
        raw_payload = event.model_dump()
    else:
        raw_payload = {
            key: getattr(event, key)
            for key in (
                "type",
                "b64_json",
                "partial_image_index",
                "usage",
                "background",
                "output_format",
                "quality",
                "size",
            )
            if hasattr(event, key)
        }

    return {key: value for key, value in raw_payload.items() if value is not None}


def response_metadata(
    *,
    response: Any,
    model: str,
    operation: str,
    image_count: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "model": model,
        "operation": operation,
        "image_count": image_count,
    }

    request_id = _get_attr(response, "_request_id") or _get_attr(response, "request_id")
    if request_id:
        metadata["request_id"] = request_id

    for key in ("background", "output_format", "quality", "size"):
        value = _get_attr(response, key)
        if value is not None:
            metadata[key] = value

    usage = usage_to_dict(_get_attr(response, "usage"))
    if usage:
        metadata["token_usage"] = usage

    return metadata


def usage_to_dict(usage: Any) -> dict[str, Any]:
    if not usage:
        return {}
    if isinstance(usage, dict):
        return {key: value for key, value in usage.items() if value is not None}

    usage_dict: dict[str, Any] = {}
    for key in ("total_tokens", "input_tokens", "output_tokens"):
        value = _get_attr(usage, key)
        if value is not None:
            usage_dict[key] = value

    input_details = _get_attr(usage, "input_tokens_details")
    input_details_dict = _token_details_to_dict(input_details)
    if input_details_dict:
        usage_dict["input_tokens_details"] = input_details_dict

    output_details = _get_attr(usage, "output_tokens_details")
    output_details_dict = _token_details_to_dict(output_details)
    if output_details_dict:
        usage_dict["output_tokens_details"] = output_details_dict

    return usage_dict


def normalize_openai_base_url(base_url: Any) -> str | None:
    if not base_url:
        return None
    if not isinstance(base_url, str):
        raise ParameterError("Invalid OpenAI base URL. Provide a string URL.")

    normalized = base_url.strip().rstrip("/")
    if not normalized:
        return None
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def mime_from_output_format(
    edit_args: dict[str, Any],
    response_or_payload: Any,
    fallback: str,
) -> str:
    output_format = edit_args.get("output_format")
    if not output_format and isinstance(response_or_payload, dict):
        output_format = response_or_payload.get("output_format")
    if not output_format:
        output_format = _get_attr(response_or_payload, "output_format")
    if output_format in {"png", "jpeg", "webp"}:
        return f"image/{output_format}"
    return fallback


def _token_details_to_dict(details: Any) -> dict[str, Any]:
    if not details:
        return {}
    if isinstance(details, dict):
        return {key: value for key, value in details.items() if value is not None}

    details_dict: dict[str, Any] = {}
    for key in ("text_tokens", "image_tokens"):
        value = _get_attr(details, key)
        if value is not None:
            details_dict[key] = value
    return details_dict


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParameterError(f"{name} is required.")
    return value.strip()


def _string_choice(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParameterError(f"{name} must be a non-empty string.")
    return value.strip()


def _bounded_int(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    try:
        int_value = int(value)
    except (TypeError, ValueError) as error:
        raise ParameterError(f"Invalid {name}. Choose an integer between {minimum} and {maximum}.") from error

    if not minimum <= int_value <= maximum:
        raise ParameterError(f"Invalid {name}. Choose an integer between {minimum} and {maximum}.")
    return int_value


def _to_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    if value in (0, None):
        return False
    if value == 1:
        return True
    raise ParameterError(f"Invalid {name}. Choose true or false.")


def _validate_size(size: str) -> None:
    match = SIZE_PATTERN.fullmatch(size)
    if not match:
        raise ParameterError("Invalid size. Use auto or a WxH string such as 1024x1024.")

    width = int(match.group(1))
    height = int(match.group(2))
    if width % 16 != 0 or height % 16 != 0:
        raise ParameterError("Invalid size. Width and height must each be a multiple of 16.")
    if width > MAX_EDGE or height > MAX_EDGE:
        raise ParameterError("Invalid size. Maximum edge length is 3840px.")

    total_pixels = width * height
    if total_pixels < MIN_PIXELS or total_pixels > MAX_PIXELS:
        raise ParameterError("Invalid size. Total pixels must be between 655360 and 8294400.")

    long_edge = max(width, height)
    short_edge = min(width, height)
    if long_edge / short_edge > MAX_RATIO:
        raise ParameterError("Invalid size. Long edge to short edge ratio must not exceed 3:1.")


def _get_attr(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
