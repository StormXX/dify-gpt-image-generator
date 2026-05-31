from typing import Any

from tools.gpt_image_2_edit_args import (
    DEFAULT_BACKGROUND,
    DEFAULT_OUTPUT_COMPRESSION,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_QUALITY,
    DEFAULT_SIZE,
    MODEL,
    BACKGROUND_VALUES,
    OUTPUT_FORMAT_VALUES,
    QUALITY_VALUES,
    ParameterError,
    _bounded_int,
    _required_string,
    _string_choice,
    _to_bool,
    _validate_size,
)


def build_generate_args(parameters: dict[str, Any]) -> dict[str, Any]:
    prompt = _required_string(parameters.get("prompt"), "prompt")
    args: dict[str, Any] = {
        "model": MODEL,
        "prompt": prompt,
    }

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
