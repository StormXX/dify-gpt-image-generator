import io
import logging
import time
from collections.abc import Generator, Iterable
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin.file.file import File
from openai import OpenAI

from tools.gpt_image_2_edit_args import (
    MODEL,
    ParameterError,
    build_edit_args,
    collect_streamed_final_images,
    decode_image_payload,
    mime_from_output_format,
    normalize_openai_base_url,
    response_metadata,
)

logger = logging.getLogger(__name__)


class GPTImage2EditTool(Tool):
    def _invoke(
        self, tool_parameters: dict
    ) -> Generator[ToolInvokeMessage, None, None]:
        started_at = time.monotonic()
        image_files: list[io.BytesIO] = []
        mask_file: io.BytesIO | None = None

        try:
            edit_args = build_edit_args(tool_parameters, include_image=False)
            image_files = _open_image_files(tool_parameters.get("image"))
            edit_args["image"] = image_files if len(image_files) > 1 else image_files[0]

            mask = tool_parameters.get("mask")
            if mask:
                mask_file = _open_file(mask, "mask_image.png", "Mask image must be a valid file.")
                edit_args["mask"] = mask_file

            client = self._client()
            logger.info(
                "gpt-image-2 edit invoked: prompt_len=%s images=%s has_mask=%s size=%s quality=%s "
                "format=%s compression=%s background=%s stream=%s n=%s",
                len(edit_args["prompt"]),
                len(image_files),
                bool(mask_file),
                edit_args.get("size", "auto"),
                edit_args.get("quality", "auto"),
                edit_args.get("output_format", "auto"),
                edit_args.get("output_compression", "default"),
                edit_args.get("background", "auto"),
                edit_args.get("stream", False),
                edit_args.get("n", 1),
            )

            response = client.images.edit(**edit_args)

            if edit_args.get("stream"):
                yield from self._yield_stream_messages(response, edit_args, started_at)
            else:
                yield from self._yield_response_messages(response, edit_args, started_at)
        except ParameterError as error:
            yield self.create_text_message(f"Invalid parameter: {error}")
        except Exception as error:
            logger.exception("gpt-image-2 edit failed")
            yield self.create_text_message(f"Failed to edit image: {error}")
        finally:
            for file_obj in image_files:
                file_obj.close()
            if mask_file:
                mask_file.close()

    def _client(self) -> OpenAI:
        return OpenAI(
            api_key=self.runtime.credentials["openai_api_key"],
            base_url=normalize_openai_base_url(self.runtime.credentials.get("openai_base_url")),
            organization=self.runtime.credentials.get("openai_organization_id") or None,
        )

    def _yield_response_messages(
        self,
        response: Any,
        edit_args: dict[str, Any],
        started_at: float,
    ) -> Generator[ToolInvokeMessage, None, None]:
        image_items = list(getattr(response, "data", []) or [])
        metadata = response_metadata(
            response=response,
            model=MODEL,
            operation="edit",
            image_count=len(image_items),
        )
        metadata["elapsed_seconds"] = round(time.monotonic() - started_at, 3)

        yielded_images = 0
        for image_data in image_items:
            b64_json = getattr(image_data, "b64_json", None)
            if not b64_json:
                continue
            mime_type, blob = decode_image_payload(b64_json)
            mime_type = mime_from_output_format(edit_args, response, mime_type)
            yielded_images += 1
            yield self.create_blob_message(
                blob=blob,
                meta={"mime_type": mime_type, **metadata},
            )

        if yielded_images == 0:
            yield self.create_text_message("OpenAI returned no image data.")
            return

        yield self.create_json_message({"data": [{**metadata, "image_count": yielded_images}]})

    def _yield_stream_messages(
        self,
        stream: Iterable[Any],
        edit_args: dict[str, Any],
        started_at: float,
    ) -> Generator[ToolInvokeMessage, None, None]:
        final_images, summary = collect_streamed_final_images(
            stream,
            edit_args,
            elapsed_seconds=round(time.monotonic() - started_at, 3),
        )

        if not final_images:
            yield self.create_text_message("OpenAI streaming response returned no image data.")
            return

        for image in final_images:
            yield self.create_blob_message(blob=image["blob"], meta=image["metadata"])

        yield self.create_json_message({"data": [summary]})


def _open_image_files(image: Any) -> list[io.BytesIO]:
    if not image:
        raise ParameterError("Input image file is required.")

    if isinstance(image, list):
        if not image:
            raise ParameterError("Input image file is required.")
        return [
            _open_file(file_obj, f"input_image_{index + 1}.png", "All input images must be valid files.")
            for index, file_obj in enumerate(image)
        ]

    return [_open_file(image, "input_image.png", "Input image must be a valid file.")]


def _open_file(file_obj: Any, fallback_name: str, error_message: str) -> io.BytesIO:
    if not isinstance(file_obj, File):
        raise ParameterError(error_message)
    blob = getattr(file_obj, "blob", None)
    if not isinstance(blob, bytes) or not blob:
        raise ParameterError(error_message)

    buffer = io.BytesIO(blob)
    buffer.name = getattr(file_obj, "filename", None) or fallback_name
    return buffer
