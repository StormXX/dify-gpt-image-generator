import logging
import time
from collections.abc import Generator, Iterable
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from openai import OpenAI

from tools.gpt_image_2_edit_args import (
    MODEL,
    ParameterError,
    collect_streamed_final_images,
    decode_image_payload,
    mime_from_output_format,
    normalize_openai_base_url,
    response_metadata,
)
from tools.gpt_image_2_generate_args import build_generate_args

logger = logging.getLogger(__name__)


class GPTImage2GenerateTool(Tool):
    def _invoke(
        self, tool_parameters: dict
    ) -> Generator[ToolInvokeMessage, None, None]:
        started_at = time.monotonic()

        try:
            generate_args = build_generate_args(tool_parameters)
            client = self._client()
            logger.info(
                "gpt-image-2 generate invoked: prompt_len=%s size=%s quality=%s "
                "format=%s compression=%s background=%s stream=%s n=%s",
                len(generate_args["prompt"]),
                generate_args.get("size", "auto"),
                generate_args.get("quality", "auto"),
                generate_args.get("output_format", "auto"),
                generate_args.get("output_compression", "default"),
                generate_args.get("background", "auto"),
                generate_args.get("stream", False),
                generate_args.get("n", 1),
            )

            response = client.images.generate(**generate_args)

            if generate_args.get("stream"):
                yield from self._yield_stream_messages(response, generate_args, started_at)
            else:
                yield from self._yield_response_messages(response, generate_args, started_at)
        except ParameterError as error:
            yield self.create_text_message(f"Invalid parameter: {error}")
        except Exception as error:
            logger.exception("gpt-image-2 generate failed")
            yield self.create_text_message(f"Failed to generate image: {error}")

    def _client(self) -> OpenAI:
        return OpenAI(
            api_key=self.runtime.credentials["openai_api_key"],
            base_url=normalize_openai_base_url(self.runtime.credentials.get("openai_base_url")),
            organization=self.runtime.credentials.get("openai_organization_id") or None,
        )

    def _yield_response_messages(
        self,
        response: Any,
        generate_args: dict[str, Any],
        started_at: float,
    ) -> Generator[ToolInvokeMessage, None, None]:
        image_items = list(getattr(response, "data", []) or [])
        metadata = response_metadata(
            response=response,
            model=MODEL,
            operation="generate",
            image_count=len(image_items),
        )
        metadata["elapsed_seconds"] = round(time.monotonic() - started_at, 3)

        yielded_images = 0
        for image_data in image_items:
            b64_json = getattr(image_data, "b64_json", None)
            if not b64_json:
                continue
            mime_type, blob = decode_image_payload(b64_json)
            mime_type = mime_from_output_format(generate_args, response, mime_type)
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
        generate_args: dict[str, Any],
        started_at: float,
    ) -> Generator[ToolInvokeMessage, None, None]:
        final_images, summary = collect_streamed_final_images(
            stream,
            generate_args,
            elapsed_seconds=round(time.monotonic() - started_at, 3),
            operation="generate",
        )

        if not final_images:
            yield self.create_text_message("OpenAI streaming response returned no image data.")
            return

        for image in final_images:
            yield self.create_blob_message(blob=image["blob"], meta=image["metadata"])

        yield self.create_json_message({"data": [summary]})
