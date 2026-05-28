import base64
import unittest
from types import SimpleNamespace

from tools.gpt_image_2_edit_args import (
    ParameterError,
    build_edit_args,
    collect_streamed_final_images,
    decode_image_payload,
    extract_event_payload,
    normalize_openai_base_url,
    response_metadata,
)


class GPTImage2EditArgsTests(unittest.TestCase):
    def test_builds_latency_oriented_defaults(self):
        args = build_edit_args({"prompt": "make it cute"}, include_image=False)

        self.assertEqual(args["model"], "gpt-image-2")
        self.assertEqual(args["prompt"], "make it cute")
        self.assertEqual(args["size"], "1024x1024")
        self.assertEqual(args["quality"], "medium")
        self.assertEqual(args["output_format"], "jpeg")
        self.assertEqual(args["output_compression"], 85)
        self.assertEqual(args["background"], "opaque")
        self.assertEqual(args["n"], 1)
        self.assertNotIn("moderation", args)
        self.assertNotIn("stream", args)

    def test_omits_compression_when_output_format_cannot_use_it(self):
        args = build_edit_args(
            {
                "prompt": "edit it",
                "output_format": "png",
                "output_compression": 20,
            },
            include_image=False,
        )

        self.assertEqual(args["output_format"], "png")
        self.assertNotIn("output_compression", args)

    def test_rejects_transparent_background_for_gpt_image_2(self):
        with self.assertRaisesRegex(ParameterError, "transparent"):
            build_edit_args(
                {"prompt": "edit it", "background": "transparent"},
                include_image=False,
            )

    def test_streaming_uses_final_only_transport(self):
        args = build_edit_args(
            {
                "prompt": "edit it",
                "stream": True,
                "partial_images": 2,
            },
            include_image=False,
        )

        self.assertIs(args["stream"], True)
        self.assertNotIn("partial_images", args)

    def test_accepts_only_official_gpt_image_size_values(self):
        for size in ("auto", "1024x1024", "1536x1024", "1024x1536"):
            with self.subTest(size=size):
                args = build_edit_args(
                    {"prompt": "edit it", "size": size},
                    include_image=False,
                )
                if size == "auto":
                    self.assertNotIn("size", args)
                else:
                    self.assertEqual(args["size"], size)

        with self.assertRaisesRegex(ParameterError, "official GPT image sizes"):
            build_edit_args({"prompt": "edit it", "size": "2048x1152"}, include_image=False)

    def test_decodes_plain_and_data_url_images(self):
        raw = b"image bytes"

        self.assertEqual(decode_image_payload(base64.b64encode(raw).decode()), ("image/png", raw))
        self.assertEqual(
            decode_image_payload(
                "data:image/webp;base64," + base64.b64encode(raw).decode()
            ),
            ("image/webp", raw),
        )

    def test_extracts_object_and_dict_event_payloads(self):
        object_event = SimpleNamespace(
            type="image_edit.partial_image",
            b64_json="abc",
            partial_image_index=1,
        )
        dict_event = {"type": "image_edit.completed", "b64_json": "xyz", "usage": {"total_tokens": 5}}

        self.assertEqual(
            extract_event_payload(object_event),
            {"type": "image_edit.partial_image", "b64_json": "abc", "partial_image_index": 1},
        )
        self.assertEqual(
            extract_event_payload(dict_event),
            {"type": "image_edit.completed", "b64_json": "xyz", "usage": {"total_tokens": 5}},
        )

    def test_response_metadata_includes_parameters_usage_and_request_id(self):
        usage = SimpleNamespace(total_tokens=7, input_tokens=3, output_tokens=4)
        response = SimpleNamespace(
            usage=usage,
            _request_id="req_123",
            output_format="jpeg",
            quality="medium",
            size="1024x1024",
        )

        metadata = response_metadata(
            response=response,
            model="gpt-image-2",
            operation="edit",
            image_count=1,
        )

        self.assertEqual(metadata["model"], "gpt-image-2")
        self.assertEqual(metadata["operation"], "edit")
        self.assertEqual(metadata["image_count"], 1)
        self.assertEqual(metadata["request_id"], "req_123")
        self.assertEqual(metadata["output_format"], "jpeg")
        self.assertEqual(metadata["token_usage"]["total_tokens"], 7)

    def test_normalizes_base_url_without_duplicating_v1(self):
        self.assertIsNone(normalize_openai_base_url(None))
        self.assertEqual(
            normalize_openai_base_url("https://api.openai.com"),
            "https://api.openai.com/v1",
        )
        self.assertEqual(
            normalize_openai_base_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1",
        )

    def test_collects_only_final_images_from_stream_events(self):
        partial = base64.b64encode(b"partial").decode()
        final = base64.b64encode(b"final").decode()

        images, summary = collect_streamed_final_images(
            [
                {"type": "image_edit.partial_image", "b64_json": partial, "partial_image_index": 0},
                {
                    "type": "image_edit.completed",
                    "b64_json": final,
                    "usage": {"total_tokens": 9},
                },
            ],
            {"output_format": "jpeg"},
            elapsed_seconds=1.25,
        )

        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["blob"], b"final")
        self.assertEqual(images[0]["metadata"]["mime_type"], "image/jpeg")
        self.assertEqual(images[0]["metadata"]["token_usage"]["total_tokens"], 9)
        self.assertEqual(summary["image_count"], 1)
        self.assertEqual(summary["partial_image_count"], 1)
        self.assertEqual(summary["token_usage"]["total_tokens"], 9)


if __name__ == "__main__":
    unittest.main()
