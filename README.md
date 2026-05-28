# GPT Image Generator Dify Plugin

Custom Dify tool plugin for OpenAI `gpt-image-2` image editing.

The first tool is `gpt_image_2_edit_plus`, based on Dify's official OpenAI image edit
tool but with these additional controls:

- `output_format`: `jpeg`, `webp`, `png`, or `auto`
- `output_compression`: `0` to `100`, used only for JPEG/WebP
- `background`: `opaque` or `auto`; transparent is intentionally blocked for `gpt-image-2`
- `stream`, while still emitting only final completed images to Dify
- `user` for OpenAI abuse monitoring

Latency-oriented defaults are:

- `size=1024x1024`
- `quality=medium`
- `output_format=jpeg`
- `output_compression=85`
- `background=opaque`
- `n=1`

## Development

Run unit tests for the API argument builder:

```bash
python3 -m unittest tests/test_gpt_image_2_edit_args.py
```

Run the plugin in Dify remote debug mode after creating a local `.env` from
`.env.example`:

```bash
python -m main
```

Package with the Dify CLI:

```bash
cd ..
dify plugin package ./gpt-image-generator
```

Do not commit OpenAI or Dify API keys. Provider credentials are configured in
Dify and are not stored in this repository.
