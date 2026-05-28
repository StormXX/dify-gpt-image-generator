# Privacy

This plugin sends user-provided prompts, input images, optional mask images, and
selected generation parameters to the configured OpenAI-compatible Images API
endpoint.

The plugin does not persist prompts, images, masks, generated images, provider
credentials, or API keys. Runtime logs intentionally avoid full prompt text,
image bytes, URLs, and credential values.

OpenAI API keys are stored as Dify provider credentials. Do not place keys in
plugin files, `.env`, tests, or documentation.
