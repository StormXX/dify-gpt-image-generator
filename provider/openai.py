from typing import Any

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError
from openai import OpenAI

from tools.gpt_image_2_edit_args import normalize_openai_base_url


class OpenAIProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        api_key = credentials.get("openai_api_key")
        if not api_key:
            raise ToolProviderCredentialValidationError("OpenAI API key is required.")

        try:
            client = OpenAI(
                api_key=api_key,
                base_url=normalize_openai_base_url(credentials.get("openai_base_url")),
                organization=credentials.get("openai_organization_id") or None,
            )
            client.models.list()
        except Exception as error:
            raise ToolProviderCredentialValidationError(str(error)) from error
