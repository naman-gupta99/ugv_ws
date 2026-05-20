from langchain_core.language_models.chat_models import BaseChatModel
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv


def _load_env_file():
    """Find and load .env file from multiple possible locations."""
    file_path = Path(__file__)

    possible_paths = [
        file_path.parent / ".env",
        file_path.parents[5].parent.parent / "src" / "ugv_main" / "ugv_tools" / "ugv_tools" / "agent" / ".env",
        Path.home() / ".config" / "ugv" / ".env",
        Path.cwd() / ".env",
    ]

    for env_path in possible_paths:
        if env_path.exists():
            print(f"[DEBUG] Loading .env from: {env_path}", file=sys.stderr)
            load_dotenv(env_path, override=True)
            return True

    print("[DEBUG] No .env file found. Checked:", file=sys.stderr)
    for path in possible_paths:
        print(f"  - {path}", file=sys.stderr)
    return False


_load_env_file()


class Models:
    GREEDY_MODEL = "greedy"
    CODE_AGENT_MODEL = "code"
    CODE_AGENT_MODEL_PREFIX = "code-"
    DEFAULT_MODEL = "gemini-2.5-pro"

    def __init__(self):
        # These models are available for this API key in Google AI Studio.
        self.ai_studio_models = [
            "gemma-4-26b-a4b-it",
            "gemma-4-31b-it",
        ]
        # No Vertex-only models in the current temporary experiment set.
        # self.vertex_only_models = []
        self.vertex_only_models = ["gemini-2.5-flash-lite",
            "gemini-2.5-pro"]
        self.code_agent_models = [
            f"{self.CODE_AGENT_MODEL_PREFIX}{model_name}"
            for model_name in (self.ai_studio_models + self.vertex_only_models)
        ]
        # Non-react agent modes; handled by LlmPtCtrl.
        self.controller_modes = [
            *self.code_agent_models,
            # self.GREEDY_MODEL,
        ]
        self.model_names = (
            self.controller_modes
        )

    def _get_google_model(self, model_name):
        from langchain_google_genai import ChatGoogleGenerativeAI

        # 1. Use AI Studio for Gemma 3 and Gemini (MaaS - No Deployment Needed)
        if model_name in self.ai_studio_models:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is required for AI Studio models.")
            if api_key.startswith(("AQ.", "ya29.")):
                raise ValueError(
                    "GOOGLE_API_KEY appears to be an OAuth access token. "
                    "AI Studio models need an API key from Google AI Studio."
                )
            
            print(f"[DEBUG] Using Google AI Studio for {model_name}")
            return ChatGoogleGenerativeAI(
                model=model_name,
                api_key=api_key,
                temperature=0.7,
            )

        # 2. Use Vertex AI for models requiring dedicated infrastructure.
        else:
            print(f"[DEBUG] Using Vertex AI for {model_name} (Requires Deployment)")
            self._verify_langsmith_config()
            project, location, _ = self._verify_vertex_config()
            return ChatGoogleGenerativeAI(
                model=model_name,
                vertexai=True,
                project=project,
                location=location,
            )

    def _verify_langsmith_config(self):
        """Verify LangSmith environment variables are properly configured."""
        tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        endpoint = os.getenv("LANGSMITH_ENDPOINT")
        api_key = os.getenv("LANGSMITH_API_KEY")
        project = os.getenv("LANGSMITH_PROJECT")

        print("[LangSmith Config Check]")
        print(f"  Tracing Enabled: {tracing}")
        print(f"  Endpoint: {endpoint}")
        print(f"  API Key Present: {'Yes' if api_key else 'No'}")
        print(f"  API Key Valid: {api_key.startswith('lsv2_pt_') if api_key else 'N/A'}")
        print(f"  Project: {project}")

        if not tracing:
            print("  WARNING: LANGSMITH_TRACING is not enabled!")
        if not api_key:
            print("  ERROR: LANGSMITH_API_KEY not found!")
        if not endpoint:
            print("  ERROR: LANGSMITH_ENDPOINT not found!")

    def _verify_vertex_config(self):
        """Verify and enforce Vertex AI configuration."""
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        model_name = os.getenv("VERTEX_MODEL", self.DEFAULT_MODEL)

        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            bundled_key = Path(__file__).parent / "arctic-defender-474101-m3-c84b6d213b37.json"
            if bundled_key.exists():
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(bundled_key)

        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        print("[Google Model Config Check]")
        print("  Mode: Vertex AI")
        print(f"  Project: {project}")
        print(f"  Location: {location}")
        print(f"  Model: {model_name}")
        print(f"  Credentials Path: {credentials_path}")

        if not project:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT is required for Vertex AI mode. "
                "Set it in your .env (e.g. GOOGLE_CLOUD_PROJECT=arctic-defender-474101-m3)."
            )

        if not credentials_path:
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS is required for Vertex AI mode. "
                "Point it to your service account JSON key file."
            )

        if not Path(credentials_path).exists():
            raise ValueError(
                f"GOOGLE_APPLICATION_CREDENTIALS file not found: {credentials_path}"
            )

        try:
            with open(credentials_path, "r", encoding="utf-8") as f:
                sa_json = json.load(f)
            client_email = sa_json.get("client_email", "unknown")
            print(f"  Service Account: {client_email}")
        except Exception:
            print("  Service Account: unable to parse JSON key metadata")

        return project, location, model_name

    def get_model(self, model_name) -> BaseChatModel:
        # Allow controller-mode names (e.g., "code-gemini-2.5-pro") by mapping
        # them to the underlying LLM model name. This keeps controller-style
        # naming while still returning a concrete LLM instance when requested.
        if model_name in self.controller_modes or model_name.startswith(self.CODE_AGENT_MODEL_PREFIX):
            if model_name.startswith(self.CODE_AGENT_MODEL_PREFIX):
                underlying = model_name[len(self.CODE_AGENT_MODEL_PREFIX):]
            else:
                # Fallback: try to derive the underlying name by stripping prefix
                underlying = model_name
            return self._get_google_model(underlying)
        # Also accept concrete LLM names directly (AI Studio or Vertex lists).
        if model_name in (self.ai_studio_models + self.vertex_only_models):
            return self._get_google_model(model_name)
        if model_name in self.model_names:
            return self._get_google_model(model_name)
        raise ValueError(f"Model {model_name} not found.")

    def available_model_names(self):
        return list(self.model_names)

    def llm_model_names(self):
        return list(self.ai_studio_models + self.vertex_only_models)

    def code_agent_model_names(self):
        return list(self.code_agent_models)
