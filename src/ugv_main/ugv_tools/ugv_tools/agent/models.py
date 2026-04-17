from langchain_core.language_models.chat_models import BaseChatModel
import os
from pathlib import Path
import json
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE importing LangChain
def _load_env_file():
    """Find and load .env file from multiple possible locations."""
    file_path = Path(__file__)
    
    possible_paths = [
        # Source directory (development) - direct
        file_path.parent / ".env",
        # Source directory via workspace (when installed)
        # Navigate back from installed location to find source tree
        file_path.parents[5].parent.parent / "src" / "ugv_main" / "ugv_tools" / "ugv_tools" / "agent" / ".env",
        # Home directory shortcut for deployment
        Path.home() / ".config" / "ugv" / ".env",
        # Current working directory (fall back)
        Path.cwd() / ".env",
    ]
    
    for env_path in possible_paths:
        if env_path.exists():
            print(f"[DEBUG] Loading .env from: {env_path}")
            load_dotenv(env_path)
            return True
    
    print("[DEBUG] No .env file found. Checked:")
    for path in possible_paths:
        print(f"  - {path}")
    return False

_load_env_file()

class Models:
    DEFAULT_VERTEX_MODEL = "gemini-2.5-flash"

    def __init__(self):
        self.models = {
            "llama-3.3-70b": self._get_llama_3_3_70b,
        }

    def _get_llama_3_3_70b(self):
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Verify LangSmith is configured
        self._verify_langsmith_config()

        project, location, model_name = self._verify_vertex_config()
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            vertexai=True,
            project=project,
            location=location,
        )
        return llm
    
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
        model_name = os.getenv("VERTEX_MODEL", self.DEFAULT_VERTEX_MODEL)

        # Auto-fill credentials path from the bundled file when not provided.
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            bundled_key = Path(__file__).parent / "arctic-defender-474101-m3-c84b6d213b37.json"
            if bundled_key.exists():
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(bundled_key)

        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        print("[Google Model Config Check]")
        print("  Mode: Vertex AI (forced)")
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
        if model_name in self.models:
            return self.models[model_name]()
        else:
            raise ValueError(f"Model {model_name} not found.")
