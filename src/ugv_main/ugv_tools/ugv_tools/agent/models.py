from langchain_core.language_models.chat_models import BaseChatModel

class Models:
    def __init__(self):
        self.models = {
            "llama-3.3-70b": self._get_llama_3_3_70b,
        }

    def _get_llama_3_3_70b(self):
        from langchain_aws import ChatBedrockConverse
        import boto3

        bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
        llm = ChatBedrockConverse(
            model="us.meta.llama3-3-70b-instruct-v1:0",
            temperature=0,
            client=bedrock_client,
        )

        return llm

    def get_model(self, model_name) -> BaseChatModel:
        if model_name in self.models:
            return self.models[model_name]()
        else:
            raise ValueError(f"Model {model_name} not found.")
