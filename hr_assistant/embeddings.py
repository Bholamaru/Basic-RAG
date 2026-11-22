"""Step 3: turn text into numbers (vectors) using jina."""

from langchain_community.embeddings import JinaEmbeddings
from hr_assistant import config

def get_embeddings_model():
    """
    Return a Jina embeddings model. Reads Jina_api_key from the enviroments.
    """
    return JinaEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)