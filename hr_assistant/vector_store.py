"""Step 4: Store chunk embeddings in FAISS se we can search them later"""

import os
from langchain_community.vectorstores import FAISS
from hr_assistant import config
from hr_assistant.embeddings import get_embeddings_model


# build_vector_store

def build_vectore_store(chunks):
    """Embed every chunk and build a searchable Faiss index in memory.
    """
    embeddings_model = get_embeddings_model()
    return FAISS.from_documents(chunks,embeddings_model)

# save vectore store

def save_vectore_store(vector_store,path:str = config.VECTOR_STORE_PATH)-> None:
    """
    Save the FAISS index to disk so we don't have to rebuild it every time
    """
    vector_store.save_local(path)

def load_vectore_store(path: str= config.VECTOR_STORE_PATH):
    """Load a previously saved FAISS index from the disk """
    embeddings_model = get_embeddings_model()
    # allow_dangerous_deserialization is safe here because we only ever load
    # an index that this same app created and saved.
    return FAISS.load_local(path,embeddings_model,allow_dangerous_deserialization=True)

def vector_store_exists(path: str=config.VECTOR_STORE_PATH)-> bool:
    """Check if a saved Faiss index already exists on disk """
    return os.path.exists(os.path.join(path,"index.faiss"))


def get_retriever(vectore_store,k:int = config.TOP_K_RESULTS):
    """Turn a vectore store into a retriver that returns the top-k matching chunks."""
    return vectore_store.as_retriever(search_kwargs={"k":k})