""" All settings for the app live here, in one place"""

import os
from dotenv import load_dotenv
load_dotenv()


## env var

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")


## Define path - data / vectore store

DATA_FILE_PATH = os.path.join('data','hr_policy.txt')


## vectore stores

VECTOR_STORE_PATH = os.path.join("data","faiss_index")



## Models
## LLM and Embeding model

LLM_MODEL_NAME="openai/gpt-oss-120b"
EMBEDDING_MODEL_NAME="jina-embeddings-v2-base-en"


## CHUNK / TEXT SPLITTING CONFIG

CHUNK_SIZE= 500
CHUNK_OVERLAP=50


# RETRIVAL RESULTS
TOP_K_RESULTS =3

## SYSTEM INSTRUCTION

SYSTEM_PROMPT=(
"You are a friendly HR assistent.Always use the search_hr_policy tool to look up"
"facts before answering.IF the answer isn't in the search results, say you don't know "
        "instead of guessing."
        
)

def check_api_keys()-> None:
    """Stop early with a clear messages if a required API key is missing"""
    if not GROQ_API_KEY:
        raise ValueError("Missing groq api key . please add it to env")
    if not JINA_API_KEY:
        raise ValueError("Missing jira api key. lease add it to env")
    