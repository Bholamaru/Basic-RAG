"""Step 6: connect to the llm (the "brain" of the assistant)"""

from langchain_groq import ChatGroq
from hr_assistant import config

def get_llm():
    """Return a groq chat model. Reads GROQ_API_KEY From the env"""
    return ChatGroq(model=config.LLM_MODEL_NAME,temperature=0)