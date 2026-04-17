from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Weaviate Cloud
    WEAVIATE_URL: str
    WEAVIATE_API_KEY: str
    WEAVIATE_INDEX_NAME: str = "LoanIQ"

    # Retrieval settings
    DENSE_TOP_K: int = 10          # dense retriever fetch count
    BM25_TOP_K: int = 10           # BM25 retriever fetch count
    RRF_TOP_K: int = 10            # after RRF fusion
    MMR_TOP_K: int = 5             # after MMR diversity filter
    CROSS_ENCODER_TOP_K: int = 3   # final docs after cross-encoder rerank
    MMR_LAMBDA: float = 0.5        # diversity vs relevance balance (0=diverse, 1=relevant)
    LANGCHAIN_TRACING_V2:str
    LANGCHAIN_API_KEY:str
    LANGCHAIN_PROJECT:str

    class Config:
        env_file = ".env"


settings = Settings()
