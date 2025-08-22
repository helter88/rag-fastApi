from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "RAG Document Manager Microservice"
    LOG_LEVEL: str = "INFO"
    MAX_FILE_SIZE_MB: int = 10
    MAX_FILES_COUNT: int = 5

    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    RAG_COLLECTION_NAME: str = "rag_collection"
    HUGGINGFACEHUB_API_TOKEN: str
    HF_EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    TEXT_SPLITTER_CHUNK_SIZE: int = 1500
    TEXT_SPLITTER_CHUNK_OVERLAP: int = 200

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

settings = Settings()