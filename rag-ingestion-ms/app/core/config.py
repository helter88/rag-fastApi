from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Document Parser Microservice"
    LOG_LEVEL: str = "INFO"
    MAX_FILE_SIZE_MB: int = 10
    MAX_FILES_COUNT: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()