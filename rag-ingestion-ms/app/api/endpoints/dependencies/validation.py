
from typing import List

from fastapi import Depends, File, HTTPException, UploadFile, status
from app.core.config import Settings, settings
from loguru import logger


SUPPORTED_CONTENT_TYPES = [
    "text/plain",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/epub+zip",
    "application/msword"
]

def get_settings() -> Settings:
    return settings

async def validate_files_payload(
    files: List[UploadFile] = File(...),
    settings: Settings = Depends(get_settings)
) -> List[UploadFile]:

    if len(files) > settings.MAX_FILES_COUNT:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Too many files. Maximum allowed is {settings.MAX_FILES_COUNT}."
        )

    for file in files:
        if file.size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File '{file.filename}' exceeds the size limit of {settings.MAX_FILE_SIZE_MB}MB."
            )
        if file.content_type not in SUPPORTED_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"File '{file.filename}' has an unsupported type: {file.content_type}."
            )
    
    logger.info(f"Payload validation successful for {len(files)} files.")
    return files