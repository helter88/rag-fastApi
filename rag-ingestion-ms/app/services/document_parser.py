import asyncio
from typing import IO, Callable, Coroutine, Dict, List

import docx
from fastapi import HTTPException, UploadFile, status
from loguru import logger
from pypdf import PdfReader
from ebooklib import epub


async def run_in_threadpool(func: Callable, *args, **kwargs) -> Coroutine:
    """Runs a synchronous function in a separate thread to avoid blocking the event loop."""
    return await asyncio.to_thread(func, *args, **kwargs)

async def _parse_txt(file: IO[bytes]) -> List[str]:
    content = file.read().decode('utf-8', errors='ignore')
    return content.splitlines()[:3]

async def _parse_docx(file: IO[bytes]) -> List[str]:
    def parse_sync(f):
        doc = docx.Document(f)
        return [p.text for p in doc.paragraphs if p.text][:3]
    return await run_in_threadpool(parse_sync, file)

async def _parse_pdf(file: IO[bytes]) -> List[str]:
    def parse_sync(f):
        reader = PdfReader(f)
        text_lines = []
        for page in reader.pages[:1]:
            text_lines.extend(page.extract_text().splitlines())
            if len(text_lines) >= 3:
                break
        return text_lines[:3]
    return await run_in_threadpool(parse_sync, file)

async def _parse_epub(file: IO[bytes]) -> List[str]:
    def parse_sync(f):
        book = epub.read_epub(f)
        lines = []
        for item in book.get_items_of_type(9):
            content = item.get_content().decode('utf-8', errors='ignore')
            lines.extend(content.splitlines())
            if len(lines) >= 3:
                break
        # Clean HTML tags
        import re
        clean_lines = [re.sub('<[^<]+?>', '', line).strip() for line in lines if line.strip()]
        return clean_lines[:3]
    return await run_in_threadpool(parse_sync, file)

async def _parse_doc(file: IO[bytes]) -> List[str]:
    logger.warning("Parsing .doc files is not fully supported and may yield poor results.")
    return ["Parsing for legacy .doc format is not implemented.", "Please convert to .docx for better results."]

PARSERS: Dict[str, Callable[[IO[bytes]], Coroutine[None, None, List[str]]]] = {
    "text/plain": _parse_txt,
    "application/pdf": _parse_pdf,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": _parse_docx,
    "application/epub+zip": _parse_epub,
    "application/msword": _parse_doc,
}

async def parse_document(file: UploadFile) -> List[str]:
    parser = PARSERS.get(file.content_type)
    
    if not parser:
        logger.error(f"Unsupported file type: {file.content_type} for file {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '{file.content_type}' for file '{file.filename}' is not supported."
        )
    
    try:
        logger.info(f"Parsing file: {file.filename} with type: {file.content_type}")
        content_lines = await parser(file.file)
        return content_lines
    except Exception as e:
        logger.error(f"Failed to parse {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not process file: {file.filename}. It might be corrupted or in an invalid format."
        )