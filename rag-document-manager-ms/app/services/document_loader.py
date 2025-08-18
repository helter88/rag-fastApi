import pathlib

from loguru import logger
from langchain_core.documents import Document
from langchain_community.document_loaders.epub import UnstructuredEPubLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.word_document import (
    UnstructuredWordDocumentLoader
)


class DocumentLoaderException(Exception):
    pass

class DocumentLoader:
    """Loads in a document with a supported extension."""

    supported_extensions = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".epub": UnstructuredEPubLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
    }

def load_document(temp_filepath: str) -> list[Document]:
    """Load a file and return it as a list of documents."""
    
    ext = pathlib.Path(temp_filepath).suffix
    loader = DocumentLoader.supported_extensions.get(ext)
    if not loader:
        raise DocumentLoaderException(
            f"Invalid extension type {ext}, cannot load this type of file"
        )

    loaded = loader(temp_filepath)
    return loaded.load()

