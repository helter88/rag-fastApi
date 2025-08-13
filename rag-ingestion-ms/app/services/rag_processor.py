import datetime
import os
import tempfile
from typing import List
import chromadb
from fastapi import HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from app.core.config import settings

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
embeddings_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=settings.HUGGINGFACEHUB_API_TOKEN,
    model_name=settings.HF_EMBEDDING_MODEL
)

chroma_client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
vector_store = Chroma(
    client=chroma_client,
    collection_name=settings.RAG_COLLECTION_NAME,
    embedding_function=embeddings_model
)

logger.info(f"Connected with ChromaDB and collection loaded: {settings.RAG_COLLECTION_NAME}")

async def process_and_store_files(files: List[UploadFile]) -> tuple[int, List[str]]:
    all_chunks = []
    error_files = []

    for file in files:
        try:
            #Temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            logger.info(f"File processing: {file.filename}")

            def process_file_sync():
                loader = UnstructuredFileLoader(tmp_path)
                docs = loader.load()
                chunks = text_splitter.split_documents(docs)
                
                logger.info(f"Metadata inspection of the first fragment from the file '{file.filename}':")
                if chunks:
                    print(chunks[0].metadata)
                
                for chunk in chunks:
                    chunk.metadata['original_filename'] = file.filename
                    chunk.metadata['ingestion_timestamp_utc'] = datetime.now(datetime.timezone.utc).isoformat()
                
                logger.info(f"File '{file.filename}' devided for {len(chunks)} chunks with metadata.")
                return chunks
            
            chunks = await run_in_threadpool(process_file_sync)
            all_chunks.extend(chunks)

        except Exception as e:
            logger.error(f"Error during file processing {file.filename}: {e}")
            error_files.append(file.filename)
        
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
            await file.close()

    if not all_chunks:
        logger.warning("No fragments found to add to the vector database.")
        return 0, error_files
    
    try:
        logger.info(f"Adding {len(all_chunks)} chunks to the vector database ...")
        await run_in_threadpool(vector_store.add_documents, documents=all_chunks)
        logger.success(f"Added {len(all_chunks)} chunks.")
        return len(all_chunks), error_files
    
    except Exception as e:
        logger.error(f"Failed to add documents to ChromaDB: {e}")
        raise HTTPException(status_code=500, detail=f"Error writing to the vector database: {e}")