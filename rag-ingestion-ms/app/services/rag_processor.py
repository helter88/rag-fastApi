import asyncio
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
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from app.core.config import settings
from app.services.document_loader import load_document

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

embeddings_model = HuggingFaceEndpointEmbeddings(
    model=settings.HF_EMBEDDING_MODEL,
    huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN

)

chroma_client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
vector_store = Chroma(
    client=chroma_client,
    collection_name=settings.RAG_COLLECTION_NAME,
    embedding_function=embeddings_model
)

logger.info(f"Connected with ChromaDB and collection loaded: {settings.RAG_COLLECTION_NAME}")

DOCUMENT_INDEX_ID = "__DOCUMENT_INDEX__"
index_update_lock = asyncio.Lock()


async def get_all_document_names() -> List[str]:
    """
    Fetches the list of document names from the dedicated index document.
    """
    try:
        logger.info(f"Fetching index document with ID: {DOCUMENT_INDEX_ID}")
        
        def get_index_sync():
            index_doc = vector_store.get(ids=[DOCUMENT_INDEX_ID], include=["metadatas"])
            metadatas = index_doc.get('metadatas')
            
            if not metadatas:
                logger.warning("Index document not yet created. The database is likely empty.")
                return []
            
            filenames = metadatas[0].get('filenames', [])
            logger.success(f"Fetched {len(filenames)} document names from the index.")
            return sorted(filenames)

        return await run_in_threadpool(get_index_sync)

    except Exception as e:
        logger.error(f"Failed to fetch document index from ChromaDB: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading index from the vector store: {e}")



async def _update_document_index(new_filenames: List[str]):
    """Internal function to update the document index."""
    async with index_update_lock:
        logger.info("Lock acquired. Updating file name index.")
        current_filenames_list = await get_all_document_names()
        current_filenames_set = set(current_filenames_list)
        
        truly_new_files = [f for f in new_filenames if f not in current_filenames_set]

        if not truly_new_files:
            logger.info("No new, unique filenames to add to the index.")
            return

        updated_list = current_filenames_list + truly_new_files
        
        def upsert_index_sync():
            vector_store.upsert(
                ids=[DOCUMENT_INDEX_ID],
                metadatas=[{"filenames": updated_list}],
                documents=["This is an index document. Do not delete."]
            )
            logger.success(f"Index updated. Added {len(truly_new_files)} new names.")

        await run_in_threadpool(upsert_index_sync)

async def process_and_store_files(files: List[UploadFile]) -> tuple[int, List[str]]:
    all_chunks = []
    error_files = []
    processed_filenames = []

    for file in files:
        try:
            #Temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            logger.info(f"File processing: {file.filename}")

            def process_file_sync():
                docs = load_document(tmp_path)
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
            processed_filenames.append(file.filename)

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

        if processed_filenames:
            await _update_document_index(processed_filenames)

        return len(all_chunks), error_files
    
    except Exception as e:
        logger.error(f"Failed to add documents to ChromaDB: {e}")
        raise HTTPException(status_code=500, detail=f"Error writing to the vector database: {e}")