import asyncio
from datetime import datetime
import os
import tempfile
from typing import List
from fastapi import HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.services.document_loader import load_document

class DocumentService:
    DOCUMENT_INDEX_ID = "__DOCUMENT_INDEX__"

    def __init__(self, vector_store: Chroma,  text_splitter: RecursiveCharacterTextSplitter):
        self.vector_store = vector_store
        self.text_splitter = text_splitter
        self.index_update_lock = asyncio.Lock()
    
    async def get_all_document_names(self) -> List[str]:
        """
        Fetches the list of document names from the dedicated index document.
        """
        try:
            logger.info(f"Fetching index document with ID: {self.DOCUMENT_INDEX_ID}")
            
            def get_index_sync():
                index_doc = self.vector_store.get(ids=[self.DOCUMENT_INDEX_ID], include=["metadatas"])
                metadatas = index_doc.get('metadatas')
                
                if not metadatas:
                    logger.warning("Index document not yet created. The database is likely empty.")
                    return []
                
                filenames_str = metadatas[0].get('filenames', '')
                if not filenames_str:
                    return []
                
                filenames = filenames_str.split('|')
                logger.success(f"Fetched {len(filenames)} document names from the index.")
                return sorted(filenames)

            return await run_in_threadpool(get_index_sync)

        except Exception as e:
            logger.error(f"Failed to fetch document index from ChromaDB: {e}")
            raise HTTPException(status_code=500, detail=f"Error reading index from the vector store: {e}")
    
    async def _update_document_index(self, new_filenames: List[str]):
        """Internal function to update the document index."""
        async with self.index_update_lock:
            logger.info("Lock acquired. Updating file name index.")
            current_filenames_list = await self.get_all_document_names()
            current_filenames_set = set(current_filenames_list)
            
            truly_new_files = [f for f in new_filenames if f not in current_filenames_set]

            if not truly_new_files:
                logger.info("No new, unique filenames to add to the index.")
                return

            updated_list = current_filenames_list + truly_new_files
            
            def upsert_index_sync():
                serialized_filenames = "|".join(updated_list)
                index_doc = Document(
                    page_content="This is an index document. Do not delete.",
                    metadata={"filenames": serialized_filenames}
                )
                self.vector_store.add_documents(documents=[index_doc], ids=[self.DOCUMENT_INDEX_ID])
                logger.success(f"Index updated. Added {len(truly_new_files)} new names.")

            await run_in_threadpool(upsert_index_sync)

    async def process_and_store_files(self, files: List[UploadFile]) -> tuple[int, List[str]]:
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
                    chunks = self.text_splitter.split_documents(docs)
                    
                    logger.info(f"Metadata inspection of the first fragment from the file '{file.filename}':")
                    if chunks:
                        print(chunks[0].metadata)
                    
                    for chunk in chunks:
                        chunk.metadata['original_filename'] = file.filename
                        chunk.metadata['ingestion_timestamp_utc'] = datetime.now().isoformat()
                    
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
            batch_size = 10
            total_added = 0
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                logger.info(f"Adding batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size} ({len(batch)} chunks)...")
                await run_in_threadpool(self.vector_store.add_documents, documents=batch)
                total_added += len(batch)
            logger.success(f"Added {len(all_chunks)} chunks.")

            if processed_filenames:
                await self._update_document_index(processed_filenames)

            return len(all_chunks), error_files
        
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise HTTPException(status_code=500, detail=f"Error writing to the vector database: {e}")
        
    async def delete_document_by_name(self, filename: str):
        logger.info(f"Deletion request for document received: {filename}")

        current_filenames = await self.get_all_document_names()
        if filename not in current_filenames:
            logger.warning(f"Document '{filename}' not found in the index. No deletion performed.")
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")

        try:
            def delete_chunks_sync():
                self.vector_store.delete(where={"original_filename": filename})
                logger.success(f"Successfully deleted all chunks for document: {filename}")
            
            await run_in_threadpool(delete_chunks_sync)

        except Exception as e:
            logger.error(f"Failed to delete document chunks from ChromaDB for '{filename}': {e}")
            raise HTTPException(status_code=500, detail=f"Error deleting chunks from vector store: {e}")

        # Krok 3: Zaktualizuj dokument indeksu, aby usunąć nazwę pliku
        async with self.index_update_lock:
            logger.info(f"Lock acquired. Updating index to remove '{filename}'.")
            updated_list = await self.get_all_document_names()
            if filename in updated_list:
                updated_list.remove(filename)

                def upsert_index_sync():
                    serialized_filenames = "|".join(updated_list)
                    index_doc = Document(
                        page_content="This is an index document. Do not delete.",
                        metadata={"filenames": serialized_filenames}
                    )

                    self.vector_store.add_documents(documents=[index_doc], ids=[self.DOCUMENT_INDEX_ID])
                    logger.success(f"Index updated successfully. Removed '{filename}'.")

                await run_in_threadpool(upsert_index_sync)
            else:
                logger.warning(f"'{filename}' was already removed from the index, possibly by a concurrent process.")