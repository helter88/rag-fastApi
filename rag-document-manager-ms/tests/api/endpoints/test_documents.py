

from fastapi import HTTPException
import pytest_asyncio
from app.main import app 
from httpx import ASGITransport, AsyncClient
import pytest

pytestmark = pytest.mark.asyncio


@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test/api/v1") as ac:
        yield ac

async def test_list_documents_success(client: AsyncClient, mocker):
    expected_docs = ["doc1.pdf", "raport.docx"]

    mock_get_docs = mocker.patch(
        "app.api.endpoints.documents.rag_processor.get_all_document_names",
        return_value=expected_docs
    )

    response = await client.get("/documents")

    assert response.status_code == 200
    assert response.json() == {
        "count": 2,
        "documents": expected_docs
    }

    mock_get_docs.assert_awaited_once()

async def test_list_documents_empty(client: AsyncClient, mocker):
    mocker.patch(
        "app.api.endpoints.documents.rag_processor.get_all_document_names",
        return_value=[]
    )

    response = await client.get("/documents")

    assert response.status_code == 200
    assert response.json() == {
        "count": 0,
        "documents": []
    }

async def test_delete_document_success(client: AsyncClient, mocker):

    filename = "File_to_delete.pdf"

    mock_delete = mocker.patch("app.api.endpoints.documents.rag_processor.delete_document_by_name")

    response = await client.delete(f"/documents/{filename}")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Document and all its associated chunks have been successfully deleted.",
        "deleted_filename": filename
    }
    mock_delete.assert_awaited_once_with(filename)

async def test_delete_document_not_found(client: AsyncClient, mocker):
    filename = "Non_Existing.txt"
    
    mocker.patch(
        "app.api.endpoints.documents.rag_processor.delete_document_by_name",
        side_effect=HTTPException(status_code=404, detail=f"Document '{filename}' not found.")
    )

    response = await client.delete(f"/documents/{filename}")

    assert response.status_code == 404
    assert response.json() == {"detail": f"Document '{filename}' not found."}

async def test_ingest_documents_success(client: AsyncClient, mocker):
    mock_process = mocker.patch(
        "app.api.endpoints.documents.rag_processor.process_and_store_files",
        return_value=(150, [])
    )

    files_to_upload = [
        ('files', ('test1.pdf', b'fake pdf content', 'application/pdf')),
        ('files', ('test2.txt', b'fake text content', 'text/plain'))
    ]

    response = await client.post("/ingest-to-rag", files=files_to_upload)

    assert response.status_code == 200
    assert response.json() == {
        "total_chunks_added": 150,
        "processed_files_count": 2,
        "files_with_errors": [],
        "message": "Ingestion process completed. Processed 2 files successfully."
    }
    assert mock_process.await_count == 1