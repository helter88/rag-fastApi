
from typing import List, TypedDict
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings


class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]

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

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=settings.GOOGLE_API_KEY, temperature=0.2)

def retrieve_documents(state: GraphState):
    """Retrieves documents from the vector store."""
    logger.info("Node: retrieve_documents")
    question = state["question"]
    documents = retriever.invoke(question)
    logger.info(f"Retrieved {len(documents)} documents.")
    # Return the updated state
    return {"documents": documents, "question": question}

def generate_answer(state: GraphState):
    """Generates an answer based on the retrieved documents."""
    logger.info("Node: generate_answer")
    question = state["question"]
    documents = state["documents"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer the user's question based solely on the context provided below."
          "If the context does not contain an answer, state that and do not try to make one up.\n\nContext:\n---\n{context}\n---"),
        ("human", "Question: {question}")
    ])
    
    rag_chain = prompt | llm | StrOutputParser()
    
    context_str = "\n\n".join(doc.page_content for doc in documents)
    generation = rag_chain.invoke({"context": context_str, "question": question})
    
    logger.info("Generated an answer.")
    return {"generation": generation}