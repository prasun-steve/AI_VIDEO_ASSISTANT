import os 
import uuid
from langchain_chroma import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {"device" : 'cpu'}
    )

def build_vector_store(transcript : str)->Chroma:
    print("Building vector Store")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    chunks = splitter.split_text(transcript)
    if not chunks:
        raise ValueError("Cannot build meeting search because the transcript is empty.")

    docs = [
        Document(page_content=chunk, metadata = {'chunk_index' : i})
        for i,chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    # Keep each meeting isolated and in memory.  Reusing one persisted collection
    # mixed earlier meetings into later answers and grew indefinitely over time.
    vector_store = Chroma.from_documents(
        documents= docs,
        embedding=embeddings,
        collection_name=f"meeting_{uuid.uuid4().hex}",
    )

    return vector_store



def load_vector_store() ->Chroma:
    raise RuntimeError("Saved vector stores are no longer used; analyse a meeting first.")

def get_retriever(vector_store : Chroma, k :int = 4):
    return vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {"k":k}
    )
