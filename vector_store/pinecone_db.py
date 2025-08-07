from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException
import os
from dotenv import load_dotenv
from utils.document_utils import chunk_text
from utils.embeddings import embeddings

load_dotenv()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))



def create_pinecone_index():
    """
    Create a Pinecone index if it does not already exist.
    
    Returns:
        Index: A Pinecone index object.
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "agentic-rag-pinecone"

    if index_name not in pc.list_indexes():
        try:
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Index '{index_name}' created.")
        except PineconeException as e:
            if e.status == 409:  # Conflict error, index already exists
                print(f"Index '{index_name}' already exists.")
            else: 
                raise e  
    else:
        print(f"Index '{index_name}' already exists.")

    index = pc.Index(index_name)
    return index


def upsert_data_to_pinecone(documents, full_text, index):
    """
    Upsert the text chunks and their embeddings into the Pinecone index.
    
    Args:
        documents (list): A list of documents.
        full_text (str): The full text to be split into chunks.
        index (Index): A Pinecone index object.
    """

    try:
        chunks = chunk_text(documents, full_text, CHUNK_SIZE, CHUNK_OVERLAP)
    except Exception as e:
        raise RuntimeError(f"An error occurred while splitting the text: {e}")
    
    # Check if data has already been upserted with the same chunk size
    upserted_flag_id = "upserted_flag"
    response = index.fetch(ids=[upserted_flag_id])
    if response and upserted_flag_id in response.vectors:
        stored_chunk_size = response.vectors[upserted_flag_id]["metadata"].get("chunk_size")
        if stored_chunk_size == CHUNK_SIZE:
            print("Data already upserted with the same chunk size. Skipping upsert.")
        else:
            print("Chunk size has changed. Deleting existing vectors and upserting new ones.")
            index.delete(delete_all=True)
    else:
        print("No existing data found. Proceeding with upsert.")

    # Upsert embeddings into the Pinecone index
    if not response or upserted_flag_id not in response.vectors or stored_chunk_size != CHUNK_SIZE:
        for i, chunk in enumerate(chunks):
            chunk_embedding = embeddings.embed_query(chunk["text"])
            index.upsert([(str(i), chunk_embedding, {"text": chunk["text"], "metadata": str(chunk["metadata"])})])

        # Upsert a flag to indicate data has been upserted, including the chunk size
        index.upsert([(upserted_flag_id, [0.1] * 1536, {"text": "upserted_flag", "chunk_size": CHUNK_SIZE})])
        print("Finished upserting embeddings.")
