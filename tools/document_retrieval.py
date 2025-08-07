import os
from langchain.agents import tool
from utils.document_utils import load_pdf
from utils.embeddings import embeddings
from vector_store.pinecone_db import create_pinecone_index, upsert_data_to_pinecone

@tool
def document_retrieval_tool(query):
    """
    Retrieve information from the embedded documents based on the query.
    
    Additional Information on when to use this tool:
        - This should be the primary source of information.
        - If no relevant information is found, return an empty string or indicate that the data is unavailable.
    """
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    documents_dir = os.path.join(current_dir, "..", "documents")
    documents, full_text = load_pdf(documents_dir)
    try:
        index = create_pinecone_index()
    except Exception as e:
        raise RuntimeError(f"An error occurred while creating the Pinecone index: {e}")
    
    upsert_data_to_pinecone(documents, full_text, index)
    query_embedding = embeddings.embed_query(query)
    response = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    matched_data = [(match.metadata["text"], match.metadata) for match in response.matches]
    if matched_data:
        return "\n\n".join([f"[Page {page}] {text}" for text, page in matched_data])
    else:
        return "No relevant information found in the provided documents."