# MCP_PROJECT/tools/document_retrieval.py

import os
from utils.initiate_mcp import mcp
from utils.document_utils import load_pdf
from utils.embeddings import embeddings
from vector_store.pinecone_db import create_pinecone_index, upsert_data_to_pinecone
from utils.openai_call import summarize_with_llm  # ✅ new import

@mcp.tool()
def document_retrieval_tool(query: str) -> str:
    """
    Retrieve information from embedded documents based on the query and return a summarized answer.

    Data Context:
        This tool has access to two core documents:

        1. AI_Agents.pdf - A comprehensive overview of artificial intelligence agents,
           including types (reactive, deliberative, hybrid, learning, multi-agent),
           core components (perception, reasoning, action, learning, autonomy),
           real-world applications (healthcare, finance, robotics), and ethical concerns
           (bias, privacy, transparency).

        2. RAG.pdf - An in-depth exploration of Retrieval-Augmented Generation (RAG),
           its architecture (retriever + generator + fusion), use cases (question answering,
           summarization, content creation), types (end-to-end, modular), working pipeline,
           and current challenges (retrieval quality, latency, hallucination).

    Usage Guidance:
        Use this tool to answer questions that can be answered using these documents.
        If no relevant content is found, consider using a web search or fallback tool.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    documents_dir = os.path.abspath(os.path.join(current_dir, "..", "documents"))
    documents, full_text = load_pdf(documents_dir)

    try:
        index = create_pinecone_index()
        upsert_data_to_pinecone(documents, full_text, index)
    except Exception as e:
        raise RuntimeError(f"Error preparing documents: {e}")

    query_embedding = embeddings.embed_query(query)
    response = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    matched_chunks = [(match.metadata["text"], match.metadata) for match in response.matches]

    if not matched_chunks:
        return "No relevant information found in the provided documents."

    context_text = "\n\n".join([chunk for chunk, _ in matched_chunks])
    summary = summarize_with_llm(question=query, context=context_text)

    return summary
