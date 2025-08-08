# MCP_PROJECT/utils/openai_call.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def summarize_with_llm(question: str, context: str) -> str:
    """
    Summarize the provided context using a ChatOpenAI LLM for a given question.

    Args:
        question (str): The user's query or question.
        context (str): The raw context text to summarize.

    Returns:
        str: The summarized answer from the LLM.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    

    prompt = ChatPromptTemplate.from_messages([
        ("system",  "You are a helpful technical assistant that answers questions using only the provided context. "
                    "Keep responses concise, grounded in facts, and avoid speculation. Do not hallucinate or make up information."),
        
        ("human",   "Based on the following context, answer the question below:\n\n"
                    "Question: {question}\n\n"
                    "Context:\n{context}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context})
    
    return response.content
