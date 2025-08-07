from langchain.agents import tool
from langchain_community.tools import DuckDuckGoSearchRun

@tool
def web_search_tool(query):
    """
    Perform a web search using DuckDuckGo and return relevant information.
    
    When to use this tool:
        - If the document_retrieval_tool does not find relevant information.
        - Use this tool only when the needed information is missing from the document retrieval tool.
    """
    
    search = DuckDuckGoSearchRun()
    return search.run(query)