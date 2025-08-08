# MCP_PROJECT/tools/web_search.py

from langchain.agents import tool
from langchain_community.tools import DuckDuckGoSearchRun
from utils.initiate_mcp import mcp

@mcp.tool()
def web_search_tool(query: str) -> str:
    """
    Perform a web search using DuckDuckGo and return relevant info.
    """
    return DuckDuckGoSearchRun().run(query)