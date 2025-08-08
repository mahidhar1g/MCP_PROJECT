# MCP_PROJECT/server.py

from dotenv import load_dotenv
from utils.initiate_mcp import mcp
from tools.document_retrieval import document_retrieval_tool
from tools.web_search import web_search_tool

load_dotenv()

if __name__ == "__main__":
    transport = "stdio"
    if transport == "stdio":
        print("Running server with stdio transport")
        mcp.run(transport="stdio")
    elif transport == "sse":
        print("Running server with SSE transport")
        mcp.run(transport="sse")
    elif transport == "streamable-http":
        print("Running server with Streamable HTTP transport")
        mcp.run(transport="streamable-http")
    else:
        raise ValueError(f"Unknown transport: {transport}")