# MCP_PROJECT/utils/initiate_mcp.py

from mcp.server.fastmcp import FastMCP

# Define MCP instance once
mcp = FastMCP(
    name="Agentic RAG MCP Server",
    host="0.0.0.0",
    port=8050,
    stateless_http=True
)
