# Model Context Protocol (MCP)

![MCP Diagram](https://raw.githubusercontent.com/mahidhar1g/MCP_PROJECT/main/documents/Images/MCP.png)

A **universal standard** for connecting AI agents (LLMs) to external tools and data sources via a simple client–server protocol.

---

## 🔍 What Is MCP?

- **Problem:** LLMs alone can generate text but can’t perform real-world actions (e.g. booking flights, querying databases) without custom integrations.  
- **Solution:** MCP defines a **common JSON-RPC protocol** so any AI client can discover and call “tools” exposed by any MCP server, eliminating per-API glue code.

---

## 🏛 Architecture

![MCP Diagram](https://github.com/mahidhar1g/MCP_PROJECT/blob/9feb226c03867fab95aa7308ad6161b350b72b39/documents/Images/FlowDiagram.png)


