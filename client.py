# MCP_PROJECT/client.py

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from mcp.client.tool_loader import ToolLoader
from mcp.client.stdio import stdio_client

load_dotenv()

def main():
    # LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Load tools from MCP server
    loader = ToolLoader(stdio_client(["python3", "server.py"]))
    tools = loader.load()

    # Create ReAct-style agent with those tools
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    print("🧠 Agent ready. Ask a question (type 'exit' to quit):")
    while True:
        query = input(">> ")
        if query.strip().lower() == "exit":
            break

        response = agent_executor.invoke({"input": query})
        print("\n📝 Final Answer:\n", response)

if __name__ == "__main__":
    main()
