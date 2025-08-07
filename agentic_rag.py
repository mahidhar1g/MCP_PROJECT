from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from dotenv import load_dotenv
from langchain import hub
from tools.document_retrieval import document_retrieval_tool
from tools.web_search import web_search_tool


load_dotenv()

# Chat model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)
        
def main():
    prompt = hub.pull("hwchase17/react")
    tools = [document_retrieval_tool, web_search_tool]
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    while True:
        query = input("Ask a question (or type exit to quit): ")
        if query.lower() == "exit":
            break
        
        response = agent_executor.invoke({"input": query})
        print(f"\nAnswer: {response}")


if __name__ == "__main__":
    main()
    