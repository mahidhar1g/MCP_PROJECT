# client_new.py
import os, asyncio, json, textwrap
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.callbacks import BaseCallbackHandler


class PrintHandler(BaseCallbackHandler):
    # def on_llm_start(self, serialized, prompts, **kwargs):
    #     print("\n===== LLM START =====")
    #     for i, p in enumerate(prompts):
    #         print(f"\n[Prompt {i}]\n{textwrap.shorten(p.replace('\r',''), width=2000, placeholder=' ...')}")
    #     print("=====================")

    # def on_llm_new_token(self, token, **kwargs):
    #     # comment out if too chatty
    #     print(token, end="", flush=True)

    # def on_llm_end(self, response, **kwargs):
    #     print("\n===== LLM END =====")

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized or {}).get("name", "tool")
        print(f"\n🔧 TOOL START: {name}")
        try:
            print("args:", json.dumps(input_str if isinstance(input_str, dict) else {"input": input_str}, indent=2))
        except Exception:
            print("args:", input_str)

    def on_tool_end(self, output, **kwargs):
        print("🔧 TOOL END (result preview):")
        try:
            text = output if isinstance(output, str) else json.dumps(output, indent=2)
        except Exception:
            text = str(output)
        print(textwrap.shorten(text, width=2000, placeholder=" ..."))


async def main():
    load_dotenv()
    sse_url = os.getenv("MCP_SSE_URL", "http://localhost:8050/sse")

    async with sse_client(sse_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)

            # Print discovered tools + robust schema preview (dict or Pydantic)
            print("\n== Discovered MCP Tools ==")
            for t in tools:
                name = getattr(t, "name", t.__class__.__name__)
                desc = getattr(t, "description", "") or ""
                print(f"- {name}: {desc}")

                js_schema = None
                schema_obj = getattr(t, "args_schema", None)
                try:
                    if isinstance(schema_obj, dict):
                        js_schema = schema_obj
                    elif schema_obj is not None:
                        # Pydantic v2 or v1
                        if hasattr(schema_obj, "model_json_schema"):
                            js_schema = schema_obj.model_json_schema()
                        elif hasattr(schema_obj, "schema"):
                            js_schema = schema_obj.schema()
                except Exception:
                    js_schema = None
                if js_schema:
                    print(json.dumps(js_schema, indent=2))

            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            memory = MemorySaver()
            agent = create_react_agent(llm, tools, checkpointer=memory)

            thread_id = os.getenv("THREAD_ID", "local-demo-thread")
            bootstrapped = False
            cb = PrintHandler()

            print("\nAgent ready. Type '/new' for a fresh thread, 'quit' to exit.")
            while True:
                q = input("\nQuery: ").strip()
                if q.lower() == "quit":
                    break
                if q.lower() == "/new":
                    from uuid import uuid4
                    thread_id = f"thread-{uuid4().hex[:8]}"
                    bootstrapped = False
                    print(f"(new thread: {thread_id})")
                    continue

                if not bootstrapped:
                    messages = [
                        SystemMessage(content="You are a precise assistant. Use tools if helpful."),
                        HumanMessage(content=q),
                    ]
                    bootstrapped = True
                else:
                    messages = [HumanMessage(content=q)]

                config = {"configurable": {"thread_id": thread_id}, "callbacks": [cb]}
                result = await agent.ainvoke({"messages": messages}, config=config)

                final = result["messages"][-1].content
                print("\n📝 FINAL ANSWER:\n" + str(final))

                # Quick state snapshot
                try:
                    state = await agent.aget_state(config=config)
                    msgs = state.values.get("messages", [])
                except Exception:
                    msgs = result.get("messages", [])
                print("\n📦 STATE SNAPSHOT")
                print(f"thread_id: {thread_id}")
                print(f"total messages stored: {len(msgs)}")
                for m in msgs[-6:]:
                    role = getattr(m, "type", getattr(m, "role", ""))
                    content = getattr(m, "content", "")
                    if isinstance(content, list):
                        content = " ".join(str(c) for c in content)
                    print(f"- {role}: {textwrap.shorten(str(content), width=180, placeholder=' ...')}")

if __name__ == "__main__":
    asyncio.run(main())
