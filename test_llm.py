import asyncio
from llm_router import ask_qwen

async def test():
    print("Asking...")
    res = await ask_qwen("what was the epl results this weekend", "No relevant context found.", selected_model="small")
    print("Response:")
    print(res)

if __name__ == "__main__":
    asyncio.run(test())
