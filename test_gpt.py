import asyncio
from llm_router import ask_qwen

async def test():
    print("Testing GPT OSS 120B...")
    try:
        res = await ask_qwen(
            question="Hello! Are you running successfully?",
            context="",
            selected_model="gpt_120b"
        )
        print("Response from GPT OSS 120B:")
        print(res)
    except Exception as e:
        print(f"Error testing GPT OSS: {e}")

if __name__ == "__main__":
    asyncio.run(test())
