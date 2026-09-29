import modal
import asyncio

async def main():
    ModelCls = modal.Cls.from_name("bruv-schools-llm", "QwenSmallModel")
    model = ModelCls()
    print("Calling generate...")
    response = await model.generate.remote.aio("System msg", "User msg")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
