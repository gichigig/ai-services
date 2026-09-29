import modal
import asyncio

async def run():
    Cls = modal.Cls.from_name('bruv-schools-llm', 'QwenSmallModel')
    model = Cls()
    print("Calling generate...")
    res = await model.generate.remote.aio("hello")
    print("Result:", res)

asyncio.run(run())
