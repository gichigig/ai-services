import modal
import asyncio

async def run():
    Cls = modal.Cls.from_name('bruv-schools-llm', 'QwenSmallModel')
    print("got Cls", Cls)
    model = Cls()
    print("got model", model)
    res = await model.generate.remote.aio("hello")
    print(res)

asyncio.run(run())
