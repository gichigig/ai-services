import modal

app = modal.App("test-a10g")

@app.function(gpu="A10G")
def f():
    print("hello")
