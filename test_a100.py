import modal

app = modal.App("test-a100")

@app.function(gpu="A100")
def f():
    print("hello")
