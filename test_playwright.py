import modal
app = modal.App("test-playwright")
image = modal.Image.debian_slim(python_version="3.11").apt_install("libgl1", "libglib2.0-0", "ffmpeg", "libsndfile1", "python3-dev", "build-essential").pip_install_from_requirements("/Users/macbook/Downloads/bruv-schools/ai-service/requirements.txt").run_commands("playwright install --with-deps chromium").add_local_python_source("llm_router")

@app.function(image=image)
def test_search():
    from llm_router import execute_web_search, _search_results_have_substance
    res, sources = execute_web_search("epl results yesterday")
    print("PLAYWRIGHT RESULT:", repr(res))
    print("SUBSTANCE:", _search_results_have_substance(res, "what was the epl results yesterday"))

@app.local_entrypoint()
def main():
    test_search.remote()
