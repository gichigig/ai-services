import modal
import os

app = modal.App("test-crawl4ai")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg", "libsndfile1", "python3-dev", "build-essential")
    .pip_install_from_requirements(os.path.join(os.path.dirname(__file__), "requirements.txt"))
    .run_commands("playwright install --with-deps chromium")
    .add_local_python_source("llm_router")
)

@app.function(image=image, timeout=60)
async def test_crawl():
    from llm_router import crawl_webpage_content, execute_web_search
    
    test_url = "https://news.ycombinator.com"
    print(f"Testing direct crawl of: {test_url}")
    markdown_content = await crawl_webpage_content(test_url, timeout=10)
    print("Direct crawl extracted characters:", len(markdown_content))
    print("Direct crawl preview (first 300 chars):")
    print(markdown_content[:300])
    
    print("\nTesting full execute_web_search pipeline:")
    res, sources = await execute_web_search("University of Nairobi latest news")
    print("Found sources count:", len(sources))
    for s in sources:
        print(" - Source:", s.get("title"), s.get("url"))
    print("Search result snippet preview (first 400 chars):")
    print(res[:400])

@app.local_entrypoint()
def main():
    test_crawl.remote()
