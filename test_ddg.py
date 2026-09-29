import modal
app = modal.App("test-ddg")
image = modal.Image.debian_slim(python_version="3.11").pip_install("playwright").run_commands("playwright install --with-deps chromium")

@app.function(image=image)
def test_search():
    from playwright.sync_api import sync_playwright
    import urllib.parse
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        url = f"https://duckduckgo.com/?q={urllib.parse.quote('premier league results yesterday')}"
        page.goto(url, wait_until="networkidle", timeout=10000)
        page.wait_for_timeout(2000)
        print("DUCKDUCKGO RESULT:", repr(page.locator("body").inner_text()[:1000]))
        browser.close()

@app.local_entrypoint()
def main():
    test_search.remote()
