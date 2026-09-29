import urllib.request
import urllib.parse
import re
import html as html_lib
import requests
from bs4 import BeautifulSoup

q = "epl results this weekend"
url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(q)}"
req = urllib.request.Request(
    url, 
    headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
)
raw_html = urllib.request.urlopen(req, timeout=8).read().decode("utf-8", errors="ignore")

blocks = re.findall(r'<div[^>]*class="[^"]*web-result[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</div>', raw_html, re.DOTALL)
if not blocks:
    blocks = re.findall(r'<div[^>]*class="[^"]*result__body[^"]*"[^>]*>(.*?)</div>\s*</div>', raw_html, re.DOTALL)

unique_results = []
seen = set()
for block in blocks[:6]:
    title_match = re.search(r'<a class="result__url"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
    if title_match:
        raw_url = title_match.group(1)
        if "uddg=" in raw_url:
            clean_url = urllib.parse.unquote(raw_url.split("uddg=")[1].split("&")[0])
        else:
            clean_url = raw_url
        if clean_url not in seen:
            seen.add(clean_url)
            unique_results.append({"href": clean_url})

if unique_results:
    top_link = unique_results[0].get("href")
    print("Top link:", top_link)
    res = requests.get(top_link, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
    print("Status:", res.status_code)
    if res.status_code == 200:
        soup = BeautifulSoup(res.text, 'html.parser')
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
        text = soup.get_text(separator=' ')
        clean_text = ' '.join(text.split())
        print("Scraped text snippet:", clean_text[:500])
