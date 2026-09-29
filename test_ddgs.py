from duckduckgo_search import DDGS
with DDGS() as ddgs:
    results = ddgs.news("epl results this weekend", max_results=5)
    print("NEWS:")
    for r in results:
        print(r)
