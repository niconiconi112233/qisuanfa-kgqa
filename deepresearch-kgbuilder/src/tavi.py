import requests

TAVILY_API_KEY = "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"  # 替换为你的 Tavily API Key

def tavily_search(query: str, max_results: int = 5):
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "include_answers": True,
        "max_results": max_results
    }

    print(f"\n[🌐] Tavily 搜索中: {query}")
    response = requests.post(url, json=payload)

    if response.status_code != 200:
        print(f"[❌] 错误: {response.status_code} - {response.text}")
        return

    data = response.json()
    results = data.get("results", [])

    if not results:
        print("[⚠️] 无搜索结果")
        return

    print(f"\n🔎 共返回 {len(results)} 条结果：\n")
    for idx, result in enumerate(results, 1):
        print(f"{idx}. {result.get('title')}")
        print(f"URL: {result.get('url')}")
        print(f"摘要: {result.get('content')[:300]}...\n")

if __name__ == "__main__":
    queries = [
        "(site:sciencedirect.com/journal/veterinary-clinics-of-north-america-small-animal-practice OR site:avmajournals.avma.org OR site:onlinelibrary.wiley.com/journal/19391676 OR site:plumbs.com OR site:vin.com) epidemiology prevalence statistics laryngeal foreign body dogs cats",
        "(site:sciencedirect.com/journal/veterinary-clinics-of-north-america-small-animal-practice OR site:avmajournals.avma.org OR site:onlinelibrary.wiley.com/journal/19391676 OR site:plumbs.com OR site:vin.com) risk factors for laryngeal foreign bodies in companion animals breed age"
    ]

    for q in queries:
        tavily_search(q)
