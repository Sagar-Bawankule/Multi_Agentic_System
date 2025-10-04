import asyncio
import httpx
from app import app

async def main():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        for path in ["/health", "/health/extended", "/agents/status"]:
            r = await client.get(path)
            print(f"{path} -> {r.status_code}\n{r.json()}\n")
        # Controller decision sample
        r = await client.get('/controller/decide', params={'query': 'recent papers on graph transformers'})
        print("/controller/decide ->", r.status_code, r.json())

        # Generate synthetic PDF and query stats
        gen = await client.post('/pdf/generate_from_text', data={'title':'Test PDF','body':'User: Hi\nAgent: Hello world from PDF.'})
        print("Generated PDF:", gen.json())
        stats = await client.get('/pdf/stats')
        print("PDF stats:", stats.json())

if __name__ == "__main__":
    asyncio.run(main())
