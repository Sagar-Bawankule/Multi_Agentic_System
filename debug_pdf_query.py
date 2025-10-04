import asyncio, json
import httpx
from app import app

QUERY = "summarize this uploaded pdf"

async def main():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post('/ask', data={'query': QUERY, 'use_pdf': 'true'})
        data = resp.json()
        print("Status:", resp.status_code)
        print("Agents:", data.get('agents_invoked'))
        print("Rationale:", data.get('decision_rationale'))
        print("Answer (first 600 chars):\n", (data.get('final_answer','') or '')[:600])
        pdf_sources = data.get('documents',{}).get('pdf_rag', [])
        print(f"Sources returned: {len(pdf_sources)}")
        if pdf_sources:
            print("First source preview:", pdf_sources[0].get('preview','')[:160])

if __name__ == '__main__':
    asyncio.run(main())
