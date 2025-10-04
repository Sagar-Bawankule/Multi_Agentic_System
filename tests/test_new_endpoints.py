import httpx
import pytest
from app import app

@pytest.mark.anyio
async def test_controller_decide():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get('/controller/decide', params={'query': 'recent papers on transformers'})
        assert r.status_code == 200
        data = r.json()
        assert 'agents' in data and isinstance(data['agents'], list)
        assert any(a in data['agents'] for a in ['arxiv'])
        assert 'rationale' in data

@pytest.mark.anyio
async def test_pdf_generate_and_stats_and_reset():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post('/pdf/generate_from_text', data={'title': 'NebulaByte Session', 'body': 'User: hi\nAgent: hello world'})
        assert r.status_code == 200
        gen = r.json()
        assert gen.get('status') == 'generated'
        s = await client.get('/pdf/stats')
        assert s.status_code == 200
        stats = s.json()
        assert stats['chunks'] >= 1
        rs = await client.post('/pdf/reset')
        assert rs.status_code == 200
        after = rs.json()['stats']
        assert after['chunks'] == 0
