import pytest
import httpx
from app import app


@pytest.mark.anyio
async def test_ask_and_logs_flow():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        resp = await client.post('/ask', data={'query': 'Explain controllers', 'use_pdf': False})
        assert resp.status_code == 200
        data = resp.json()
        assert 'final_answer' in data
        assert 'agents_invoked' in data
        assert 'decision_rationale' in data
        logs_resp = await client.get('/logs?limit=5')
        assert logs_resp.status_code == 200
        logs_json = logs_resp.json()
        assert 'logs' in logs_json
        assert any(log.get('query') == 'Explain controllers' for log in logs_json['logs'])


@pytest.mark.anyio
async def test_health_extended():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        resp = await client.get('/health/extended')
        assert resp.status_code == 200
        h = resp.json()
        assert 'pdf_chunks' in h