import os
from agents.controller_agent import ControllerAgent

class DummyPDF:
    def answer(self, q):
        return {"answer": "pdf answer", "sources": []}

class DummyWeb:
    def search_and_summarize(self, q):
        return {"summary": "web summary", "results": []}

class DummyArxiv:
    def search_and_summarize(self, q):
        return {"summary": "arxiv summary", "papers": []}


def build_controller():
    return ControllerAgent(pdf_agent=DummyPDF(), web_agent=DummyWeb(), arxiv_agent=DummyArxiv())


def test_routing_pdf_keyword():
    c = build_controller()
    d = c.decide("Please analyze this PDF about climate")
    assert "pdf_rag" in d["agents"]


def test_routing_arxiv_keyword():
    c = build_controller()
    d = c.decide("recent papers in quantum computing")
    assert "arxiv" in d["agents"]


def test_routing_news_keyword():
    c = build_controller()
    d = c.decide("latest news on AI regulation")
    assert "web_search" in d["agents"]


def test_routing_fallback():
    c = build_controller()
    d = c.decide("Explain gravity to me")
    assert d["agents"] == ["llm"]
