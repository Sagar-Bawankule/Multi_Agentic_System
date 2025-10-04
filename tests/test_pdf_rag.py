import os
import tempfile
from agents.pdf_rag_agent import PDFRAGAgent

PDF_CONTENT = "This is a small PDF about multi agent systems and retrieval augmented generation."

def create_minimal_pdf(path: str):
    # Write a minimal PDF with the text (very naive)
    with open(path, 'wb') as f:
        f.write(b"%PDF-1.4\n1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n")
        f.write(b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n")
        stream_text = f"BT /F1 12 Tf 20 720 Td ({PDF_CONTENT}) Tj ET".encode()
        f.write(b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>endobj\n")
        f.write(f"4 0 obj<< /Length {len(stream_text)} >>stream\n".encode())
        f.write(stream_text + b"\nendstream endobj\ntrailer<< /Root 1 0 R >>\n%%EOF")


def test_pdf_ingest_and_answer():
    agent = PDFRAGAgent(storage_dir="test_vector_store")
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = os.path.join(tmp, "temp.pdf")
        create_minimal_pdf(pdf_path)
        res = agent.ingest_pdf(pdf_path)
        assert res["status"] == "ok"
        out = agent.answer("What is this PDF about?")
        assert "answer" in out
