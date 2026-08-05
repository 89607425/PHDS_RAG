import os
from docx import Document
from pathlib import Path

def load_docx(file_path: str) -> dict:
    doc = Document(file_path)

    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs)

    tables_text = []
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells)
            if row_text.strip():
                tables_text.append(row_text)
    if tables_text:
        full_text += "\n\n--- 表格 ---\n" + "\n".join(tables_text)

    file_name = Path(file_path).stem
    return {
        "text": full_text,
        "metadata": {
            "source": file_path,
            "title": file_name,
            "file_size": os.path.getsize(file_path),
        }
    }

def load_all_docx(folder: str) -> list[dict]:
    docs = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith('.docx') and not f.startswith('~$'):
                docs.append(load_docx(os.path.join(root, f)))
    return docs
