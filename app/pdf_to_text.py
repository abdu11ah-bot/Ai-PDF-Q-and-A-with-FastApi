
from fastapi import HTTPException

def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, int]:
    """Extract text from PDF bytes. Returns (text, page_count)."""
    try:
        import pypdf
        import io
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages = len(reader.pages)
        text = "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        )
        return text.strip(), pages
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="pypdf not installed. Run: pip install pypdf"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}")