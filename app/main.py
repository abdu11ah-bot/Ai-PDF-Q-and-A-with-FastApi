import asyncio
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from app.config import (PDF_MAX_SIZE_MB, PDF_MAX_PAGES, REQUEST_TIMEOUT_S, OPENAI_MODEL)
from app.pdf_to_text import extract_text_from_pdf
from app.schemas import QuestionRequest, AnswerResponse, UploadResponse


client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)

# ─── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="PDF Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (the frontend)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# ─── In-memory session store ────────────────────────────────────────────────
# Maps session_id -> {"pdf_text": str, "messages": [{"role","content"}]}
sessions: dict[str, dict] = {}



# ─── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
):
    # 1. Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    size_kb = len(file_bytes) / 1024

    # 2. Validate file size
    if size_mb > PDF_MAX_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"PDF is {size_mb:.1f} MB. Maximum allowed size is {PDF_MAX_SIZE_MB} MB."
        )

    # 3. Extract text & validate page count
    pdf_text, pages = extract_text_from_pdf(file_bytes)

    if pages > PDF_MAX_PAGES:
        raise HTTPException(
            status_code=400,
            detail=f"PDF has {pages} pages. Maximum allowed is {PDF_MAX_PAGES} pages."
        )

    if not pdf_text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from PDF. Make sure it's not a scanned image-only PDF."
        )

    # 4. Create or reset session
    import uuid
    sid = session_id or str(uuid.uuid4())
    sessions[sid] = {
        "pdf_text": pdf_text,
        "filename": file.filename,
        "messages": [],   # chat history: [{"role": "user"|"assistant", "content": "..."}]
    }

    return UploadResponse(
        session_id=sid,
        filename=file.filename,
        pages=pages,
        size_kb=round(size_kb, 1),
        message=f"PDF uploaded successfully! {pages} pages extracted."
    )


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(body: QuestionRequest):
    sid = body.session_id
    question = body.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if sid not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload a PDF first."
        )

    session = sessions[sid]
    pdf_text = session["pdf_text"]

    # Build messages for OpenAI
    system_prompt = (
        "You are a helpful assistant. Answer questions based ONLY on the provided PDF content. "
        "If the answer is not in the PDF, say so clearly. Be concise and accurate.\n\n"
        f"--- PDF CONTENT ---\n{pdf_text[:12000]}\n--- END PDF CONTENT ---"
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Include last 10 exchanges (20 messages) for context
    history = session["messages"][-20:]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    # Call OpenAI with timeout
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=800,
            extra_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "PDF QA App"
            }
        )
        answer = response.choices[0].message.content.strip()
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Request timed out after {REQUEST_TIMEOUT_S} seconds. Please try again."
        )
   
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")

    # Save to chat history
    session["messages"].append({"role": "user", "content": question})
    session["messages"].append({"role": "assistant", "content": answer})

    return AnswerResponse(answer=answer, session_id=sid)


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "filename": session.get("filename", ""),
        "messages": session["messages"],
    }


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"message": "Session cleared."}
