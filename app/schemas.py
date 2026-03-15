from pydantic import BaseModel

from openai import BaseModel


class QuestionRequest(BaseModel):
    session_id: str
    question: str


class AnswerResponse(BaseModel):
    answer: str
    session_id: str


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    pages: int
    size_kb: float
    message: str
