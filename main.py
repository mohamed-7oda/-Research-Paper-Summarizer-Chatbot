from fastapi import FastAPI, UploadFile, File, Body
from pydantic import BaseModel
import os
import requests
import re
from dotenv import load_dotenv

from langchain_cohere import ChatCohere
from langchain.schema import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# =========================
# 0. Setup
# =========================
load_dotenv()
app = FastAPI(title="Research Paper Summarizer + Chatbot")

# Google API key (embedding)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Cohere API key
os.environ["CO_API_KEY"] = os.getenv("CO_API_KEY")

# Global Chroma DB (empty at startup)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
db = None   # will be created after upload


# =========================
# Utility: Clean Text
# =========================
def clean_text(text: str) -> str:
    # Remove control characters and null bytes
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)

    # Collapse multiple newlines and spaces
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    return text.strip()


# =========================
# 1. Upload Endpoint (PDF)
# =========================
@app.post("/upload")
async def upload_paper(file: UploadFile = File(...)):
    """
    Upload a research paper PDF, process it, and store in vector DB.
    """
    global db

    save_path = f"uploaded_papers/{file.filename}"
    os.makedirs("uploaded_papers", exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(await file.read())

    try:
        loader = PyPDFLoader(save_path)
        docs = loader.load()
    except Exception:
        return {"message": "❌ Could not process this PDF."}

    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    for chunk in chunks:
        chunk.metadata["paper_name"] = file.filename

    if db is None:
        db = Chroma.from_documents(chunks, embedding=embeddings, collection_name="research_papers")
    else:
        db.add_documents(chunks)

    return {"message": f"✅ {file.filename} uploaded and indexed successfully."}


# =========================
# 1b. Fetch from Link Endpoint (PDF or Webpage)
# =========================
@app.post("/fetch_link")
async def fetch_link(payload: dict = Body(...)):
    """
    Fetch a research paper from a given URL (PDF or webpage),
    extract text, clean, and index it into the vector DB.
    """
    global db
    url = payload.get("url")

    if not url:
        return {"message": "❌ No URL provided."}

    try:
        if url.lower().endswith(".pdf"):
            # Handle remote PDF
            pdf_path = f"downloaded_papers/{os.path.basename(url)}"
            os.makedirs("downloaded_papers", exist_ok=True)

            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                return {"message": "❌ Could not download PDF."}

            with open(pdf_path, "wb") as f:
                f.write(r.content)

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
        else:
            # Handle webpage
            loader = WebBaseLoader(url)
            docs = loader.load()

        if not docs:
            return {"message": "❌ No text could be extracted."}

        for doc in docs:
            doc.page_content = clean_text(doc.page_content)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        for chunk in chunks:
            chunk.metadata["source_url"] = url

        if db is None:
            db = Chroma.from_documents(chunks, embedding=embeddings, collection_name="research_papers")
        else:
            db.add_documents(chunks)

        return {"message": f"✅ Paper fetched and indexed from {url}"}

    except Exception as e:
        return {"message": f"❌ Failed to fetch: {str(e)}"}


# =========================
# 2. Retrieval
# =========================
def get_relevant_chunks_with_scores(question, n_results=5):
    results = db.similarity_search_with_score(question, k=n_results)
    sorted_results = sorted(results, key=lambda x: x[1])
    return [
        f"From: {doc.metadata.get('paper_name', doc.metadata.get('source_url', 'Unknown'))}\n\n{doc.page_content}"
        for doc, score in sorted_results
    ]


# =========================
# 3. LLM Setup
# =========================
llm = ChatCohere(model="command-a-03-2025")

qa_prompt = ChatPromptTemplate.from_template(
    """You are an expert research assistant. 
Use the provided paper chunks to answer the user’s question. 
Be clear, and reference the paper name or source URL when useful.  

User Question:
{question}

Relevant Paper Chunks:
{context}

Answer in an academic but easy-to-understand style.
"""
)

summary_prompt = ChatPromptTemplate.from_template(
    """You are an expert summarizer. 
Summarize the following research paper chunks into a structured summary. 
Include: Objective, Methods, Results, and Conclusion.  

Paper Content:
{context}

Provide the summary in bullet points.
"""
)


def generate_answer(question, n_results=5):
    relevant_chunks = get_relevant_chunks_with_scores(question, n_results)
    context = "\n\n".join(relevant_chunks)
    chain = qa_prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


def generate_summary(n_results=8):
    results = db.similarity_search("summary", k=n_results)
    context = "\n\n".join([doc.page_content for doc in results])
    chain = summary_prompt | llm | StrOutputParser()
    return chain.invoke({"context": context})


# =========================
# 4. API Models
# =========================
class QuestionRequest(BaseModel):
    question: str
    n_results: int = 5


class AnswerResponse(BaseModel):
    answer: str


class SummaryResponse(BaseModel):
    summary: str


# =========================
# 5. Endpoints
# =========================
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if db is None:
        return {"answer": "⚠️ No papers uploaded yet."}
    answer = generate_answer(request.question, request.n_results)
    return AnswerResponse(answer=answer)


@app.get("/summarize", response_model=SummaryResponse)
async def summarize_papers():
    if db is None:
        return {"summary": "⚠️ No papers uploaded yet."}
    summary = generate_summary()
    return SummaryResponse(summary=summary)



# =========================
# 6. Translation
# =========================
# Request schema
class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "Arabic"

# Translation endpoint using LLM
@app.post("/translate")
async def translate_text(request: TranslateRequest):
    prompt = ChatPromptTemplate.from_template(
        """
        You are a professional translator. 
        Translate the following text into {target_lang}, keeping the meaning precise and clear:

        {text}
        """
    )

    chain = prompt | llm | StrOutputParser()
    translated = chain.invoke(
        {"text": request.text, "target_lang": request.target_lang}
    )
    return {"translated_text": translated}
