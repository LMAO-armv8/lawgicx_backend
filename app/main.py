from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from bs4 import BeautifulSoup
from .dependencies import get_db
import os

app = FastAPI(title="Legal Chatbot API", version="1.0")

# Load Model & Tokenizer
model_name = "law-ai/InLegalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Embedding model for FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# Load FAISS Index (or Create If Not Exists)
if os.path.exists("embeddings/index.faiss"):
    vector_store = FAISS.load_local("embeddings", embedding_model)
else:
    vector_store = None  # Create FAISS later

# Function to extract text from HTML
def html_to_text(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()

# API Request Model
class QueryRequest(BaseModel):
    query: str

# Endpoint to Add Legal Documents to FAISS

@app.get("/")
def read_root():
    return {"message": "Welcome to the Legal Chatbot API"}

# @app.get("/legal-query")
# def legal_query(query: str, db: dict = Depends(get_db)):
#     return {"query": query, "db_status": db["message"]}

@app.post("/add_document/")
async def add_document(html_content: str):
    global vector_store
    plain_text = html_to_text(html_content)
    
    if vector_store is None:
        vector_store = FAISS.from_texts([plain_text], embedding_model)
    else:
        vector_store.add_texts([plain_text])

    vector_store.save_local("embeddings")
    return {"message": "Document indexed successfully!"}

# Endpoint to Answer Legal Questions
@app.post("/ask/")
async def answer_query(request: QueryRequest):
    if vector_store is None:
        raise HTTPException(status_code=404, detail="No documents indexed!")

    query = request.query
    query_embedding = embedding_model.embed_query(query)
    results = vector_store.similarity_search_by_vector(query_embedding, k=3)
    
    context = " ".join([res.page_content for res in results])
    answer = qa_pipeline(question=query, context=context)

    return {"answer": answer["answer"]}

# Run server: uvicorn main:app --reload
