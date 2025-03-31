from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
import os
import threading

app = FastAPI(title="Legal RAG Chatbot API", version="1.0")

# Load Legal LLM
model_name = "law-ai/InLegalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Load Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# Load FAISS Index
faiss_lock = threading.Lock()
if os.path.exists("embeddings/index.faiss"):
    vector_store = FAISS.load_local("embeddings", embedding_model, allow_dangerous_deserialization=True)
else:
    vector_store = None  # Initialize later

# Create LLM Pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Setup RAG with LangChain
def get_rag_chain():
    if vector_store is None:
        raise HTTPException(status_code=404, detail="No documents indexed!")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    template = """You are a legal AI assistant. Given the legal documents below and a user's question, provide a detailed answer.
    DOCUMENTS: {context}
    QUESTION: {question}
    ANSWER:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", chain_type_kwargs={"prompt": prompt})

# Extract text from HTML
def html_to_text(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()

# Request Models
class QueryRequest(BaseModel):
    query: str
    html_content: str | None = None

class DocumentRequest(BaseModel):
    html_content: str

# Root Endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Legal RAG Chatbot API"}

# Add Legal Documents to FAISS
@app.post("/add_document/")
async def add_document(doc: DocumentRequest):
    """Indexes the document for retrieval"""
    global vector_store
    plain_text = html_to_text(doc.html_content)

    with faiss_lock:
        if vector_store is None:
            vector_store = FAISS.from_texts([plain_text], embedding_model)
        else:
            vector_store.add_texts([plain_text])
        vector_store.save_local("embeddings")

    return {"message": "Document indexed successfully!"}

# Answer Legal Questions using RAG
@app.post("/ask/")
async def answer_query(request: QueryRequest):
    """Processes user queries and retrieves relevant legal documents for AI response"""
    if vector_store is None:
        raise HTTPException(status_code=404, detail="No documents indexed!")

    rag_chain = get_rag_chain()
    response = rag_chain({"question": request.query})  # 🔹 Fixed key

    return {"answer": response["result"]}

# Run server: uvicorn app.main:app --reload
