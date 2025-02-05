import os
import rag
from openai import OpenAI  # Import the OpenAI client
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from PyPDF2 import PdfReader
from typing import List
from dotenv import load_dotenv
import faiss  # Correct FAISS import

app = FastAPI()
load_dotenv()

# OpenAI API Key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FAISS Index & Metadata Storage
embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
embedding_dim = 1536  # Dimension of OpenAI embeddings
faiss_index = faiss.IndexFlatL2(embedding_dim)
doc_map = {}

# Uploads Directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text


def process_document(file_path: str):
    """Process PDF, extract text, split, embed, and store in FAISS."""
    global faiss_index, doc_map

    # Extract text from PDF
    text = extract_text_from_pdf(file_path)

    # Chunk the text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # Generate embeddings
    embeddings = embedding_model.embed_documents(chunks)
    embeddings_np = np.array(embeddings, dtype=np.float32)

    # Update FAISS index
    faiss_index.add(embeddings_np)

    # Store chunk mappings
    doc_map.update({i: chunks[i] for i in range(len(chunks))})


@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """Handle PDF upload, process document, and store embeddings."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Process document (Extract, Embed, Store)
    process_document(file_path)
    return {"message": f"Processed {file.filename} successfully!"}


def retrieve_documents(query: str, top_k=3) -> List[str]:
    """Retrieve relevant document chunks using FAISS similarity search."""
    query_embedding = embedding_model.embed_query(query)
    query_vector = np.array([query_embedding], dtype=np.float32)

    # Perform FAISS search
    _, indices = faiss_index.search(query_vector, top_k)
    retrieved_docs = [doc_map[i] for i in indices[0] if i in doc_map]
    return retrieved_docs


@app.post("/query/")
async def query_rag(query: str = Form(...)):
    """Retrieve documents and generate a response using OpenAI's GPT-4."""
    retrieved_docs = retrieve_documents(query)

    if not retrieved_docs:
        return {"response": "No relevant information found."}

    # Construct prompt with retrieved documents
    prompt = f"Using the following retrieved information: {retrieved_docs}, answer the query: {query}"

    # Call OpenAI GPT-4
    response = client.chat.completions.create(
        model="gpt-4",  # Specify the model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}  # Use the prompt you constructed
        ]
    )
    print(response)


    content = response.choices[0].message.content

    return {"response": content}
