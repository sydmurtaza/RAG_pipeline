import os
import shutil
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Configuration
PDF_PATH = "3. FYP-Handbook-2023.pdf"
CHROMA_PATH = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def ingest_docs():
    if not os.path.exists(PDF_PATH):
        print(f"Error: {PDF_PATH} not found.")
        return

    print(f"Loading {PDF_PATH}...")
    pages = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append({"text": text, "page": i + 1})

    print(f"Extracted {len(pages)} pages.")

    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    texts = []
    metadatas = []
    for page_data in pages:
        page_chunks = text_splitter.split_text(page_data["text"])
        for chunk in page_chunks:
            texts.append(chunk)
            metadatas.append({"page": page_data["page"]})

    print(f"Created {len(texts)} chunks.")

    print("Creating embeddings and indexing...")
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    
    # Remove existing db if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_texts(
        texts=texts,
        embedding=embedding_function,
        metadatas=metadatas,
        persist_directory=CHROMA_PATH
    )
    
    print(f"Done! VectorDB saved to {CHROMA_PATH}")

if __name__ == "__main__":
    ingest_docs()
