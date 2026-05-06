import os
import sys
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate

# Configuration
CHROMA_PATH = "chroma_db"
THRESHOLD = 0.2
MODEL_NAME = "phi3"
EMBED_MODEL = "nomic-embed-text"

def ask_question(query):
    embedding_function = OllamaEmbeddings(model=EMBED_MODEL)
    if not os.path.exists(CHROMA_PATH):
        print(f"Error: VectorDB not found at {CHROMA_PATH}. Please run ingest.py first.")
        return

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Retrieve
    # Chroma returns cosine distance, but similarity_search_with_relevance_scores should normalize it
    results = db.similarity_search_with_relevance_scores(query, k=5)
    filtered_results = [res for res in results if res[1] >= THRESHOLD]

    if not filtered_results:
        print("\nI don’t have that in the handbook.")
        return

    context_text = ""
    for doc, score in filtered_results:
        context_text += f"\n--- Page {doc.metadata['page']} (Similarity: {score:.4f}) ---\n{doc.page_content}\n"

    prompt_template = PromptTemplate.from_template("""
You are a handbook assistant. Answer ONLY from the context provided below. 
Cite page numbers like "(p. X)". If the answer is not in the context, say you don't know.

Context: 
{top_chunks_text}

Question: {user_question}

Answer:""")

    prompt = prompt_template.format(user_question=query, top_chunks_text=context_text)

    print("\nThinking...")
    llm = ChatOllama(model=MODEL_NAME)
    response = llm.invoke(prompt)

    print("\n" + "="*50)
    print("ANSWER:")
    print("="*50)
    print(response.content)
    print("="*50)
    
    print("\n[DEBUG] Sources retrieved:")
    for doc, score in filtered_results:
        print(f"- Page {doc.metadata['page']} (Score: {score:.4f})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("\nAsk a question about the FYP handbook: ")
    
    if query.strip():
        ask_question(query)
