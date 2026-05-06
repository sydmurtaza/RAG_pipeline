import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate

# Configuration
CHROMA_PATH = "chroma_db"
THRESHOLD = 0.2
MODEL_NAME = "phi3"
EMBED_MODEL = "nomic-embed-text"

st.set_page_config(page_title="FYP Handbook Assistant", layout="wide")

st.title("📚 FYP Handbook Assistant")
st.markdown("Ask questions about the FAST-NUCES FYP process.")

# Initialize components
@st.cache_resource
def load_db():
    embedding_function = OllamaEmbeddings(model=EMBED_MODEL)
    if not os.path.exists(CHROMA_PATH):
        return None
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

@st.cache_resource
def load_llm():
    return ChatOllama(model=MODEL_NAME)

db = load_db()
llm = load_llm()

if db is None:
    st.error("VectorDB not found. Please run `python ingest.py` first.")
else:
    query = st.text_input("Enter your question:", placeholder="e.g., What are the required margins?")
    ask_button = st.button("Ask")

    if ask_button and query:
        with st.spinner("Searching and thinking..."):
            # Retrieve
            results = db.similarity_search_with_relevance_scores(query, k=5)
            filtered_results = [res for res in results if res[1] >= THRESHOLD]

            if not filtered_results:
                st.warning("I don’t have that in the handbook.")
            else:
                # Prepare Context
                context_text = ""
                for doc, score in filtered_results:
                    context_text += f"\n--- Page {doc.metadata['page']} ---\n{doc.page_content}\n"

                # Prompt
                prompt_template = PromptTemplate.from_template("""
You are a handbook assistant. Answer ONLY from the context provided below. 
Cite page numbers like "(p. X)". If the answer is not in the context, say you don't know.

Context: 
{top_chunks_text}

Question: {user_question}

Answer:""")
                prompt = prompt_template.format(user_question=query, top_chunks_text=context_text)

                # LLM Call
                response = llm.invoke(prompt)

                # Display Answer
                st.subheader("Answer:")
                st.write(response.content)

                # Sources
                with st.expander("Sources (page refs)"):
                    for doc, score in filtered_results:
                        st.markdown(f"**Page {doc.metadata['page']}** (Similarity: {score:.4f})")
                        st.text(doc.page_content)
