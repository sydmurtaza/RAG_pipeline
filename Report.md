# Generative AI - Assignment #3: RAG: FYP Handbook Assistant

**Group Members:**
- Syed Hussain Murtaza (22i-1513)
- Affan Hameed (22i-2582)

## 1. System Overview
The system is a text-only RAG (Retrieval-Augmented Generation) pipeline designed to answer questions based on the **BS Final Year Project Handbook 2023**. It consists of an ingestion script, a retrieval engine, and a Streamlit-based user interface.

## 2. Technical Settings
- **Chunking Strategy**: `RecursiveCharacterTextSplitter`
    - **Chunk Size**: 1000 characters
    - **Chunk Overlap**: 100 characters
- **Embedding Model**: `nomic-embed-text` (via Ollama)
- **Vector Database**: `ChromaDB` (Local persistence)
- **LLM**: `Phi-3-mini` (via Ollama)
- **Retrieval settings (k)**: 5
- **Similarity Threshold**: 0.2 (Cosine similarity relevance score)

## 3. Implementation Details
- **Ingestion (`ingest.py`)**: Uses `pdfplumber` for robust text extraction from the PDF. It preserves page numbers in metadata for grounding.
- **Retrieval (`ask.py`)**: Computes cosine similarity between query and document embeddings. Chunks with relevance scores below the threshold are discarded.
- **Grounding**: The prompt instructs the LLM to answer *ONLY* from the provided context and cite specific page numbers using the format `(p. X)`.

## 4. Validation Q&A Examples

### Example 1: Margins and Spacing
**Question**: "What margins and spacing do we use?"
**Answer**: Top margin = 1.5", bottom margin = 1.0", left margin = 2.0", right margin = 1.0", line spacing = 1.5, paragraph spacing = 6 pts (p. 40) as per the BS Final Year Project Handbook 2023.

### Example 2: Endnotes Usage
**Question**: "How should endnotes like ‘Ibid.’ and ‘op. cit.’ be used?"
**Answer**: To use “Ibid.” (citing the same source), if only page numbers differ, add them after Ibid., e.g., "Ibid. p. XX". For multiple sources with intervening references, use “op. cit.” followed by the page number (p. 41).

## 5. UI Screenshot (Mockup Description)
The Streamlit interface (`app.py`) features:
- A clean header with a book icon.
- A prominent text input for student queries.
- An "Ask" button with a processing spinner.
- An answer panel that highlights the generated response.
- A collapsible **"Sources (page refs)"** section showing the exact text chunks used for the answer along with their similarity scores.
