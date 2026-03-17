"""
Ingestion pipeline: load PDFs → chunk text → embed → persist FAISS index.
"""

from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Path to the Harry Potter PDFs folder (one level up from this project)
PDF_DIR = Path(__file__).parent.parent / "Harry Potter"


def load_all_books() -> list:
    """Load all PDFs from the Harry Potter folder, one Document per page."""
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {PDF_DIR}")

    all_docs = []

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()

        # Attach a clean book title to each page's metadata
        book_title = pdf_path.stem  # filename without .pdf
        for doc in docs:
            doc.metadata["book"] = book_title
            doc.metadata["source"] = pdf_path.name

        all_docs.extend(docs)
        print(f"  -> {len(docs)} pages loaded")

    print(f"\nTotal pages loaded across all books: {len(all_docs)}")
    return all_docs


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""],
)


def chunk_documents(docs: list) -> list:
    """Split pages into overlapping chunks; drop empty/whitespace-only chunks."""
    chunks = _splitter.split_documents(docs)
    chunks = [c for c in chunks if c.page_content.strip()]
    print(f"Total chunks after splitting: {len(chunks)}")
    return chunks


FAISS_INDEX_DIR = Path(__file__).parent / "faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"


def build_vectorstore(chunks: list) -> FAISS:
    """Embed all chunks and persist a FAISS index to disk."""
    print(f"\nEmbedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(FAISS_INDEX_DIR))
    print(f"FAISS index saved to {FAISS_INDEX_DIR}/")
    return vectorstore


def load_vectorstore() -> FAISS:
    """Load a previously persisted FAISS index from disk."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


if __name__ == "__main__":
    docs = load_all_books()
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)
    print("\nDone. Vector store is ready.")
