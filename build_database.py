import os
import chromadb

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# 1. Read PDF page by page
# -----------------------------

pdf_path = "data/sample.pdf"

reader = PdfReader(pdf_path)

pages = []

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text()

    if text:
        pages.append({
            "text": text,
            "page": page_number
        })


print("Pages with text:", len(pages))


# -----------------------------
# 2. Create text splitter
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)


# -----------------------------
# 3. Create chunks
# -----------------------------

chunks = []
metadatas = []

for page in pages:

    page_chunks = splitter.split_text(page["text"])

    for chunk in page_chunks:

        chunks.append(chunk)

        metadatas.append({
            "source": "sample.pdf",
            "page": page["page"]
        })


print("Total chunks:", len(chunks))


# -----------------------------
# 4. Create embeddings
# -----------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(chunks)

print("Embeddings created!")


# -----------------------------
# 5. Create ChromaDB
# -----------------------------

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="documind_v2"
)


# -----------------------------
# 6. Store everything
# -----------------------------

collection.add(
    ids=[str(i) for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)


print("Database created successfully!")