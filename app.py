import os
import requests
import chromadb
import streamlit as st

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="centered"
)


# -----------------------------
# Load API key
# -----------------------------

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")


# -----------------------------
# Load embedding model
# -----------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()


# -----------------------------
# Connect to ChromaDB
# -----------------------------

@st.cache_resource
def load_collection():

    client = chromadb.PersistentClient(
        path="chroma_db"
    )

    return client.get_collection(
        name="documind_v2"
    )


collection = load_collection()


# -----------------------------
# UI
# -----------------------------

st.title("🧠 DocuMind")

st.subheader("AI Document Research Assistant")

st.write(
    "Ask questions about your uploaded document "
    "and get answers with source references."
)


# -----------------------------
# Question input
# -----------------------------

question = st.text_input(
    "Ask a question about the document:"
)


# -----------------------------
# Ask button
# -----------------------------

if st.button("Ask DocuMind"):

    if not question:

        st.warning("Please enter a question.")

    else:

        # Create question embedding
        question_embedding = model.encode([question])


        # Retrieve relevant chunks
        results = collection.query(
            query_embeddings=question_embedding.tolist(),
            n_results=3
        )


        documents = results["documents"][0]
        metadatas = results["metadatas"][0]


        # Create context
        context_parts = []

        for document, metadata in zip(
            documents,
            metadatas
        ):

            context_parts.append(
                f"Source: {metadata['source']}, "
                f"Page: {metadata['page']}\n"
                f"{document}"
            )


        context = "\n\n".join(
            context_parts
        )


        # -----------------------------
        # Create prompt
        # -----------------------------

        prompt = f"""
You are DocuMind, an AI document assistant.

Answer the user's question using ONLY the
information provided in the document context.

Do not use outside knowledge.

If the answer cannot be found in the
provided context, say:

"The information is not available in the document."

Give a clear and concise answer.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""


        # -----------------------------
        # Call OpenRouter
        # -----------------------------

        url = "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "openrouter/free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }


        with st.spinner("Thinking..."):

            response = requests.post(
                url,
                headers=headers,
                json=data
            )


        # -----------------------------
        # Handle response
        # -----------------------------

        result = response.json()


        if "choices" not in result:

            st.error("Something went wrong.")

            st.write(result)

        else:

            answer = result[
                "choices"
            ][0]["message"]["content"]


            # -----------------------------
            # Display answer
            # -----------------------------

            st.markdown("### 💡 Answer")

            st.write(answer)


            # -----------------------------
            # Display sources
            # -----------------------------

            st.markdown("### 📚 Sources")

            seen_sources = set()

            for metadata in metadatas:

                source = metadata["source"]

                page = metadata["page"]

                source_text = (
                    f"{source} — Page {page}"
                )

                if source_text not in seen_sources:

                    st.write(
                        f"📄 {source_text}"
                    )

                    seen_sources.add(
                        source_text
                    )