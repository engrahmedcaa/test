# app.py - Streamlit Chatbot Interface (Google Gemini-powered, cloud robust)
import os
import time
import streamlit as st

# --- 0. BRIDGE STREAMLIT SECRETS -> ENV VAR (must happen BEFORE importing rag_pipeline,
#     since rag_pipeline reads os.getenv("GEMINI_API_KEY") at import time) ---
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ.setdefault("GEMINI_API_KEY", st.secrets["GEMINI_API_KEY"])
except Exception:
    # st.secrets raises if no secrets.toml exists at all (e.g. pure local run using .env) — that's fine.
    pass

# Import the components from rag_pipeline.py (AFTER the secrets bridge above)
from rag_pipeline import load_or_build_vector_db, answer_question, DOCUMENT_FILE

# --- 1. SET UP THE RAG CORE ---


@st.cache_resource(show_spinner="Loading Aerodrome Design Manual Database...")
def setup_rag_bot():
    """Loads the vector database and necessary components once."""
    try:
        vectordb = load_or_build_vector_db()
        return vectordb
    except FileNotFoundError as e:
        st.error(f"Error: Required file not found. Ensure '{DOCUMENT_FILE}' is in your GitHub repository.")
        st.stop()
    except ConnectionError as e:
        st.error(f"Initialization Failed: {e}. Please check your GEMINI_API_KEY secret.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during RAG component loading: {e}")
        st.stop()


# --- 2. STREAMLIT UI SETUP ---

st.set_page_config(
    page_title="Aerodrome Design Manual Chatbot (Gemini-RAG)",
    layout="centered"
)
st.title("✈️ Aerodrome Design Manual RAG Chatbot (Gemini-Powered)")
st.caption("🚀 Ask questions about ICAO Doc 9157 — Aerodrome Design Manual. Responses powered by Google Gemini 2.5 Flash, running with Sentence Transformers and Reranking.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am an expert on the **ICAO Aerodrome Design Manual (Doc 9157)**. How can I help you today?"}
    ]

# --- 3. LOAD COMPONENTS (CACHED) ---
vectordb = setup_rag_bot()
st.success("Database Loaded. AerodromeBot ready to answer!")
time.sleep(1)
st.empty()

# --- 4. DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. HANDLE USER INPUT ---

if prompt := st.chat_input("Ask a question about the Aerodrome Design Manual..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        start_time = time.time()

        try:
            response = answer_question(prompt, vectordb)
            end_time = time.time()

            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.005)
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            st.caption(f"Time: {end_time - start_time:.2f}s | Model: Google Gemini (gemini-2.5-flash) | RAG Parameters: k=10, Rerank=4")

        except Exception as e:
            error_msg = f"An error occurred while fetching the answer: {e}"
            message_placeholder.error(error_msg)
            full_response = error_msg

    st.session_state.messages.append({"role": "assistant", "content": full_response})
