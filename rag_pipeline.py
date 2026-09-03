# rag_pipeline.py (Google Gemini CLIENT, OPENAI-COMPATIBLE, CLOUD COMPATIBLE)
import os
import time
from openai import OpenAI  # Gemini provides OpenAI-compatible API access
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder  # Used for Reranker
from dotenv import load_dotenv

# Load environment variables from .env file FIRST (local dev only —
# on Streamlit Cloud, secrets are injected by app.py before this module loads)
load_dotenv()

# ---------------- CONFIG ----------------
# NOTE: change the extension below if your file isn't a PDF.
DOCUMENT_FILE = "icao_doc_9157_aerodromedesignmanual.pdf"
PERSIST_DIRECTORY = "./aerodrome_vector_db"

# LLM for FINAL ANSWER GENERATION (Google Gemini API)
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_TEMPERATURE = 0.3           # lower = more literal/grounded, less "creative"
LLM_MAX_TOKENS = 500

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

RETRIEVER_TOP_K = 10
RERANK_TOP_K = 4

### OPTIMIZATION: Reranking toggle
USE_RERANKING = True  # False is fastest!

# ---------------- API KEY SETUP ----------------
# !!! CRITICAL FOR PUBLIC HOSTING: Rely ONLY on GEMINI_API_KEY environment variable !!!
# NEVER hardcode a real key in this file. Locally it comes from .env (see .env.example).
# On Streamlit Cloud it should come from st.secrets, bridged into os.environ by app.py.
# ---------------------------------------------------

# ---------------- GLOBAL PRELOADED MODELS ----------------
print("Initializing Google Gemini Client (one-time)...")
try:
    _api_key = os.getenv("GEMINI_API_KEY")
    if not _api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    GLOBAL_LLM = OpenAI(api_key=_api_key, base_url=GEMINI_BASE_URL)
except Exception as e:
    print(f"Error initializing Gemini client. Check GEMINI_API_KEY: {e}")
    GLOBAL_LLM = None

### OPTIMIZATION: Load CrossEncoder reranker ONCE
print("Loading CrossEncoder reranker (one-time)...")
GLOBAL_RERANKER = CrossEncoder("BAAI/bge-reranker-base")


# ---------------- PROMPT ----------------
SYSTEM_PROMPT_TEMPLATE = """
You are an expert assistant for the ICAO Doc 9157 AERODROME DESIGN MANUAL.
Answer the user's question using ONLY the information from the retrieved context below.

INSTRUCTIONS:
- Provide ONE short, direct sentence or two.
- Summarize using the context.
- If answer is not present: "The information is not available in the Aerodrome Design Manual excerpt provided."
- Do NOT add external knowledge.

CONTEXT:
{context}
"""
USER_PROMPT_TEMPLATE = "QUESTION: {question}\n\nClear answer:"


# ---------------- 1. PDF load + chunk ----------------
def load_and_chunk_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"{pdf_path} not found!")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    return chunks


def get_embeddings_function():
    """Initializes the cloud-compatible Sentence Transformer embeddings."""
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# ---------------- 2. Vector DB ----------------
def build_vector_db(chunks):
    embeddings = get_embeddings_function()
    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print("Vector DB created and saved successfully.")
    return vectordb


def load_or_build_vector_db():
    embeddings = get_embeddings_function()
    if os.path.exists(PERSIST_DIRECTORY):
        vectordb = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        print("Loaded existing vector DB.")
    else:
        print("Vector DB not found. Building new DB...")
        if not os.path.exists(DOCUMENT_FILE):
            raise FileNotFoundError(f"Source document '{DOCUMENT_FILE}' not found. Cannot build database.")
        chunks = load_and_chunk_pdf(DOCUMENT_FILE)
        vectordb = build_vector_db(chunks)
    return vectordb


# ---------------- 3. Retrieval ----------------
def retrieve_initial(query, vectordb, top_k=RETRIEVER_TOP_K):
    return vectordb.similarity_search(query, k=top_k)


# ---------------- 4. Reranking ----------------
def rerank_results(query, retrieved_docs, top_k=RERANK_TOP_K):
    if not USE_RERANKING:
        return retrieved_docs[:top_k]

    pairs = [[query, doc.page_content] for doc in retrieved_docs]
    scores = GLOBAL_RERANKER.predict(pairs)
    ranked = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
    return [doc for score, doc in ranked[:top_k]]


# ---------------- 5. LLM answer (Google Gemini) ----------------
def generate_answer(query, reranked_docs):
    if GLOBAL_LLM is None:
        raise ConnectionError("Gemini client failed to initialize. Please check the GEMINI_API_KEY secret.")

    context = "\n\n---\n\n".join([doc.page_content for doc in reranked_docs])

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    user_prompt = USER_PROMPT_TEMPLATE.format(question=query)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    completion = GLOBAL_LLM.chat.completions.create(
        model=GEMINI_MODEL,
        messages=messages,
        temperature=GEMINI_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        stream=False
    )

    answer_text = completion.choices[0].message.content
    answer = answer_text.replace("\n", " ").strip()

    # Enforce short answer rule
    if len(answer.split(". ")) > 2:
        temp_sentences = answer.split(". ")
        answer = ". ".join(temp_sentences[:2])
        if not answer.endswith('.'):
            answer += '.'

    return answer


# ---------------- 6. Main pipeline ----------------
def answer_question(query, vectordb):
    retrieved = retrieve_initial(query, vectordb)
    reranked = rerank_results(query, retrieved)
    return generate_answer(query, reranked)


# ---------------- 7. CLI interactive ----------------
if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("!!! WARNING: GEMINI_API_KEY environment variable is not set. API calls may fail !!!")

    if GLOBAL_LLM is None:
        exit(1)

    try:
        vectordb = load_or_build_vector_db()
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: {e}")
        exit(1)

    print("\nAerodromeBot CLI ready! Type 'exit' to quit.")

    while True:
        q = input("👤 Your Question: ").strip()
        if q.lower() in ["exit", "quit"]:
            break

        if not q:
            print("Please enter a question.")
            continue

        try:
            start = time.time()
            ans = answer_question(q, vectordb)
            end = time.time()

            print("\n🤖 Answer:\n", ans)
            print(f"(Response time: {end-start:.2f}s)")
            print("-------------------------------")
        except Exception as e:
            print(f"Error during RAG pipeline execution: {e}")
