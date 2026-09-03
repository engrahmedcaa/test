# Aerodrome Design Manual RAG Chatbot (Grok-powered)

RAG chatbot over ICAO Doc 9157 (Aerodrome Design Manual), using:
- SentenceTransformers embeddings + Chroma vector DB
- BAAI/bge-reranker-base for reranking
- xAI's Grok (`grok-4-fast`) for final answer generation
- Streamlit for the UI

## 0. First, get a fresh xAI API key

If you shared an xAI key anywhere outside this repo (chat, email, etc.), **revoke it** at
https://console.x.ai and generate a new one. Never put a real key in code or commit it to git.

## 1. Put the source PDF in the project folder

Place your document at the project root, named exactly:

```
icao_doc_9157_aerodromedesignmanual.pdf
```

(If your file has a different extension, edit `DOCUMENT_FILE` at the top of `rag_pipeline.py` to match.)

## 2. Test locally first

**Python version note:** LangChain/Chroma/sentence-transformers can lag behind the newest
Python releases. If you hit install errors on Python 3.14, create the venv with Python 3.11
or 3.12 instead (`py -3.11 -m venv venv` on Windows, or `python3.11 -m venv venv` on Mac/Linux) —
everything else below is identical.

```bash
# from inside the project folder
python -m venv venv

# activate it
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

Create your local secrets file:

```bash
copy .env.example .env      # Windows
cp .env.example .env        # Mac/Linux
```

Open `.env` and paste your real (new) key:

```
XAI_API_KEY=xai-...your-real-key...
```

Run it:

```bash
streamlit run app.py
```

**Verify:** your browser should open `http://localhost:8501`. The first run will take a while —
it's building the vector DB from the PDF (chunking + embedding). You'll see "Database Loaded.
AerodromeBot ready to answer!" once it's done. A folder called `aerodrome_vector_db/` will appear —
that's the cached vector store, so the next run starts instantly. Ask it a question about the manual
and confirm you get a grounded answer back.

If something breaks, run the CLI version instead for clearer error output:

```bash
python rag_pipeline.py
```

## 3. Push to GitHub

```bash
git init
git add .
git commit -m "Aerodrome Design Manual RAG chatbot"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

**Verify:** open your repo on GitHub and confirm `.env` is NOT there (it's excluded by
`.gitignore`) — only `.env.example` should be visible. Also confirm your PDF was uploaded
(GitHub free repos handle files up to 25MB fine via the web UI; use `git lfs` or a plain
`git push` from the CLI if it's larger and the web upload rejects it).

## 4. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click "New app", pick your repo/branch, and set the main file path to `app.py`.
3. Before (or right after) deploying, open **Advanced settings → Secrets** (or
   App settings → Secrets once it's deployed) and paste:

   ```toml
   XAI_API_KEY = "xai-...your-real-key..."
   ```

4. Click Deploy.

**Verify:** watch the build log — it should install `requirements.txt`, then show
"Database Loaded. AerodromeBot ready to answer!" in the running app. Ask it the same
test question you used locally and confirm the answer matches.

## Notes / things you may want to tune

- `XAI_TEMPERATURE` in `rag_pipeline.py` is set to `0.3` (fairly literal/factual). Raise it
  toward 0.7–1.0 if you want more natural-sounding phrasing, at some cost to strict groundedness.
- `RETRIEVER_TOP_K` / `RERANK_TOP_K` control how many chunks are retrieved vs. kept after
  reranking — raise these if answers seem to be missing relevant context.
- The first deploy on Streamlit Cloud will rebuild the vector DB from the PDF (same as local
  first run), so expect the first load to take a minute or two.
