# RAG Chat Bot — Smart Document Intelligence Assistant

A friendly Streamlit app that lets you upload PDF documents, extract their contents, and ask questions. The app uses Google Generative AI (Gemini) embeddings and a Chroma vector store to enable retrieval-augmented generation (RAG).

Key features

- Upload one or more PDFs and automatically extract text.
- Split documents into searchable chunks and store vectors in a local Chroma DB.
- Use Google Generative AI embeddings and chat model to answer questions about your PDFs.
- Lightweight UI built with Streamlit for quick local development and easy deployment.

Who is this for?

- Developers who want a simple RAG demo powered by Google Generative AI and LangChain.
- Teams who need a quick way to ask questions over a set of PDFs.

Quick start (local)

1. Clone the repo and move into the project folder:

```powershell
git clone https://github.com/Amryahmath/RAG-Chat-Bot.git
cd RAG-Chat-Bot
```

2. Create and activate a Python virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Provide your Google API key. You can set it in one of three ways (recommended: Streamlit secrets for deployment):

- Local `.env` file (do NOT commit):

```
GOOGLE_API_KEY=your_api_key_here
```

- Environment variable (PowerShell):

```powershell
$env:GOOGLE_API_KEY='your_api_key_here'
```

- Streamlit Cloud: go to `Manage app` → `Settings` → `Secrets` and add `GOOGLE_API_KEY`.

5. Run the app locally:

```powershell
streamlit run app.py
```

What the app does

- The app extracts text from uploaded PDFs using `PyPDF2`.
- Documents are split into chunks via LangChain's `RecursiveCharacterTextSplitter` and stored in a Chroma vector store under `chroma_db/`.
- When you ask a question, the app uses a retrieval-augmented chain (RetrievalQA) with Google Generative AI-based embeddings and chat model to produce an answer.

Files of interest

- `app.py` — Streamlit app and main logic.
- `requirements.txt` — Python dependencies used by the project.
- `chroma_db/` — Local Chroma DB directory (ignored in `.gitignore`).
- `import_test.py` — (optional) small script to verify imports locally.

Deployment notes (Streamlit Cloud)

- Make sure `requirements.txt` is up to date (this repo includes `PyPDF2` and `langchain-community`).
- Add `GOOGLE_API_KEY` to Streamlit secrets before starting the app.
- If you change dependencies, trigger a rebuild in Streamlit Cloud to refresh the environment.

Troubleshooting

- ModuleNotFoundError on deploy: Ensure the missing package is listed in `requirements.txt` and push the change.
- `Google API key not found` warning: confirm that `GOOGLE_API_KEY` is set in one of the supported locations (secrets, env var, `.env`).
- Slow processing for large PDFs: the app processes text in-memory and builds chunks — use smaller chunk sizes or pre-process large files offline for heavy workloads.

Security notes

- Do NOT commit your API keys. Use Streamlit Secrets or an environment variable on your host.
- `.env` is included in `.gitignore`.

Next steps and improvements

- Add tests and/or a CI job that runs `import_test.py` to catch dependency issues early.
- Add a clear UI element for showing which embeddings/model are in use and the vector DB status.
- Optionally support more document types (Word, plain text) via LangChain document loaders.

License

MIT — adapt this code as you like.
