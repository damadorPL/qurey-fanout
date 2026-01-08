# Gemini Query Fanout (Streamlit)

A Streamlit GUI that replicates the logic from `Query_fanout.ipynb`:

- Generates 10 semantically related questions using Gemini generate API
- Uses Gemini embeddings to compute cosine similarities and rank questions
- Allows downloading the top results as a JSON file

## Quick start (using `uv`)

1. Install `uv` (see https://docs.astral.sh/uv/)
2. In the project folder:

```bash
uv add streamlit requests numpy
uv run streamlit run app.py
```

## Quick start (pip)

```bash
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The Gemini API Key must be provided via the GUI (password field). The key is only used for API calls and not stored on disk.
- If you want to use a project environment manager, `uv` is supported and recommended for faster installs.
