# Multimodal PDF RAG

Ingest PDFs - r&eacute;sum&eacute;s, invoices, reports - as text, tables **and embedded
images**, index them in pgvector with Azure OpenAI embeddings, then query and match
against them. The same retrieval core is exposed three ways: a FastAPI service, a
Streamlit app, and an MCP server.

## What it does

- **Parse** - PyMuPDF for text, `tabula` for tables pulled out of PDFs
- **Embed the images too** - `open_clip` (ViT-B-32) embeds figures and scanned regions,
  so a diagram is retrievable rather than only the text around it
- **Index** - Azure OpenAI `text-embedding-3-large` into `pgvector` through LangChain's
  `PGVector`, one collection per upload
- **Answer** - Azure OpenAI `gpt-4.1` behind a LangChain agent with tools over the
  retrieved chunks
- **Match** - the FastAPI backend pulls candidates from the Lever ATS, matches them
  against the uploaded document, and can send the shortlist over SMTP

## Entry points

| File | What it is |
| --- | --- |
| `AIapi.py` | FastAPI - upload, parse, embed, store in pgvector, agentic Q&A |
| `streamlitMCPRAGServer.py` | Streamlit UI plus a **FastMCP** server exposing retrieval as MCP tools |
| `streamlitfrontend.py` | Streamlit client driving the FastAPI `/upload/` and `/search/` endpoints |
| `fastapibackend/main.py` | FastAPI - Azure OpenAI + Lever ATS matching + emailing the shortlist |
| `frontend/` | React UI (`MatchedCandidates.js`) for the matched results |
| `multi_modal_aws.ipynb` | Notebook covering the multimodal path on AWS |

## How retrieval works

```
PDF -> text (PyMuPDF) + tables (tabula) + images (open_clip ViT-B-32)
         |
         v
   chunks -> embed (text-embedding-3-large) -> pgvector
         |
    query -> similarity search -> LangChain agent (tools) -> answer
```

## Stack

- Azure OpenAI: `gpt-4.1` for generation, `text-embedding-3-large` for embeddings
- `pgvector` on Postgres as the vector store
- LangChain agents for the tool-using question path
- FastMCP for publishing retrieval as MCP tools
- FastAPI + Streamlit + React as the three front doors
- PyMuPDF, tabula, open_clip and torch for document and image handling

## Configuration

Everything is read from the environment; no keys are in the repo.

```
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1
PGVECTOR_CONN=postgresql+psycopg://<user>:<password>@<host>:5432/<db>
LEVER_API_KEY=
GMAIL_SMTP_SERVER=
GMAIL_SMTP_PORT=587
GMAIL_EMAIL=
GMAIL_APP_PASSWORD=
```

`fastapibackend/main.py` loads a `.env` file when one is present, but anything
already exported takes precedence, so it stays optional in a container.

## Run it

pgvector:

```
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16
```

API:

```
pip install -r requirements.txt
uvicorn AIapi:app --reload --port 8000
```

Streamlit UI + MCP server:

```
streamlit run streamlitMCPRAGServer.py
```

React client:

```
cd frontend
npm install
npm start
```

## Notes

- Each upload gets its own pgvector collection, so separate document sets cannot
  bleed into each other.
- The MCP server is what makes the retrieval reusable outside this app - an agent
  elsewhere can call the search tools directly.