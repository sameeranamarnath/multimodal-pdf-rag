import streamlit as st
import os
import threading
import asyncio
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings,AzureOpenAI
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import initialize_agent, Tool
from langchain_core.rate_limiters     import InMemoryRateLimiter
from fastmcp import FastMCP

rate_limiter = InMemoryRateLimiter(
    requests_per_second=1,  # <-- Can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)


# ==== Azure OpenAI Configuration ====
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4.1"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
PGVECTOR_CONN = os.environ["PGVECTOR_CONN"]
COLLECTION_NAME = "pdf_chunks"


if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []
if "last_report" not in st.session_state:
    st.session_state["last_report"] = {}

st.title("PDF Analytics MCP Server (Azure OpenAI, LangChain, pgvector)")

uploaded_files = st.file_uploader("Upload PDFs (large files supported)", type="pdf", accept_multiple_files=True)
if uploaded_files:
    docs = []
    for file in uploaded_files:
        # Save to disk to avoid memory spikes with large files
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(temp_path)
        # Load and process page by page to minimize memory usage
        for doc in loader.load():
            docs.append(doc)
        os.remove(temp_path)
    st.success(f"Loaded {len(docs)} pages.")

    # Chunking for large files
    splitter = RecursiveCharacterTextSplitter(chunk_size=4500, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    st.session_state["chunks"] = chunks
    st.info(f"Chunked into {len(chunks)} segments.")
    
    embeddings_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    embeddings_model_name = "text-embedding-3-large"
    embeddingsDeployment = "text-embedding-3-large"
    embeddings_api_key = os.environ["AZURE_OPENAI_API_KEY"]
    embeddings_api_version = "2024-02-01"

    # Embeddings and vectorstore (Azure OpenAI)
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=embeddingsDeployment,
        api_key=embeddings_api_key,
        azure_endpoint=embeddings_endpoint,
        api_version=embeddings_api_version,
    )
    vectorstore = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_string=PGVECTOR_CONN,
        collection_name=COLLECTION_NAME,
        
    )
    st.session_state["vectorstore"] = vectorstore
    st.success("Documents embedded and stored.")

def search_tool(query):
    vectorstore = st.session_state["vectorstore"]
    if not vectorstore:
        return "No documents loaded."
    results = vectorstore.similarity_search(query, k=5)
    return "\n\n".join([doc.page_content for doc in results])

tools = [
    Tool(name="SemanticSearch", func=search_tool, description="Semantic search over PDFs")
]
llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    rate_limiter=rate_limiter,
    temperature=0,
)


agent = initialize_agent(tools, llm, verbose=True)

user_query = st.text_input("Ask a question about the PDFs:")
if user_query and st.session_state["vectorstore"]:
    answer = agent.run(user_query)
    st.write("**Answer:**", answer)

    report = {
        "query": user_query,
        "answer": answer,
        "num_chunks": len(st.session_state["chunks"]),
    }
    st.session_state["last_report"] = report
    st.write("**Analytics Report:**")
    st.json(report)


mcp = FastMCP("pdf_analytics_server") 


@mcp.tool("get_pdf_analytics")
def get_pdf_analytics(query: str = ""):
    if not st.session_state["vectorstore"]:
        return {"error": "No documents loaded."}
    if not query:
        return {"error": "No query provided."}
    answer = agent.run(query)
    report = {
        "query": query,
        "answer": answer,
        "num_chunks": len(st.session_state["chunks"]),
    }
    st.session_state["last_report"] = report
    return report

st.markdown("---")
st.info(
    '''DRL Rag MCP Server for PDF Analytics \n,
    The `get_pdf_analytics` tool returns analytics for a given query over the uploaded PDFs \nc. 
    Efficient chunking and disk buffering support large file processing.'''
)

