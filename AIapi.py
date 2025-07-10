from fastapi import FastAPI, File, UploadFile, HTTPException
import tempfile
import os
import fitz
import open_clip
import torch
import pandas as pd
import tabula
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.agents import initialize_agent, Tool
from PIL import Image
import io
import uuid
from uuid import uuid4

app = FastAPI()

AZURE_OPENAI_API_KEY = "BU21ep2zab4JWbBEUOstRuNiz5vVb4IJO0VFsdnqtW0hb2UX5TokJQQJ99BAACHYHv6XJ3w3AAABACOGCIXn"
AZURE_OPENAI_ENDPOINT = "https://chatdrl-gpt.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4.1"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
PGVECTOR_CONN = "postgresql+psycopg://postgres:proplusV!4@localhost:5433/zen-ai"
COLLECTION_NAME = "pdf_chunks"

embeddings_endpoint = "https://chatdrl-gpt.openai.azure.com/"
embeddings_model_name = "text-embedding-3-large"
embeddingsDeployment = "text-embedding-3-large"
embeddings_api_key = "BU21ep2zab4JWbBEUOstRuNiz5vVb4IJO0VFsdnqtW0hb2UX5TokJQQJ99BAACHYHv6XJ3w3AAABACOGCIXn"
embeddings_api_version = "2024-02-01"

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')


async def process_pdf(file: UploadFile, vector_store: PGVector):

    
   
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    extracted_tables = tabula.read_pdf(temp_file_path, pages="all", multiple_tables=True)
    table_texts = [Document(page_content=table.to_csv(index=False)) for table in extracted_tables if isinstance(table, pd.DataFrame)]

    images = []
    pdf_document = fitz.open(temp_file_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            images.append(base_image["image"])

    image_docs = []
    for idx, image_bytes in enumerate(images):
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = preprocess(image_pil).unsqueeze(0)
        with torch.no_grad():
            embedding = model.encode_image(image)
            image_docs.append(Document(page_content=f"Embedding metadata for Image_{idx}: {embedding.numpy().flatten().tolist()}"))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    text_chunks = splitter.split_documents(docs)

    all_chunks = text_chunks + table_texts + image_docs
    pdf_document.close()
    os.remove(temp_file_path)
    for chunk in all_chunks:
     chunk.metadata["id"] = str(uuid4())
    

    vector_store.add_documents(all_chunks, ids=[doc.metadata["id"] for doc in all_chunks])

@app.post("/upload/")
async def upload_files(files: list[UploadFile] = File(...)):
    session_collection_name = f"pdf_chunks_{uuid.uuid4().hex}"
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-large",
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version="2024-02-01",
    )
    vectorstore = PGVector(
        embeddings=embeddings,
        connection=PGVECTOR_CONN,
        collection_name=session_collection_name
    )
    for file in files:
        await process_pdf(file,vectorstore)
    return {"message": "Files processed successfully", "collection": session_collection_name}

@app.post("/search/")
async def semantic_search(query: str, collection: str):
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=embeddingsDeployment,
        api_key=embeddings_api_key,
        azure_endpoint=embeddings_endpoint,
        api_version=embeddings_api_version
    )

    vectorstore = PGVector(
        embeddings=embeddings,
        connection=PGVECTOR_CONN,
        collection_name=collection
    )

    results = vectorstore.similarity_search(query, k=20)

    llm = AzureChatOpenAI(
        deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )

    prompt = f"Answer the query based on these texts:\n\n{''.join([res.page_content for res in results])}\n\nQuery: {query}\nAnswer:"

    answer = llm.invoke(prompt)
    return {"answer": answer.content, "sources": [res.page_content for res in results]}