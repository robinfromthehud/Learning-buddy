from fastapi import FastAPI
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
from models import *
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from constants import *
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import getpass

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))


print(os.getenv("GOOGLE_API_KEY"))
print("bitch ass nigga")
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")

os.environ["LANGSMITH_TRACING"] = "true"
if not os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your Langsmith API key: ")

api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.RI0LPaDwrGH_Jjpm2wh_TskXjZjtD_BSAt_2cn88cvs"
url = "https://7ad96036-d937-4859-b0cf-2f85302e1c1c.europe-west3-0.gcp.cloud.qdrant.io"


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
app = FastAPI()


@app.get("/")
async def root():
    return {"message":"Hello world"}

@app.post("/ask")
async def ask_query(data: Item):
    client = QdrantClient(url=url, api_key=api_key)

    qdrant = QdrantVectorStore(
        client=client,
        embedding=embeddings,
        collection_name=data.collection,
        vector_name="dense",
        retrieval_mode=RetrievalMode.DENSE,
    )

    results = qdrant.similarity_search(data.query)

    return {"results": [doc.page_content for doc in results]}







     