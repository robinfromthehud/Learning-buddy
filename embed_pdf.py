#from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import uuid

# 1. Extract text
# def extract_text_from_pdf(pdf_path):
#    reader = PdfReader(pdf_path)
#    return "\n".join(page.extract_text() for page in reader.pages)

def extract_text_from_txt(pdf_path):
    with open(pdf_path, "r", encoding="utf-8") as f:
        return f.read()
    


# 2. Chunk text
def chunk_text(text, max_length=500):
    sentences = text.split(". ")
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) < max_length:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks

# 3. Upload to Qdrant Cloud
def upload_chunks(chunks, model, client, collection_name):
    embeddings = model.encode(chunks).tolist()
    points = [
        {"id": str(uuid.uuid4()), "vector": emb, "payload": {"text": chunk}}
        for emb, chunk in zip(embeddings, chunks)
    ]
    client.upsert(collection_name=collection_name, points=points)

# === MAIN SCRIPT ===
pdf_path = "docc.txt"
text = extract_text_from_txt(pdf_path)
chunks = chunk_text(text)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔗 Qdrant Cloud setup
client = QdrantClient(
    url="https://966db0bd-6889-47bf-bd8a-dd54bb28ee83.europe-west3-0.gcp.cloud.qdrant.io:6333",  # Replace
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.DMDqdhFufh4b2RqQgKQV3o6vvBfbUO4S7cZZieqdYdU",                           # Replace
)

collection_name = "pdf_chunks"
if collection_name not in [col.name for col in client.get_collections().collections]:
    client.create_collection(
        collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

upload_chunks(chunks, model, client, collection_name)
