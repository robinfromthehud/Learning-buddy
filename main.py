import whisperx
import gc
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from uuid import uuid4
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
import getpass
from google import genai
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv


print(os.path.exists("audio/LLMEB.mp3"))


if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
os.environ["LANGSMITH_TRACING"] = "true"
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

load_dotenv()

url = os.environ["url"]
api_key = os.environ["api_key"]


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

collection_name = "butter"
"""url = "https://7ad96036-d937-4859-b0cf-2f85302e1c1c.europe-west3-0.gcp.cloud.qdrant.io:6333"
api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.RI0LPaDwrGH_Jjpm2wh_TskXjZjtD_BSAt_2cn88cvs"""


device = "cpu" 
audio_file = "audio/LLMEB.mp3"
batch_size = 1 
compute_type = "int8" 

model = whisperx.load_model("large-v2", device, compute_type=compute_type)

audio = whisperx.load_audio(audio_file)
print("kuch toh chala 1")
result = model.transcribe(audio, batch_size=batch_size)
#print(result["segments"])  

gc.collect()
print("pehla collect hogya -1 ")

model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
#print(result["segments"])  

gc.collect()
print("doosra collect hogya")

diarize_model = whisperx.diarize.DiarizationPipeline(
    use_auth_token="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",
    device=device
)

print("diarize hogya")

diarize_segments = diarize_model(audio)
result = whisperx.assign_word_speakers(diarize_segments, result)

print("we won")

#print(diarize_segments)
#print(result["segments"])


client = QdrantClient(url=url, api_key=api_key, prefer_grpc=True)
client.delete_collection(collection_name=collection_name)

existing_collections = [c.name for c in client.get_collections().collections]
if collection_name not in existing_collections:
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    print("collection created")

def chunk_segments(segments, chunk_duration=10.0):
    chunks = []
    current_chunk = ""
    current_start = None
    current_end = None

    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        text = seg["text"]

        if current_start is None:
            current_start = start
            current_end = end
            current_chunk = text
        elif end - current_start <= chunk_duration:
            current_chunk += " " + text
            current_end = end
        else:
            chunks.append({
                "text": current_chunk.strip(),
                "start": current_start,
                "end": current_end
            })
            current_chunk = text
            current_start = start
            current_end = end

    if current_chunk:
        chunks.append({
            "text": current_chunk.strip(),
            "start": current_start,
            "end": current_end
        })

    return chunks

chunked_segments = chunk_segments(result["segments"], chunk_duration=5.0)

documents = []
for chunk in chunked_segments:
    doc = Document(
        page_content=chunk["text"],
        metadata={
            "start": chunk["start"],
            "end": chunk["end"],
            "source": "audio"
        }
    )
    documents.append(doc)

print("ye rhe docs")
print(documents)


uuids = [str(uuid4()) for _ in range(len(documents))]


qdrant = QdrantVectorStore(
    client=client,
    collection_name="butter",
    embedding=embeddings,
    retrieval_mode=RetrievalMode.DENSE,
)

qdrant.add_documents(documents=documents, ids=uuids)

client.create_payload_index(
    collection_name="butter",
    field_name="metadata.start",
    field_schema="float"
)

client.create_payload_index(
    collection_name="butter",
    field_name="metadata.end",
    field_schema="float"
)

question = "What are LLMs ?"

print("Question is: ?")
print(question)



timestamp = 360

timestamp_embedding = qdrant.similarity_search(
    query=question,
    k=1,
    filter=models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.start",
                range=models.Range(lte=timestamp)
            ),
            models.FieldCondition(
                key="metadata.end",
                range=models.Range(gte=timestamp)
            ),
        ]
    )
)

context_embeddings = qdrant.similarity_search(
    query=question,k=20
)


segment_text = "\n".join(doc.page_content for doc in timestamp_embedding)
context_text = "\n".join(doc.page_content for doc in context_embeddings)

print("segment text")
print(segment_text)

print("context_text")
print(context_text)


prompt = (
    f"""You are an expert assistant. Use ONLY the information provided in the context below to answer the question. You assist during lectures so the segment of
    lecture where the doubt aries is given and relevant context from the whole lecture is also given so treat segement given as of foremost priority to answer the query.\n"""
    f"Then after analysing whole context answer the query"
    f"Do not use any outside knowledge or make assumptions.\n\n"
    f"Lecture Segment: \n{segment_text}\n\n"
    f"Context:\n{context_text}\n\n"
    f"Question: {question}"
)

client2 = genai.Client(api_key="AIzaSyAONqsu5KEdXIo7cpoVqirkP_vGYQQWV4U")

response = client2.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt
)

print("this is gemini response")
print(response.text)
