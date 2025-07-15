# === BACKEND of Text Simplification (FastAPI + OpenAI + Qdrant) ===

from fastapi import FastAPI, Query as FastQuery
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchRequest
from db_textsimplification import users_collection
from openai import OpenAI
import os
import re
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL") 
COLLECTION_NAME = "pdf_chunks"

openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_response(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text


model = SentenceTransformer("all-MiniLM-L6-v2")

class Query(BaseModel):
    query: str

def get_user_profile(user_id: str):
    user = users_collection.find_one({"user_id": user_id})
    return user if user else None

def generate_prompt(user_profile, query_text, context):
    preferred_output = ", ".join(user_profile.get("preferred_output", []))
    interests = ", ".join(user_profile.get("interests", []))
    return f"""
You are helping a student understand a complex concept.

Student Profile:
- Learning Style: {user_profile['learning_style']}
- Language Proficiency: {user_profile['language_proficiency']}
- Academic Level: {user_profile['academic_level']}
- Learning Speed: {user_profile['learning_speed']}
- Confidence Level: {user_profile['confidence_level']}
- Tech Familiarity: {", ".join(user_profile['tech_familiarity'])}
- Interests: {interests}
- Preferred Output: {preferred_output}
- Last Module Completed: {user_profile['last_module_completed']}

Instructions:
Simplify the content using the student's preferences. Use step-by-step explanation, examples, or visuals/metaphors if suitable.

Text to simplify:
\"\"\"{query_text}\"\"\"

Relevant context from the document:
{context}

Respond with a clear, personalized explanation.
"""

@app.post("/simplify")
async def simplify(query: Query, user_id: str = FastQuery(...)):
    user_profile = get_user_profile(user_id)
    if not user_profile:
        return {"error": "User not found."}

    query_vector = model.encode(query.query).tolist()

    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=3
    )

    context = "\n".join([hit.payload["text"] for hit in search_result])
    prompt = generate_prompt(user_profile, query.query, context)

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You simplify complex concepts for learners."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        simplified_text = clean_response(response.choices[0].message.content)
        return {"simplified": simplified_text}
    except Exception as e:
        return {"error": str(e)}
