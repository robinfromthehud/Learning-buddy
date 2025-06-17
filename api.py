from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
import os
import getpass
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from dotenv import load_dotenv
from google.genai import types
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from gtts import gTTS
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import base64
from fastapi.responses import JSONResponse
import requests
import xml.etree.ElementTree as ET
import urllib.parse, urllib.request
from typing import List
import json
import time
import arxiv



if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
os.environ["LANGSMITH_TRACING"] = "true"
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

load_dotenv()

url = os.environ["url"]
api_key = os.environ["api_key"]

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llmclient = genai.Client(api_key="AIzaSyAONqsu5KEdXIo7cpoVqirkP_vGYQQWV4U")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

elevenlabs = ElevenLabs(
    api_key="sk_255078d13bd58904406213b40a8f747b4347bc7c0c8c360e"
)


class Item(BaseModel):
    collection_name: str
    query_type: str
    time_stamp: float
    tts_enabled:bool

@app.post("/items/")
def ask_query(item: Item):
    client = QdrantClient(url=url, api_key=api_key, prefer_grpc=True)
    qdrant = QdrantVectorStore(
        client=client,
        collection_name=item.collection_name,
        embedding=embeddings,
        retrieval_mode=RetrievalMode.DENSE,
    )
    
    timestamp_embedding = qdrant.similarity_search(
        query=item.query_type,
        k=2,
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.start",
                    range=models.Range(lte=item.time_stamp-5)
                ),
                models.FieldCondition(
                    key="metadata.end",
                    range=models.Range(gte=item.time_stamp-5)
                ),
            ]
        )
    )

    previous_embeddings = qdrant.similarity_search(
        query=item.query_type,
        k=15,
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.start",
                    range=models.Range(lte=item.time_stamp-5)
                )
            ]
        )
    ) 

    if previous_embeddings:
        previous_text = " ".join(
            " ".join(doc.page_content.replace("\n", " ").split()) 
            for doc in previous_embeddings
        )
    else:
        previous_text = ""

    if not previous_text.strip():
        return {
            'message': "No relevant context found",
            'context_text': "",
            'response_by_gemini': "Please start the lecture. You shouldn't have doubts even before lecture starts.",
        }

    if timestamp_embedding:
        segment_text = "\n".join(doc.page_content for doc in timestamp_embedding)
    else:
        segment_text = ""

    context_embeddings = qdrant.similarity_search(
        query=segment_text,
        k=10
    )


    if context_embeddings:
        context_text = " ".join(
            " ".join(doc.page_content.replace("\n", " ").split()) 
            for doc in context_embeddings
        )
    else:
        context_text = ""

    if not context_text.strip():
        return {
            'message': "No relevant context found",
            'context_text': "",
            'response_by_gemini': "No context available to answer the query.",
        }

    if(item.query_type=="Explain the current concept in simpler terms"):
        print("simpler terms it is")
        response = llmclient.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"Here is the text from the moment where the student asked for easy summarization: {segment_text} and here is the relevant context from the lecture: {context_text}",
        config=types.GenerateContentConfig(
            system_instruction=[
                    """
                        You are a concise and helpful doubt-solving assistant integrated with a video lecture platform.
                        Follow these system prompts strictly:
                        1. You have to summarize the given context in simpler words that the user can understand as it getting tough for user.
                        2. You're given a text part where the student has asked doubt so keep in mind that the segment's theory should be there in your response.
                        3. You're also provided with the relevant context from lecture so keep in mind the way you explain should be easier than that.
                        4. Keep the explaination between 30-60 words.
                        6. Follow above given prompts strongly.
                    """
                ]
            )
        )
    
    elif(item.query_type=="Provide a real-world example of the current concept"):
        print("example given")
        response = llmclient.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Here is the text from the moment where the student asked for examples: {segment_text} and here is the relevant context from the lecture for reference: {context_text}",
            config=types.GenerateContentConfig(
                system_instruction=[
                    """
                        You are a concise and helpful doubt-solving assistant integrated with a video lecture platform.
                        Follow these system prompts strictly:
                        1. You have to provide real-life examples to the user on the topic that is running in the lecture.
                        2. You're given a text part from the moment where the student has asked examples so keep in mind that the examples should relate to that topic.
                        3. You're also provided with the relevant context from lecture so keep in mind the way you explain should be easier than that.
                        4. Keep the explaination between 30-60 words.
                        6. Follow above given prompts strongly.
                    """
                ]
            )
        )
    
    else:
        print("summary till now")
        response = llmclient.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Here is the text transcript of the lecture required to summarize:{context_text}",
            config=types.GenerateContentConfig(
                system_instruction=[
                    """
                        You are a concise and helpful doubt-solving assistant integrated with a video lecture platform.
                        Follow these system prompts strictly:
                        1. You have to provide a brief summary of what has happened in the lecture till now.
                        2. You're given a text part from the moment where the student and also the previous context of the lecture.
                        3. You're also provided with the relevant context from lecture so keep in mind the way you explain should be easier than that.
                        4. Keep the explaination between 40-60 words.
                        5. Explain like you are daeling with a high-school child year old child.
                        6. It would be better if you answer answer in points.
                        7. use - for starting a new line for points and do not use any other special symbol.
                        7. Follow above given prompts strongly.
                    """
                ]
            )
        )


    print(f"Timestamp: {item.time_stamp}")

    if(item.tts_enabled==1):
        audio_bytes = elevenlabs.text_to_speech.convert(
            text=response.text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        audio_bytes_combined = b"".join(audio_bytes)
        audio_base64 = base64.b64encode(audio_bytes_combined).decode('utf-8')
        return JSONResponse(content={
            'response_by_gemini': " ".join(response.text.replace("\n", " ").split()),
            'audio_base64':audio_base64
        })
    
    return {
        'message': "Working Fine!",
        'context_text': context_text,
        'segment_text': segment_text,
        'response_by_gemini': " ".join(response.text.replace("\n", " ").split()),
    }

class PauseList(BaseModel):
    timestamps: List[float]
    collection_name: str

import requests
import xml.etree.ElementTree as ET
import time

@app.post("/pauses/")
def create_pauses(item: PauseList):
    print("pause toh chala")
    ts = item.timestamps
    docs = ""
    
    client = QdrantClient(url=url, api_key=api_key, prefer_grpc=True)
    qdrant = QdrantVectorStore(
        client=client,
        collection_name=item.collection_name,
        embedding=embeddings,
        retrieval_mode=RetrievalMode.DENSE,
    )
    
    for i in ts:
        timestamp_embedding = qdrant.similarity_search(
            query="",
            k=1,
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.start",
                        range=models.Range(lte=i)
                    ),
                    models.FieldCondition(
                        key="metadata.end",
                        range=models.Range(gte=i)
                    ),
                ]
            )
        )
        
        if timestamp_embedding:
            
            doc = timestamp_embedding[0].page_content.replace("\n", " ").strip()
            
            doc = " ".join(doc.split())
            docs += doc + " "
    
    
    docs = docs.strip()

    response = llmclient.models.generate_content(
        contents=f"Here is the part of transcript of the lecture where the student asked doubts so you have to focus on recommending topics based on this {docs}",
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
                system_instruction=[
                    """
                        You are a concise and helpful doubt-solving assistant integrated with a video lecture platform.
                        Follow these system prompts strictly:
                        1. You will be provided some lines of the lecture where student asked doubts.
                        2. You are required to go through those lines and recommend that learner can read further through articles and research papers.
                        3. You just have to recommend the key concepts that can be explored further.
                        4. You can recommend at max 1 sentence, follow this strictly.
                        5. The recommendation should be single sentence like "attention in transformers".
                        6. it shouldn't contain any special characters, just a plain simple sentence.
                        7. Follow above given prompts strongly.
                    """
                ]
        )
    )


    client = arxiv.Client()
    query_text = response.text.strip().replace("\n", " ").replace("\r", " ")
    search = arxiv.Search(query=query_text,max_results=10)

    arxiv_titles = []
    arxiv_links = []
    for result in client.results(search):
        arxiv_titles.append(result.title)
        arxiv_links.append(result.entry_id)

    response = llmclient.models.generate_content(
        contents=f"Here is the title for which research paper were searched {query_text} and here the text from lecture where user asked assitance {docs} and here are the titles {arxiv_titles}",
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
                system_instruction=[
                    """
                        Follow the given rules strictly:
                        1. You will be provided with a list of titles of research papers.
                        2. You have to give the index of that reserach paper that is most relevant to the title for which paper were searched.
                        3. The search query was formed using text transcript of timestamp where user asked for assistance so also have to consider that piece of transcript. 
                        4. You have to answer just a single character that is the index of the title (follow zero based indexing).
                    """
                ]
        )
    )
    #print(response.text)
    for result in client.results(search):
        print(result.title, result.entry_id)

        return {
            'title':arxiv_titles[int(response.text)],
            'url':arxiv_links[int(response.text)],
        }
 

class debate(BaseModel):
    topic:str


@app.post("/debate-topic/")
def topic(item:debate):
    response = llmclient.models.generate_content(
        contents=f"here is the word on which you have to give the topic of debate : {item.topic}",
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
                system_instruction=[
                    """
                     You are a concise and helpful doubt-solving assistant integrated with a video lecture platform.
                        Follow these system prompts strictly:
                        1. You will be given a word and based on that you have to suggest a topic for debate.
                        2. The debate will be taking place between the user and LLM.
                    """
                ]
        )
    )

    return {
        'topic':response.text
    }


class counters(BaseModel):
    userresponse:str
    botresponse:str

class Topic(BaseModel):
    topic:str

@app.post("/quiz/")
async def get_quiz_questions_flexible(item: Topic):
    try:
        response = llmclient.models.generate_content(
            contents=f"""Generate 7 multiple choice questions about: {item.topic}
            
            Return ONLY the questions in JSON format like this:
            [
                {{
                    "question": "What is Python?",
                    "options": ["Programming language", "Animal", "Car brand", "Food"],
                    "correctAnswer": 0
                }},
                {{
                    "question": "What is 2+2?", 
                    "options": ["3", "4", "5", "6"],
                    "correctAnswer": 1
                }}
            ]""",
            model="gemini-2.0-flash"
        )
        
        import json
        import re
        
        # Try to extract JSON from response
        response_text = response.text.strip()
        
        # Look for JSON array in the response
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            questions_data = json.loads(json_str)
            
            # Validate the structure
            valid_questions = []
            for q in questions_data:
                if (isinstance(q, dict) and 
                    'question' in q and 'options' in q and 'correctAnswer' in q and
                    len(q['options']) == 4 and
                    isinstance(q['correctAnswer'], int) and
                    0 <= q['correctAnswer'] <= 3):
                    valid_questions.append(q)
            
            if valid_questions:
                return {"questions": valid_questions}
        
        raise HTTPException(status_code=400, detail="Could not parse JSON response")
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in response: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))