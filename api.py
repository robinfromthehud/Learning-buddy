from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
import os
import getpass
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
from google.genai import types
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import httpx
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from gtts import gTTS



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
                        4. Keep the explaination between 60-100 words.
                        5. Explain like you are daeling with a high-school child year old child.
                        6. It would be better if you answer answer in points.
                        7. Follow above given prompts strongly.
                    """
                ]
            )
        )


    print(f"Timestamp: {item.time_stamp}")

    if(item.tts_enabled==1):
        language = 'en'
        tts = gTTS(text=response.text,lang=language,slow=False,tld='co.in')
        tts.save("tts.mp3")
        return {
            'response_by_gemini': " ".join(response.text.replace("\n", " ").split()),
        }
    
    return {
        'message': "Working Fine!",
        'context_text': context_text,
        'segment_text': segment_text,
        'response_by_gemini': " ".join(response.text.replace("\n", " ").split()),
    }


import httpx
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from typing import List

ALLOWED_DOMAINS = []  # Removed domain restrictions

def get_web_sources(keywords: List[str], per_keyword: int = 3):
    search_results = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for keyword in keywords:
        # Simple Google search
        query = f"{keyword}"
        search_url = f"https://www.google.com/search?q={quote_plus(query)}&num=10"
        
        try:
            with httpx.Client(headers=headers, timeout=10.0, follow_redirects=True) as client:
                resp = client.get(search_url)
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Multiple selectors to catch Google search results
                results = soup.select("div.g h3 a") + soup.select("div.yuRUbf a") + soup.select("a[href^='http']")
                
                count = 0
                seen_urls = set()
                
                for tag in results:
                    href = tag.get("href")
                    if href and href.startswith("http") and href not in seen_urls:
                        # Skip Google's own URLs and other unwanted domains
                        if any(skip in href for skip in ["google.com", "youtube.com", "facebook.com", "twitter.com"]):
                            continue
                            
                        title = tag.get_text(strip=True)
                        if not title:
                            # Try to get title from parent elements
                            parent = tag.find_parent()
                            if parent:
                                title = parent.get_text(strip=True)
                        
                        if title and len(title) > 5:
                            search_results.append({
                                "keyword": keyword, 
                                "title": title[:100] + "..." if len(title) > 100 else title,
                                "url": href
                            })
                            seen_urls.add(href)
                            count += 1
                            
                        if count >= per_keyword:
                            break
                                
        except Exception as e:
            print(f"Error searching for {keyword}: {e}")
            continue

    return search_results

class PauseList(BaseModel):
    pauses: List[float]
    collection_name: str

@app.post("/pauses/")
def create_pauses(item: PauseList):
    ts = item.pauses
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
        model="gemini-2.0-flash",
        contents=f"Here is the query: {docs}. Give top keywords (at max 3) from the given query sentences for further reading and assessments.",
        config=types.GenerateContentConfig(
            system_instruction=[
                """
                You are an AI keyword extraction engine. Your task is to extract up to 3 space-separated keywords from a given text, based on the following rules:

                1. Only include terms that are conceptually difficult and directly related to Artificial Intelligence.
                2. Only output the final answer as a single string — one line, no newlines, no punctuation.
                3. The output must contain no more than three words, space-separated. Fewer is acceptable. No explanations.
                4. If no relevant AI terms are found, output nothing.
                5. Keywords must be standalone, meaningful concepts like: "backpropagation", "transformer", "gradient descent", etc.
                6. Never output newlines, bullet points, or extra formatting.

                Correct output examples:
                - backpropagation
                - gradient descent overfitting  
                - attention transformer reinforcement

                Incorrect examples:
                - Backpropagation\\nTransformer → ❌ newline
                - ["Transformer", "RNN"] → ❌ formatting
                - transformer, RNN → ❌ punctuation
                """
            ]
        )
    )
    
    # Process keywords to remove newlines and convert to list
    keywords_text = response.text.replace("\n", "").strip()
    keywords_list = keywords_text.split() if keywords_text else []
    
    # Get web sources based on keywords - like Perplexity
    web_sources = []
    if keywords_list:
        try:
            web_sources = get_web_sources(keywords_list, per_keyword=3)
        except Exception as e:
            print(f"Error fetching web sources: {e}")
    
    return {
        "top_keywords": keywords_list,
        "docs": docs,
        "sources": web_sources
    }