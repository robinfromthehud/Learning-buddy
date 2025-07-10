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
from typing import Dict, List
from elevenlabs.client import ElevenLabs
import base64
from fastapi.responses import JSONResponse
from typing import List
import arxiv
from pymongo import MongoClient
import json
import re

MONGO_URI = "mongodb+srv://dbuser:dbuser2280@test.stzoarv.mongodb.net/"
DB_NAME = "learning-buddy"
COLLECTION_NAME = "test-1"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]



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

grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)


class Item(BaseModel):
    id:str
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

    user_doc = collection.find_one({"id": item.id})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found.")

    communication_preferences = user_doc.get("communication_preferences_summary", "No summary available.")
    if(item.query_type=="Explain the current concept in simpler terms"):
        print("simpler terms it is")
        response = llmclient.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"Here is the text from the moment where the student asked for easy summarization: {segment_text} and here is the relevant context from the lecture: {context_text} ad here is the user profile {communication_preferences}",
        config=types.GenerateContentConfig(
            system_instruction=[
                    """
                        You are a concise and helpful doubt-solving assistant integrated with a video lecture platform.
                        Follow these system prompts strictly:
                        1. You have to summarize the given context in simpler words that the user can understand as it getting tough for user.
                        2. You will be provided user communication preferences, you must follow those preferences to customize your response tailored to user.
                        3. You're given a text part where the student has asked doubt so keep in mind that the segment's theory should be there in your response.
                        4. You're also provided with the relevant context from lecture so keep in mind the way you explain should be easier than that.
                        5. Keep the explaination between 30-60 words.
                        6. Follow above given prompts strongly.
                        7. You must not target any specific race, community or religion and don't use any abusive language.
                    """
                ]
            )
        )
    
    elif(item.query_type=="Provide a real-world example of the current concept"):
        print("example given")
        response = llmclient.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Here is the text from the moment where the student asked for examples: {segment_text} and here is the relevant context from the lecture for reference: {context_text} Here are the communication preferences of user {communication_preferences}",
            config=types.GenerateContentConfig(
                system_instruction=[
                    """
                        You are a concise and helpful doubt-solving assistant integrated with a video lecture platform.
                        Follow these system prompts strictly:
                        1. You have to provide real-life examples to the user on the topic that is running in the lecture.
                        2. You're given a text part from the moment where the student has asked examples so keep in mind that the examples should relate to that topic.
                        3. You must respond according to the communication preferences of the user, that you will be provided with.
                        4. Keep the explaination between 30-60 words.
                        6. Follow above given prompts strongly.
                        11. You must not target any specific race, community or religion and don't use any abusive language.
                    """
                ]
            )
        )
    
    else:
        response = llmclient.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Here is the text transcript of the lecture required to summarize:{context_text} and here is the user communication preferences profile : {communication_preferences}",
            config=types.GenerateContentConfig(
                system_instruction=[
                    """
                        You are a concise and helpful doubt-solving assistant integrated with a video lecture platform.
                        Follow these system prompts strictly:
                        1. You have to provide a brief summary of what has happened in the lecture till now.
                        2. You're given a text part from the moment where the learner and also the previous context of the lecture.
                        3. You're also provided with the relevant context from lecture.
                        4. Keep the explaination between 40-60 words.
                        5. You are provided with the communication preferences of the user so you must maintain your response tone according user preferences.
                        6. You must give answers only, without stating what type of tone you will use, you are answering a learner directly.
                        7. It would be better if you answer in points.
                        8. use - for starting a new line for points and do not use any other special symbol.
                        9. Follow above given prompts strongly.
                        10. You must not target any specific race, community or religion and don't use any abusive language.
                    """
                ]
            )
        )

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
                        8. You must not target any specific race, community or religion and don't use any abusive language.
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
 
class user(BaseModel):
    id:str

@app.post("/choices/")
def choices(item:user):
    user_doc = collection.find_one({"id": item.id})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found.")

    user_summary = user_doc.get("academic_profile_summary", "No summary available.")

    print(user_summary)

    response = llmclient.models.generate_content(
        contents=f"Here is the academic context summary based on which you have to suggest topics {user_summary}",
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
                system_instruction=[
                    """
                        Follow these system prompts strictly:
                        1. You are provided with academic summary and by that you must assess user's knowledge base.
                        2. Based on the users prerequisite you have to suggest a list of topics of AI on which problem statement can be made.
                        3. Your topics must be relevant to the users current knowledge level, not easy and not hard.
                        4. You must give at least 2 and at max 8 topics.
                        5. You response must follow the exact pattern- "Topic1  Topic2  Topic3".
                        6. The topics must be separated by a double space.
                        7. You must not target any specific race, community or religion and don't use any abusive language.
                    """
                ]
        )
    )

    topics_list = response.text.strip().split("  ")

    return {
        'topics': topics_list
    }

class debate(BaseModel):
    id:str
    topic:str


@app.post("/debate-topic/")
def topic(item:debate):
    user_doc = collection.find_one({"id": item.id})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found.")

    academic_profile = user_doc.get("academic_profile_summary", "No summary available.")
    learning_profile = user_doc.get("learning_profile_summary", "No summary available.")
    response = llmclient.models.generate_content(
        contents=f"here is the word on which you have to give the topic of debate : {item.topic}, here is the academic profile : {academic_profile}, here is the learning profile : {learning_profile}",
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
                system_instruction=[
                    """
                    You are an assistant that generates debate topics.
                    1. You will be given a **single concept or keyword**.
                    2. You must generate a debate topic in the form of a **single-sided claim** (not "X or Y").
                    For example:
                    - "AI will make teachers obsolete"
                    - "AI improves decision making"
                    - "ChatGPT is disrupting education"
                    - "Data privacy is at risk due to AI"
                    3. Your topic must make a bold statement that can be debated IN FAVOUR or AGAINST.
                    4. Return only the topic. Do not include explanations or context.
                    5. You will be provided with the academic and learning profile of the user and you must use that for suggesting the problem statement.
                    6. You must not target any specific race, community or religion and don't use any abusive language.
                    """
                ]
        )
    )

    return {
        'topic':response.text
    }

class DebateResponseRequest(BaseModel):
    user_id: str
    topic: str
    user_position: str  
    bot_position: str   
    debate_history: List[Dict[str, str]]
    end_debate: bool = False

@app.post('/debate-response/')
def debateresponse(item: DebateResponseRequest):
    user_doc = collection.find_one({"id": item.user_id})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found.")

    preferences = user_doc.get("communication_preferences_summary", "No summary available.")
    academic = user_doc.get("academic_profile_summary", "No summary available.")


    history_formatted = "\n".join([f"{msg['speaker']}: {msg['message']}" for msg in item.debate_history])
    content_text = (
        f"Debate Topic: {item.topic}\n"
        f"User is arguing: {item.user_position}\n"
        f"AI must argue: {item.bot_position}\n"
        f"Conversation so far:\n{history_formatted}\n"
        f"User Preferences: {preferences}\n"
        f"User Academic Profile: {academic}\n"
    )

    if item.end_debate:
        content_text += (
            "\n\nNow the debate has ended. Based on the entire conversation, declare the winner in exactly two lines. "
            "Justify your choice clearly and respectfully. Be fair, but decisive."
        )

    response = llmclient.models.generate_content(
        contents=content_text,
        model="gemini-2.0-flash", 
        config=types.GenerateContentConfig(
            system_instruction=[
                f"""
                You are participating in a formal AI debate.
                1. Take a strict stand on the position: {item.bot_position}.
                2. Use a communication style matching the user's preferences.
                3. Don't use abucive or aggressive language.
                4. If the user uses inappropriate language, respond with: "Let's keep this professional."'
                3. Adapt your vocabulary and complexity to the user's academic level.
                4. Do not represent or explore both sides—argue only your assigned side.
                5. If the user tries to change topic, say: "Let's not get diverted."
                6. Stay professional and respectful throughout.
                7. Do not use inappropriate or aggressive language.
                8. Limit each response to a maximum of 50 words.
                9. If end_debate is True, analyze the debate history and declare a clear winner in 2 lines.
                10. Act like a real person, not an AI model.
                11. You must not target any specific race, community or religion and don't use any abusive language and if user uses any abusive language then you must say "Let's keep this professional."
                """
            ],
            tools=[grounding_tool],
        )
    )

    return {'response': response.text}

class counters(BaseModel):
    userresponse:str
    botresponse:str



class user(BaseModel):
    id: str

@app.post("/get-user-courses/")
async def get_user_courses(item: user):
    user_doc = collection.find_one({"id": item.id})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found.")
    
    courses = user_doc.get("courses_covered", [])
    return {"courses": courses}



class QuizRequest(BaseModel):
    id: str
    selected_courses: list[str]

@app.post("/get-quiz-topics/")
async def get_quiz_topics(item: QuizRequest):
    user_doc = collection.find_one({"id": item.id})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found.")
    
    user_summary = user_doc.get("academic_profile_summary", "No summary available.")
    courses_covered = user_doc.get("courses_covered", [])

    selected_courses_str = ", ".join(item.selected_courses)

    prompt = f"""
    Here is the academic context summary: {user_summary}
    
    The user has selected the following course(s) for quiz generation: {selected_courses_str}.
    """

    response = llmclient.models.generate_content(
        contents=prompt,
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=[
                """
                Follow these system prompts strictly:
                1. You are provided with academic summary and selected course(s). Assess user's relevant knowledge base.
                2. Based on the user's knowledge in the selected course(s), suggest a list of quiz topics in AI based only on the selected courses.
                3. Your topics must be relevant to the user's current level — not too easy, not too hard.
                4. Provide at least 2 and at most 8 topics.
                5. Format your response as: "Topic1  Topic2  Topic3".
                6. Topics must be separated by double spaces only.
                """
            ]
        )
    )

    topics_list = response.text.strip().split("  ")
    return {'topics': topics_list}


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
        
        response_text = response.text.strip()
        
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            questions_data = json.loads(json_str)
            
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
    


class StudentRequest(BaseModel):
    id: str


@app.post("/summarize-profile-bucket/")
async def summarize_student(data: StudentRequest):
    student_doc = collection.find_one({"id":data.id})
    if not student_doc:
        raise HTTPException(status_code=404, detail="Student not found.")
    learning_profile = student_doc.get("learning_profile")
    if not learning_profile:
        raise HTTPException(status_code=400, detail="Learning profile not found for this student.")
    

    try:
        response = llmclient.models.generate_content(
            contents=f"here is the data about the learning profile student you need to summarize {learning_profile}",
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                    system_instruction=[
                        """
                        1. You will be provided with th learning profile of a user.
                        2. You must summarize the profile of the user without losing any important information.
                        3. Keep your summary of about 20-30 words.
                        4. Summarization must be effective so that tailored responses can be provided to the user.
                        """
                    ]
            )
        )
        summary = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    # Step 3: Update the document
    collection.update_one({"id": data.id}, {"$set": {"learning_profile_summary": summary}})

    return {
        'learning_profile_summary':summary
    }


@app.post("/summarize-preferences-bucket/")
async def summarize_student(data: StudentRequest):
    student_doc = collection.find_one({"id":data.id})
    if not student_doc:
        raise HTTPException(status_code=404, detail="Student not found.")
    communication_preferences = student_doc.get("communication_preferences")
    if not communication_preferences:
        raise HTTPException(status_code=400, detail="Communication Preferences not found for this student.")
    

    try:
        response = llmclient.models.generate_content(
            contents=f"here is the data about the communication preferences student you need to summarize {communication_preferences}",
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                    system_instruction=[
                        """
                        Follow these given prompts strictly:
                        1. You will be provided with the information about the academic context of a user.
                        2. You must summarize the profile of the user without losing any important information.
                        3. Summarization must be effective so that tailored responses can be provided to the user.
                        4. Keep your summary of about 30-40 words.
                        
                        """
                    ]
            )
        )
        summary = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    # Step 3: Update the document
    collection.update_one({"id": data.id}, {"$set": {"communication_preferences_summary": summary}})

    return {
        'communication_preferences_summary':summary
    }


@app.post("/summarize-academic-bucket/")
async def summarize_student(data: StudentRequest):
    student_doc = collection.find_one({"id": data.id})
    if not student_doc:
        raise HTTPException(status_code=404, detail="Student not found.")

    print("Student Document:", student_doc)

    academic_context = student_doc.get("academic_context")
    if not academic_context:
        raise HTTPException(
            status_code=400,
            detail=f"Academic context not found. Available keys: {list(student_doc.keys())}"
        )

    try:
        response = llmclient.models.generate_content(
            contents=f"Here is the academic context of a student: {academic_context}",
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=[
                    """
                    Follow these given prompts strictly:
                    1. You will be provided with the academic context of a user.
                    2. You must summarize it without losing key information.
                    3. Must focus on what user has covered till now.
                    3. Keep the summary around 30-40 words.
                    """
                ]
            )
        )
        summary = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    collection.update_one({"id": data.id}, {"$set": {"academic_profile_summary": summary}})

    return {
        'academic_context_summary': summary
    }
