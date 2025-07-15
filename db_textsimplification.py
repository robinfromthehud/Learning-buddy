from pymongo import MongoClient

MONGO_URI = "mongodb+srv://hinalh:tGkiflIMrreFbxbO@cluster0.qoxulpa.mongodb.net/"
client = MongoClient(MONGO_URI)

db = client["learn_the_learner"]
users_collection = db["users"]
#mongodb+srv://hinalh:tGkiflIMrreFbxbO@cluster0.qoxulpa.mongodb.net/