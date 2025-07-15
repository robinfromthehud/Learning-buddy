from db_textsimplification import users_collection

users = [
    {
        "user_id": "ayesha_k",
        "learning_style": "visual",
        "language_proficiency": "moderate",
        "learning_speed": "slow",
        "academic_level": "below_average",
        "tech_familiarity": ["Google Docs", "YouTube", "Canva", "Android Phone"],
        "interests": ["anime", "biology", "cartoons"],
        "preferred_output": ["visual", "simple_language", "step_by_step"],
        "confidence_level": "low",
        "last_module_completed": "Intro to Neural Networks – Basics with Diagrams"
    },
    {
        "user_id": "rohan_s",
        "learning_style": "logical",
        "language_proficiency": "fluent",
        "learning_speed": "fast",
        "academic_level": "above_average",
        "tech_familiarity": ["Python", "NumPy", "Pandas", "Jupyter", "VS Code", "Git"],
        "interests": ["machine_learning", "math", "robotics"],
        "preferred_output": ["concise", "formula_based", "real_world_data"],
        "confidence_level": "high",
        "last_module_completed": "Backpropagation and Loss Functions in Deep Learning"
    }
]

for user in users:
    users_collection.update_one({"user_id": user["user_id"]}, {"$set": user}, upsert=True)

print("Users added to MongoDB")
