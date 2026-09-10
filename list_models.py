# list_models.py
from dotenv import load_dotenv
import os
from google import genai

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

print("📋 Modèles disponibles pour votre clé API :\n")
for m in client.models.list():
    # m.name ressemble à "models/gemini-3.6-flash"
    # on affiche uniquement les modèles qui supportent generateContent
    supported = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None)
    print(f"  - {m.name}   (actions: {supported})")