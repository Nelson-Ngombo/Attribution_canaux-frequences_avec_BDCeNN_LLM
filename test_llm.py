# test_llm.py
from dotenv import load_dotenv
import os
from google import genai

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
print(f"🔑 Clé détectée : {api_key[:10]}...{api_key[-4:] if api_key else 'ABSENTE'}")

client = genai.Client(api_key=api_key)

# --- Option A : lister les modèles (décommenter si besoin) ---
# for m in client.models.list():
#     print(m.name)

# --- Option B : tester un modèle précis ---
MODEL = "gemini-3.6-flash"   # ← remplacer si nécessaire
response = client.models.generate_content(
    model=MODEL,
    contents=" es-tu opérationnel ?",
)
print(f"🤖 [{MODEL}] Réponse : {response.text}")