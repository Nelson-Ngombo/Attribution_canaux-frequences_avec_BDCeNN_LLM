# test_llm.py
from dotenv import load_dotenv
import os
from google import genai

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
print(f"🔑 Clé détectée : {api_key[:10]}...{api_key[-4:] if api_key else 'ABSENTE'}")

client = genai.Client(api_key=api_key)

# Liste des modèles réellement testables dans l'ordre de préférence
candidates = [
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
]

working_model = None
for model in candidates:
    try:
        response = client.models.generate_content(
            model=model,
            contents="Réponds en une phrase : es-tu opérationnel ?",
        )
        print(f"✅ [{model}] Réponse : {response.text.strip()}")
        working_model = model
        break
    except Exception as e:
        print(f"❌ [{model}] Échec : {type(e).__name__} — {e}")

if working_model:
    print(f"\n🎯 Modèle à utiliser dans llm_assistant.py : {working_model}")
else:
    print("\n❌ Aucun modèle n'a fonctionné. Vérifiez votre clé API.")