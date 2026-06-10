"""
test_gemini.py
Test rapide de génération avec Gemini.
Usage : python test_gemini.py
"""
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODELES = [
    "gemini-3-flash-preview"
]

PROMPT = """Génère 1 paire question/réponse sur la perméabilité des membranes PDMS. la question doit etre de type calcule.
Réponds UNIQUEMENT avec un JSON valide :
{"question": "...", "answer": "..."}"""

for modele in MODELES:
    print(f"\nTest {modele}...", end=" ", flush=True)
    try:
        response = client.models.generate_content(
            model=modele,
            contents=PROMPT,
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=500,
            ),
        )
        print(f"✓")
        print(f"  {response.text[:1000]}...")
        break  # s'arrête au premier modèle qui répond
    except Exception as e:
        print(f"✗ {str(e)[:80]}")