"""
relance_q052.py
Relance uniquement la question Q052 qui avait crashé.
"""
import json
import requests
import time
from pathlib import Path

API_URL = "http://localhost:8000/v1/chat/completions"
BENCHMARK_FILE = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\data\\datasets\\benchmark\\split\\benchmark_test.jsonl")
RESPONSES_FILE = Path("C:\\Users\\Samir\\Documents\\LRGP-PROJECT\\lrgp_llm\\evaluation\\results\\srar_gp_responses.jsonl")

# 1. Trouver Q052 dans le benchmark
target = None
with open(BENCHMARK_FILE, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            q = json.loads(line)
            if q["id"] == "Q052":
                target = q
                break

if target is None:
    print("✗ Q052 introuvable dans le benchmark")
    exit(1)

print(f"Q052 trouvée :")
print(f"  Type : {target['type']}")
print(f"  Question : {target['question'][:150]}")
print(f"  Référence : {target.get('answer', '')[:200]}")

# 2. Appeler l'API
print(f"\nAppel SRAR-GP...")
payload = {
    "model": "srar-gp-verbose",
    "messages": [{"role": "user", "content": target["question"]}],
    "stream": False,
    "temperature": 0.1,
}

t0 = time.time()
try:
    response = requests.post(API_URL, json=payload, timeout=360)
    latency = time.time() - t0
    
    print(f"\n✓ Status HTTP : {response.status_code}")
    print(f"✓ Latence : {latency:.1f}s")
    
    if response.status_code == 200:
        data = response.json()
        contenu = data["choices"][0]["message"]["content"]
        print(f"\n✓ Réponse ({len(contenu)} chars) :")
        print("─" * 60)
        print(contenu[:1500])
        print("─" * 60)
        
        # Demande confirmation avant d'écrire
        choix = input("\n>>> Écrire ce résultat dans srar_gp_responses.jsonl ? (o/n) : ")
        
        if choix.lower() in ("o", "oui", "y", "yes"):
            # Lire les réponses existantes
            existing = []
            with open(RESPONSES_FILE, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        existing.append(json.loads(line))
            
            # Remplacer Q052
            import re
            voie = "UNKNOWN"
            parcours = []
            
            voie_match = re.search(r"\*\*Voie\*\*\s*:\s*(\w+)", contenu)
            if voie_match:
                voie = voie_match.group(1)
            
            parcours_match = re.search(r"\*\*Agents traversés\*\*\s*:\s*([^\n]+)", contenu)
            if parcours_match:
                parcours = [a.strip() for a in parcours_match.group(1).split("→")]
            
            parcours_str = str(parcours).lower()
            
            new_entry = {
                "id":              "Q052",
                "type":            target["type"],
                "question":        target["question"],
                "reference":       target.get("answer", ""),
                "reponse_modele":  contenu,
                "latence_sec":     round(latency, 2),
                "voie":            voie,
                "parcours":        parcours,
                "renegociation":   "correction_code" in parcours_str or "correction_physique" in parcours_str,
                "web_search_used": "web_search" in parcours_str,
                "missing_data":    "missing_data_handler" in parcours_str,
                "calcul_failed":   "calculation_expert_failed" in parcours_str,
                "domaine":         target.get("domaine", ""),
                "difficulté":      target.get("difficulté", ""),
            }
            
            # Réécrire le fichier complet avec Q052 mise à jour
            with open(RESPONSES_FILE, "w", encoding="utf-8") as f:
                for r in existing:
                    if r["id"] == "Q052":
                        f.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
                    else:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
            print(f"\n✓ Q052 mise à jour dans {RESPONSES_FILE}")
        else:
            print("\n✗ Pas d'écriture — résultat conservé en console uniquement")
    else:
        print(f"✗ Erreur HTTP {response.status_code} : {response.text[:500]}")

except requests.Timeout:
    print(f"✗ TIMEOUT après {time.time() - t0:.1f}s")
except Exception as e:
    print(f"✗ Erreur : {e}")