"""
fusion_benchmark.py
Fusionne toutes les sources de questions dans questions.jsonl
Lance depuis la racine du projet : python evaluation/fusion_benchmark.py
"""
import json
from pathlib import Path
from collections import Counter

OUTPUT = Path("data/datasets/benchmark/questions.jsonl")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# ── Toutes les questions — sources fusionnées ─────────────────────
QUESTIONS = [

    # ══ EXAMEN CORRIGÉ 2023 ═══════════════════════════════════════
    {
        "question": "Donner le principe de base des procédés de séparation par adsorption modulée en pression (PSA) et par adsorption modulée en température (TSA).",
        "answer": "PSA : adsorption sous pression pour isoler certains gaz (séparation physique), régénération par décompression. TSA : adsorption à basse température, régénération des adsorbants par élévation de température.",
        "type": "FACTUEL", "domaine": "adsorption", "difficulté": "N1", "source": "examen_corrige_2023"
    },
    {
        "question": "Donner la signification des deux termes : Rétentat et Perméat.",
        "answer": "Rétentat : partie contenant les molécules ou particules retenues par la membrane. Perméat : partie contenant les molécules qui traversent la membrane.",
        "type": "FACTUEL", "domaine": "séparation_membranaire", "difficulté": "N1", "source": "examen_corrige_2023"
    },
    {
        "question": "Les techniques de séparation membranaires mettent en oeuvre des systèmes polyphasés constitués par quels éléments ?",
        "answer": "1) Le fluide à traiter, 2) Le fluide traité (solution à dépolluer, eau à dessaler...), 3) La membrane.",
        "type": "FACTUEL", "domaine": "séparation_membranaire", "difficulté": "N1", "source": "examen_corrige_2023"
    },
    {
        "question": "L'isotherme d'adsorption est donnée par : 1/V = 1/(3×10⁵ × P/P°) + 1/120. Écrire l'expression sous forme courante et identifier la théorie correspondante.",
        "answer": "V = 120×(2500/P°)×P / (1+(2500/P°)×P) — forme V=Vm×bP/(1+bP) → isotherme de Langmuir avec Vm=120 ml/g et b=2500/P°.",
        "type": "CALCUL", "domaine": "isothermes_adsorption", "difficulté": "N2", "source": "examen_corrige_2023"
    },
    {
        "question": "Pour l'isotherme de Langmuir précédente avec b=5 mm Hg⁻¹ : déterminer Vm, la surface spécifique (m²/g) et P° (surface molécule adsorbat = 0,2 nm²).",
        "answer": "Vm=120 ml/g. P°=2500/5=500 mm Hg. S=(1×120×10⁻³×6,023×10²³)/(0,082×273)×0,2×10⁻¹⁸=645 m²/g.",
        "type": "CALCUL", "domaine": "isothermes_adsorption", "difficulté": "N3", "source": "examen_corrige_2023"
    },
    {
        "question": "Données Langmuir : P=[10,30,40,80,120] mm Hg, V=[117,64;119,20;119,40;119,70;119,80] ml/g. Vérifier graphiquement la loi de Langmuir et déterminer Vm, b et la surface spécifique.",
        "answer": "Tracer P/V=f(P) → droite y=0,0083x+0,0017 (R²=1) → Langmuir vérifié. Vm=120,48 ml/g, b=4,88 mm Hg⁻¹. S≈648 m²/g.",
        "type": "CALCUL", "domaine": "isothermes_adsorption", "difficulté": "N4", "source": "examen_corrige_2023"
    },

    # ══ POLYCOPIÉ — ADSORPTION ════════════════════════════════════
    {
        "question": "Déterminer l'isotherme pour : C=[0,322;0,117;0,039;0,0061;0,0011] kg/m³, m=[0,150;0,122;0,094;0,059;0,045] kg/kg (phénol/charbon actif).",
        "answer": "Tracer ln(m)=f(ln(c)) → droite → Freundlich. Équation : m=0,199×c^0,229.",
        "type": "CALCUL", "domaine": "isothermes_adsorption", "difficulté": "N2", "source": "polycopie"
    },
    {
        "question": "Adsorption batch : 1 m³ eau usée, C₀=0,21 kg phénol/m³, 1,4 kg charbon actif (m=0,199×c^0,229). Trouver m et c à l'équilibre et le % phénol extrait.",
        "answer": "Intersection isotherme et bilan 1,4m+c=0,21 : m=0,106 kg/kg, c=0,062 kg/m³. Phénol extrait=70,47%.",
        "type": "CALCUL", "domaine": "procédés_discontinus", "difficulté": "N2", "source": "polycopie"
    },
    {
        "question": "Déterminer l'isotherme pour l'adsorption du glucose sur alumine activée : C=[0,004;0,0087;0,019;0,027;0,094;0,195] g/cm³, m=[0,026;0,053;0,075;0,082;0,123;0,129] g/g.",
        "answer": "À compléter avec le corrigé du polycopié.",
        "type": "CALCUL", "domaine": "isothermes_adsorption", "difficulté": "N2", "source": "polycopie"
    },
    {
        "question": "Cinétique d'adsorption charbon actif — substance colorante. D=0,7×10⁻⁹ m²/s, Dpore=0,5×10⁻¹⁰, Dp=0,2×10⁻¹¹, kL=1,2×10⁴, dp=800 µm, ε=0,4, U=0,2×10⁻² m/s. Exprimer la vitesse en concentrations réduites.",
        "answer": "R=0,076. T=43,48%. Résistance externe=175,68 s. Résistance pores=7336,75 s. Résistance surface=5780,35 s. K=1,81×10⁻⁴ s⁻¹. Loi : dY/dt=K(X(1-Y)-RY(1-X)).",
        "type": "CALCUL", "domaine": "dynamique_adsorption", "difficulté": "N4", "source": "polycopie"
    },
    {
        "question": "Épuration eau (0,09 kg phénol/m³, V=4 m³), 2 kg charbon actif, dp=250 µm, 15°C. Isotherme Langmuir : m=C/(5,13×10⁻³C+4,06×10⁻²). kcpap=0,2×10⁻⁴ s⁻¹. Déterminer R, la concentration limite et la durée pour atteindre 0,009 kg/m³.",
        "answer": "R=0,989. Concentration limite C=0,0068 kg/m³. Durée t=13 740 s ≈ 3,82 h.",
        "type": "CALCUL", "domaine": "procédés_discontinus", "difficulté": "N4", "source": "polycopie"
    },


    {
        "question": "Adsorption batch sur charbon actif. Une solution d'eau usée de volume 0,89 m³ et de concentration 0,22 kg de phénol/m³. Une quantité de 1,5 kg de charbon actif frais est ajoutée. En utilisant l'isotherme C=[0,33; 0,12; 0,03; 0,006; 0,001] et m=[0,15; 0,115; 0,09; 0,05; 0,04], trouver m et c à l'équilibre et le % de phénol extrait.",
        "answer": "À compléter avec le corrigé du polycopié.",
        "type": "CALCUL", "domaine": "procédés_discontinus", "difficulté": "N2", "source": "polycopie"
    },

    # ══ POLYCOPIÉ — OSMOSE INVERSE ════════════════════════════════
    {
        "question": "Calculer la pression osmotique d'une solution de 0,10 g mol NaCl / 1000 g H₂O à 25°C.",
        "answer": "NaCl → 2 ions, n=2×10⁻⁴ mol. V=1/997 m³. π=CRT=4,88 atm.",
        "type": "CALCUL", "domaine": "osmose_inverse", "difficulté": "N1", "source": "polycopie"
    },
    {
        "question": "Membrane acétate de cellulose : A=2×10⁻³ m², C1=10,0 kg/m³ NaCl, C2=0,39 kg/m³, débit=1,92×10⁻⁸ m³/s, ΔP=54,42 atm. Calculer Aw, As et la rétention R.",
        "answer": "Nw=9,57×10⁻³ kg/m²·s. Δπ=7,48 atm. Aw=2,039×10⁻⁴ kg/s·m²·atm. As=3,81×10⁻⁷ m/s. R=0,961.",
        "type": "CALCUL", "domaine": "osmose_inverse", "difficulté": "N3", "source": "polycopie"
    },
    {
        "question": "Calculer la pression osmotique à 25°C pour : 1. 0,5 g mol NaCl/kg H₂O. 2. 1,0 g saccharose/kg H₂O (exp=0,0714 atm). 3. 1,0 g MgCl₂/kg H₂O (exp=0,660 atm).",
        "answer": "À compléter avec le corrigé du polycopié.",
        "type": "CALCUL", "domaine": "osmose_inverse", "difficulté": "N2", "source": "polycopie"
    },
    {
        "question": "Membrane osmose inverse NaCl 2,5 g/L, Aw=4,81×10⁻⁴ kg/s·m²·atm, As=4,42×10⁻⁷ m/s, ΔP=27,20 atm. Calculer le flux d'eau, flux soluté, rétention R et C2.",
        "answer": "À compléter avec le corrigé du polycopié.",
        "type": "CALCUL", "domaine": "osmose_inverse", "difficulté": "N3", "source": "polycopie"
    },

    # ══ QCM — PERMÉATION GAZEUSE =════════════════════════════════
    {
        "question": "Quelles sont les caractéristiques générales et les domaines d'application des procédés de séparation par membranes ?",
        "answer": "Applicables aux mélanges liquides, gazeux et hétérogènes. Reposent sur un transfert cinétique de matière — pas sur un équilibre thermodynamique.",
        "type": "FACTUEL", "domaine": "séparation_membranaire", "difficulté": "N1", "source": "qcm_membranes"
    },
    {
        "question": "Quels types de forces motrices peuvent être à la base d'un procédé de séparation par membrane ?",
        "answer": "Différence de pression (osmose inverse, ultrafiltration), différence de pression partielle (perméation gazeuse), différence de température (distillation membranaire), champ électrique (électrodialyse).",
        "type": "FACTUEL", "domaine": "séparation_membranaire", "difficulté": "N1", "source": "qcm_membranes"
    },
    {
        "question": "Quels types de matériaux et structures sont employés pour réaliser une séparation membranaire ?",
        "answer": "Matériaux organiques (polymères) ou inorganiques (céramiques, métaux). Structures poreuses (micro/ultrafiltration) ou denses (osmose inverse, perméation gazeuse).",
        "type": "FACTUEL", "domaine": "matériaux_membranaires", "difficulté": "N1", "source": "qcm_membranes"
    },
    {
        "question": "Sur quel principe repose le procédé de perméation gazeuse et quelles sont les configurations de pression typiques pour l'opérer ?",
        "answer": "Repose sur une différence de perméabilité entre les composés. Force motrice créée par : compression amont, mise sous vide aval, ou combinaison des deux.",
        "type": "FACTUEL", "domaine": "perméation_gazeuse", "difficulté": "N2", "source": "qcm_membranes"
    },
    {
        "question": "De quels paramètres dépendent les performances de séparation d'un module de perméation gazeuse ?",
        "answer": "Sélectivité du matériau membranaire, rapport de pression amont/aval, taux de prélèvement (stage cut). Pas de dépendance à une courbe d'équilibre thermodynamique.",
        "type": "FACTUEL", "domaine": "perméation_gazeuse", "difficulté": "N2", "source": "qcm_membranes"
    },
    {
        "question": "Dans quelles conditions obtient-on la teneur maximale en composé rapide au perméat d'un module de perméation gazeuse ?",
        "answer": "Quand le stage cut ET le rapport de pression tendent tous les deux vers zéro simultanément — maximisant la force motrice locale sans appauvrir le rétentat.",
        "type": "COMPARAISON", "domaine": "perméation_gazeuse", "difficulté": "N3", "source": "qcm_membranes"
    },
    {
        "question": "Pour un matériau et une fraction molaire au perméat fixés, de quoi dépendent les performances d'un module de perméation gazeuse et l'épaisseur a-t-elle un rôle ?",
        "answer": "Dépendent du stage cut et du rapport de pression. L'épaisseur n'influence pas la sélectivité — elle impacte uniquement le flux de perméation global (loi solution-diffusion : J=P×Δp/l).",
        "type": "FACTUEL", "domaine": "perméation_gazeuse", "difficulté": "N2", "source": "qcm_membranes"
    },
    {
        "question": "Quelles sont les principales applications industrielles du procédé de perméation gazeuse ?",
        "answer": "Production d'azote à partir de l'air, purification du gaz naturel et du biogaz (séparation CO₂/CH₄). L'extraction d'hélium se fait à partir du gaz naturel. La séparation d'énantiomères relève de techniques en phase liquide.",
        "type": "FACTUEL", "domaine": "perméation_gazeuse", "difficulté": "N1", "source": "qcm_membranes"
    },
    {
        "question": "Dans quels régimes (permanent ou transitoire) les procédés membranaires peuvent-ils s'appliquer et peuvent-ils être mis en œuvre dans des systèmes multi-étagés ?",
        "answer": "Les procédés membranaires peuvent s'appliquer aussi bien en régime permanent qu'en régime transitoire (ex: filtration frontale). Ils ne nécessitent pas systématiquement de recyclage et peuvent tout à fait être mis en œuvre dans des systèmes multi-étagés.",
        "type": "FACTUEL", "domaine": "séparation_membranaire", "difficulté": "N1", "source": "qcm_membranes"
    },
    {
        "question": "Les procédés de séparation de gaz par membranes peuvent-ils être multi-étagés et peuvent-ils intégrer des systèmes réactifs ?",
        "answer": "Oui, ils peuvent être configurés en systèmes multi-étagés et peuvent intégrer des systèmes réactifs (réacteurs membranaires). Contrairement à d'autres méthodes, ils peuvent s'appuyer uniquement sur des processus physiques via des différences de pression, sans nécessiter de différence de température.",
        "type": "FACTUEL", "domaine": "perméation_gazeuse", "difficulté": "N1", "source": "qcm_membranes"
    },
]

# ── Ajouter les IDs ───────────────────────────────────────────────
for i, q in enumerate(QUESTIONS, 1):
    q["id"] = f"Q{i:03d}"

# ── Sauvegarder ───────────────────────────────────────────────────
with open(OUTPUT, "w", encoding="utf-8") as f:
    for q in QUESTIONS:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

# ── Stats ─────────────────────────────────────────────────────────
print(f"\n{'═'*55}")
print(f"  BENCHMARK LRGP — {len(QUESTIONS)} questions")
print(f"{'═'*55}")

types    = Counter(q["type"] for q in QUESTIONS)
domaines = Counter(q["domaine"] for q in QUESTIONS)
niveaux  = Counter(q["difficulté"] for q in QUESTIONS)
sources  = Counter(q["source"] for q in QUESTIONS)

print(f"\n  Par type :")
for t, n in sorted(types.items()):
    bar = "█" * n
    print(f"    {t:<15} {n:2d}  {bar}")

print(f"\n  Par domaine :")
for d, n in sorted(domaines.items(), key=lambda x: -x[1]):
    print(f"    {d:<30} {n:2d}")

print(f"\n  Par niveau :")
for nv, n in sorted(niveaux.items()):
    bar = "█" * n
    print(f"    {nv}  {n:2d}  {bar}")

print(f"\n  Par source :")
for s, n in sorted(sources.items(), key=lambda x: -x[1]):
    print(f"    {s:<30} {n:2d}")

print(f"\n  Progression : {len(QUESTIONS)}/80 minimum")
print(f"  Domaines manquants : transfert_matière, contacteurs_fibres, CH4_CO2")
print(f"{'═'*55}\n")