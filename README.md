
# 🍋 AI Drug Discovery Studio
### RDKit + Lemonade Server + Local LLM + Streamlit

Author: Abhishek Nandy

---

# Overview

This project is a **local AI-assisted drug discovery platform** that integrates:

• RDKit for cheminformatics  
• Lemonade Server for running local LLM models  
• Streamlit for interactive dashboards  
• Gemma‑3‑4B as a medicinal chemistry reasoning model  

The goal is to provide a **fully offline research environment** where scientists can:

- analyze molecules
- compute drug‑likeness descriptors
- visualize molecules in 2D and 3D
- evaluate structural similarity
- receive AI‑assisted medicinal chemistry insights

All computation runs locally without external APIs.

---

# System Architecture

User Input (SMILES)
        │
        ▼
Streamlit Interface
        │
        ▼
RDKit Processing Layer
    • SMILES parsing
    • descriptor calculation
    • fingerprints
    • similarity matrix
    • conformer generation
        │
        ▼
Lemonade Server (Local LLM)
        │
        ▼
Medicinal Chemistry AI Insights

---

# Key Components

## RDKit

RDKit performs the chemical analysis including:

• Molecular descriptor computation  
• Lipinski Rule‑of‑Five checks  
• Morgan fingerprints (ECFP)  
• Tanimoto similarity calculation  
• 2D molecule rendering  
• 3D conformer generation (ETKDG)

Important descriptors calculated:

| Descriptor | Meaning |
|-------------|--------|
| MolWt | Molecular weight |
| LogP | Lipophilicity |
| TPSA | Polar surface area |
| HBD | Hydrogen bond donors |
| HBA | Hydrogen bond acceptors |
| RotBonds | Molecular flexibility |
| Rings | Ring count |
| HeavyAtoms | Molecular complexity |

---

# Lemonade Server

Lemonade Server allows running **local LLM models** with an OpenAI‑compatible API.

This project uses:

Gemma‑3‑4B‑it‑GGUF

Benefits:

• Fully offline  
• GPU acceleration via Vulkan  
• OpenAI compatible API  
• GGUF model support

---

# Lemonade Installation

Download:

https://lemonade-server.ai

Verify installation:

```
lemonade-server -h
```

---

# List Available Models

```
lemonade-server list
```

Example output:

Gemma-3-4b-it-GGUF     Yes
DeepSeek-Qwen3-8B      No
Phi-4-mini             No

---

# Run Lemonade Server

```
lemonade-server serve
```

Server endpoint:

http://localhost:8000/api/v1

Health check:

http://localhost:8000/api/v1/health

---

# Python LLM Integration

```
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/api/v1",
    api_key="lemonade"
)

response = client.chat.completions.create(
    model="Gemma-3-4b-it-GGUF",
    messages=[{"role":"user","content":"Analyze aspirin"}]
)
```

---

# Running the Application

Start Lemonade server:

```
lemonade-server serve
```

Activate environment:

```
conda activate lemonade_rdkit
```

Run Streamlit:

```
streamlit run app.py
```

Open:

http://localhost:8501

---

# Example Workflow

1. Input SMILES molecules
2. RDKit computes descriptors
3. Molecules visualized in 2D and 3D
4. Similarity matrix generated
5. LLM produces medicinal chemistry analysis

---

# Current Features

SMILES parsing ✓  
Descriptor calculation ✓  
2D visualization ✓  
3D conformer generation ✓  
Similarity matrix ✓  
Local LLM analysis ✓  
Offline execution ✓  

---

# Future Roadmap

## ADMET Prediction

Add machine learning models for:

• toxicity prediction  
• BBB penetration  
• solubility  
• metabolic stability

Possible libraries:

DeepChem  
Chemprop  
PyTorch

---

# Molecular Generation

Integrate generative models:

• MolGPT  
• Diffusion models  
• Graph neural networks

Use cases:

Lead optimization  
Novel molecule discovery  
Scaffold hopping

---

# Docking Integration

Add docking engines:

AutoDock Vina  
DiffDock  
GNINA

Pipeline:

SMILES → 3D conformer → Docking → Scoring

---

# Literature‑Aware AI

Integrate RAG with:

PubChem  
ChEMBL  
PubMed  

Allow LLM to answer research questions with scientific data.

---

# Active Learning Drug Discovery

Future system:

AI proposes molecules
↓
Predict properties
↓
Evaluate drug‑likeness
↓
Generate improved molecules

Closed loop discovery.

---

# Long Term Vision

Create a **complete AI drug discovery stack** combining:

RDKit  
Graph Neural Networks  
Diffusion Models  
Docking Engines  
Local LLM reasoning

Similar in spirit to:

AlphaFold + DeepChem style platforms.

---

# Safety Disclaimer

LLM outputs are **for research assistance only**.

They do not replace:

• docking validation  
• molecular dynamics  
• wet‑lab experiments  
• clinical trials

---

# License

MIT License

---

# Author

Abhishek Nandy  
Chief Data Scientist Rezolve| Lead Engineer Founding team Adstitch
