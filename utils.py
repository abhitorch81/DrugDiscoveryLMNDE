import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, Lipinski, Crippen, rdMolDescriptors
from openai import OpenAI


def get_llm_client(base_url: str = "http://localhost:8000/api/v1") -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key="lemonade",
    )


def ask_local_llm(summary_payload: dict,
                  model: str = "Gemma-3-4b-it-GGUF",
                  base_url: str = "http://localhost:8000/api/v1") -> str:
    client = get_llm_client(base_url)

    system_prompt = """
You are a medicinal chemistry research assistant.
Use ONLY the provided computed chemistry values.
Do not invent assay data, docking results, toxicity outcomes, or clinical claims.
Comment on:
1. molecular complexity
2. lipophilicity trend
3. oral drug-likeness heuristics
4. possible medicinal chemistry liabilities
5. prioritization suggestions

Be practical, cautious, and concise.
"""

    user_prompt = f"Analyze this candidate set:\n{json.dumps(summary_payload, indent=2)}"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        stream=False,
    )

    text = resp.choices[0].message.content
    if not text:
        raise RuntimeError("Empty response from Lemonade LLM")

    return text


def mol_from_smiles(smiles: str):
    return Chem.MolFromSmiles(smiles)


def build_3d_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if status != 0:
        return None
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    return mol


def molblock_3d(smiles: str) -> Optional[str]:
    mol = build_3d_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToMolBlock(mol)


def compute_basic_descriptors(mol) -> dict:
    return {
        "MolWt": round(Descriptors.MolWt(mol), 3),
        "ExactMolWt": round(Descriptors.ExactMolWt(mol), 3),
        "LogP": round(Crippen.MolLogP(mol), 3),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "HBD": int(Lipinski.NumHDonors(mol)),
        "HBA": int(Lipinski.NumHAcceptors(mol)),
        "RotBonds": int(Lipinski.NumRotatableBonds(mol)),
        "Rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "HeavyAtoms": int(mol.GetNumHeavyAtoms()),
    }


def lipinski_rule_of_five(desc: dict) -> int:
    violations = 0
    if desc["MolWt"] > 500:
        violations += 1
    if desc["LogP"] > 5:
        violations += 1
    if desc["HBD"] > 5:
        violations += 1
    if desc["HBA"] > 10:
        violations += 1
    return violations


def smiles_to_fingerprint(mol, radius=2, nbits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def pairwise_tanimoto(fps: List):
    n = len(fps)
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            sim[i, j] = DataStructs.TanimotoSimilarity(fps[i], fps[j])
    return sim


def molecules_dataframe(smiles_list: List[str]) -> Tuple[pd.DataFrame, List]:
    records = []
    mols = []

    for smi in smiles_list:
        smi = smi.strip()
        if not smi:
            continue

        mol = mol_from_smiles(smi)
        if mol is None:
            records.append({
                "SMILES": smi,
                "Valid": False,
                "Error": "Invalid SMILES"
            })
            mols.append(None)
            continue

        desc = compute_basic_descriptors(mol)
        ro5 = lipinski_rule_of_five(desc)

        records.append({
            "SMILES": smi,
            "Valid": True,
            **desc,
            "Ro5Violations": ro5
        })
        mols.append(mol)

    return pd.DataFrame(records), mols


def valid_mols_and_fps(df: pd.DataFrame, mols: List):
    valid_mols = []
    fps = []
    for mol in mols:
        if mol is not None:
            valid_mols.append(mol)
            fps.append(smiles_to_fingerprint(mol))
    return valid_mols, fps


def summary_payload_from_df(df: pd.DataFrame) -> dict:
    valid_df = df[df["Valid"] == True].copy()
    if valid_df.empty:
        return {"message": "No valid molecules"}

    numeric_cols = [
        "MolWt", "ExactMolWt", "LogP", "TPSA",
        "HBD", "HBA", "RotBonds", "Rings",
        "HeavyAtoms", "Ro5Violations"
    ]

    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "min": float(valid_df[col].min()),
            "max": float(valid_df[col].max()),
            "mean": float(valid_df[col].mean())
        }

    ranked = valid_df.sort_values(
        by=["Ro5Violations", "LogP", "MolWt"],
        ascending=[True, True, True]
    )

    return {
        "num_valid_molecules": int(len(valid_df)),
        "aggregate_stats": stats,
        "top_priority_candidates": ranked.head(5).to_dict(orient="records"),
        "all_candidates": valid_df.to_dict(orient="records"),
    }