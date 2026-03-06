import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import py3Dmol

from rdkit.Chem import Draw

from utils import (
    molecules_dataframe,
    valid_mols_and_fps,
    pairwise_tanimoto,
    summary_payload_from_df,
    ask_local_llm,
    molblock_3d,
)

st.set_page_config(page_title="Lemonade + RDKit Drug Discovery", layout="wide")
st.title("Lemonade + RDKit Drug Discovery Studio")
st.caption("Local LLM via Lemonade Server + molecular analysis via RDKit")

# ----------------------------
# Session state init
# ----------------------------
if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False

if "df" not in st.session_state:
    st.session_state.df = None

if "mols" not in st.session_state:
    st.session_state.mols = None

if "llm_answer" not in st.session_state:
    st.session_state.llm_answer = ""

with st.sidebar:
    st.header("Settings")
    lemonade_base_url = st.text_input("Lemonade base URL", "http://localhost:8000/api/v1")
    llm_model = st.text_input("Model name", "Gemma-3-4b-it-GGUF")

default_smiles = """CC(=O)OC1=CC=CC=C1C(=O)O
CC(C)CC1=CC=C(C=C1)C(C)C(=O)O
Cn1cnc2n(C)c(=O)n(C)c(=O)c12
CC1=C(C(=O)NC(=O)N1)N
CCOC(=O)C1=CC=CC=C1N"""

st.subheader("Input SMILES")
smiles_text = st.text_area("Paste one SMILES per line", value=default_smiles, height=180)
smiles_list = [x.strip() for x in smiles_text.splitlines() if x.strip()]

# ----------------------------
# Main analysis trigger
# ----------------------------
if st.button("Run Drug Discovery Analysis", type="primary", key="run_analysis"):
    df, mols = molecules_dataframe(smiles_list)
    st.session_state.df = df
    st.session_state.mols = mols
    st.session_state.analysis_ready = True
    st.session_state.llm_answer = ""

# ----------------------------
# Show analysis if ready
# ----------------------------
if st.session_state.analysis_ready and st.session_state.df is not None:
    df = st.session_state.df
    mols = st.session_state.mols

    st.subheader("Descriptor Table")
    st.dataframe(df, use_container_width=True)

    valid_rows = df[df["Valid"] == True].reset_index(drop=True)
    valid_mols, fps = valid_mols_and_fps(df, mols)

    if len(valid_mols) == 0:
        st.error("No valid molecules found.")
        st.stop()

    st.subheader("2D Molecule Gallery")
    legends = [
        f"M{i+1} | MW={valid_rows.iloc[i]['MolWt']} | LogP={valid_rows.iloc[i]['LogP']} | TPSA={valid_rows.iloc[i]['TPSA']}"
        for i in range(len(valid_mols))
    ]
    grid_img = Draw.MolsToGridImage(
        valid_mols,
        molsPerRow=3,
        subImgSize=(320, 220),
        legends=legends,
        useSVG=False,
    )
    st.image(grid_img, use_container_width=True)

    st.subheader("Descriptor Visualization")
    numeric_cols = ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "Rings", "Ro5Violations"]

    col1, col2 = st.columns(2)

    with col1:
        feature = st.selectbox("Bar chart feature", numeric_cols, index=0, key="bar_feature")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(valid_rows)), valid_rows[feature].values)
        ax.set_xticks(range(len(valid_rows)))
        ax.set_xticklabels([f"M{i+1}" for i in range(len(valid_rows))], rotation=45)
        ax.set_title(f"{feature} across molecules")
        ax.set_ylabel(feature)
        st.pyplot(fig)

    with col2:
        x_axis = st.selectbox("Scatter X", numeric_cols, index=0, key="scatter_x")
        y_axis = st.selectbox("Scatter Y", numeric_cols, index=1, key="scatter_y")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.scatter(valid_rows[x_axis], valid_rows[y_axis])
        for i in range(len(valid_rows)):
            ax2.annotate(f"M{i+1}", (valid_rows[x_axis].iloc[i], valid_rows[y_axis].iloc[i]))
        ax2.set_xlabel(x_axis)
        ax2.set_ylabel(y_axis)
        ax2.set_title(f"{x_axis} vs {y_axis}")
        st.pyplot(fig2)

    st.subheader("Pairwise Similarity Matrix")
    sim = pairwise_tanimoto(fps)

    fig3, ax3 = plt.subplots(figsize=(6, 5))
    im = ax3.imshow(sim)
    ax3.set_xticks(range(len(valid_rows)))
    ax3.set_yticks(range(len(valid_rows)))
    ax3.set_xticklabels([f"M{i+1}" for i in range(len(valid_rows))])
    ax3.set_yticklabels([f"M{i+1}" for i in range(len(valid_rows))])
    ax3.set_title("Tanimoto Similarity")
    fig3.colorbar(im, ax=ax3)
    st.pyplot(fig3)

    sim_df = pd.DataFrame(
        sim,
        index=[f"M{i+1}" for i in range(len(valid_rows))],
        columns=[f"M{i+1}" for i in range(len(valid_rows))]
    )
    st.dataframe(sim_df, use_container_width=True)

    st.subheader("Heuristic Prioritization")
    ranked = valid_rows.sort_values(
        by=["Ro5Violations", "LogP", "MolWt"],
        ascending=[True, True, True]
    ).reset_index(drop=True)
    ranked.insert(0, "Rank", range(1, len(ranked) + 1))
    st.dataframe(ranked, use_container_width=True)

    st.subheader("3D Molecule Viewer")
    selected_idx = st.selectbox(
        "Select molecule",
        options=list(range(len(valid_rows))),
        format_func=lambda x: f"M{x+1}: {valid_rows.iloc[x]['SMILES'][:60]}",
        key="mol_select"
    )

    selected_smiles = valid_rows.iloc[selected_idx]["SMILES"]
    mol_block = molblock_3d(selected_smiles)

    if mol_block is None:
        st.warning("3D conformer generation failed.")
    else:
        viewer = py3Dmol.view(width=700, height=450)
        viewer.addModel(mol_block, "mol")
        viewer.setStyle({"stick": {}})
        viewer.zoomTo()
        st.components.v1.html(viewer._make_html(), height=450, scrolling=False)

    st.subheader("LLM Medicinal Chemistry Interpretation")
    payload = summary_payload_from_df(df)

    with st.expander("Payload sent to Lemonade"):
        st.json(payload)

    if st.button("Ask Lemonade LLM", key="ask_llm"):
        try:
            with st.spinner("Querying Lemonade..."):
                st.session_state.llm_answer = ask_local_llm(
                    summary_payload=payload,
                    model=llm_model,
                    base_url=lemonade_base_url,
                )
        except Exception as e:
            st.exception(e)

    if st.session_state.llm_answer:
        st.markdown(st.session_state.llm_answer)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results CSV",
        data=csv_bytes,
        file_name="drug_discovery_results.csv",
        mime="text/csv"
    )