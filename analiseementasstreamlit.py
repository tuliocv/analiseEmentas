# -*- coding: utf-8 -*-
"""analiseEmentasStreamlit.ipynb

"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import zipfile
import tempfile
import pdfplumber
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")
st.title("📂📑 Análise de Ementas de Psicologia via ZIP")

# --- 1) Upload do ZIP de PDFs de ementas ---
uploaded_zip = st.file_uploader(
    "Faça upload do arquivo ZIP com todos os PDFs de ementas",
    type=["zip"]
)
if not uploaded_zip:
    st.info("Aguardando upload do ZIP...")
    st.stop()

# --- 2) Extrai e processa os PDFs ---
with tempfile.TemporaryDirectory() as tmpdir:
    z = zipfile.ZipFile(uploaded_zip)
    z.extractall(tmpdir)

    registros = []
    for root, _, files in os.walk(tmpdir):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                path = os.path.join(root, fn)
                texto = ""
                with pdfplumber.open(path) as pdf:
                    for p in pdf.pages:
                        texto += (p.extract_text() or "") + "\n"

                m = re.search(
                    r"UNIDADE CURRICULAR[:\s]*(.+?)\s*\(\s*(\d+)\s*\)",
                    texto, re.IGNORECASE | re.DOTALL
                )
                nome = m.group(1).strip() if m else fn
                cod  = m.group(2).strip() if m else fn

                m2 = re.search(
                    r"Conte[úu]do program[aá]tico\s*[:\-–]?\s*(.*?)\s*(?:\n\s*Bibliografia|\Z)",
                    texto, re.IGNORECASE | re.DOTALL
                )
                conteudo = m2.group(1).strip() if m2 else ""
                registros.append({
                    "COD_EMENTA": cod,
                    "NOME UC": nome,
                    "CONTEUDO_PROGRAMATICO": conteudo
                })

    df_ementas = pd.DataFrame(registros)

st.success(f"{len(df_ementas)} ementas carregadas.")
st.subheader("Preview das primeiras ementas")
st.dataframe(df_ementas.head(5))

# --- 3) Upload do Excel ENADE ---
uploaded_enade = st.file_uploader(
    "Faça upload do Excel de competências ENADE",
    type=["xlsx"], key="enade"
)
if not uploaded_enade:
    st.info("Envie o arquivo ENADE para prosseguir.")
    st.stop()

enade = pd.read_excel(uploaded_enade).dropna(subset=['DESCRIÇÃO'])

# Explode frases ENADE
enade['FRASE_ENADE'] = (
    enade['DESCRIÇÃO']
    .str.replace('\n',' ')
    .str.split(r'[.;]')
)
enade_expl = (
    enade.explode('FRASE_ENADE')
         .assign(FRASE_ENADE=lambda df: df['FRASE_ENADE'].str.strip())
)
enade_expl = enade_expl[enade_expl['FRASE_ENADE'].str.len() > 5].reset_index(drop=True)

# --- 4) Escolha da análise ---
analise = st.sidebar.selectbox("Escolha a Análise", [
    "t-SNE das UCs",
    "Matriz de Similaridade ENADE × Ementas",
    "Matriz de Redundância",
    "Análise Ementa Expandida vs ENADE"
])

# --- 5) Carrega modelo em cache ---
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

# --- 6A) t-SNE das UCs ---
if analise == "t-SNE das UCs":
    # ... (mesmo código anterior para t-SNE) ...
    pass

# --- 6B) Matriz de Similaridade ---
elif analise == "Matriz de Similaridade ENADE × Ementas":
    # ... (mesmo código anterior para similaridade) ...
    pass

# --- 6C) Matriz de Redundância ---
elif analise == "Matriz de Redundância":
    # ... (mesmo código anterior para redundância) ...
    pass

# --- 6D) Nova: Análise Ementa Expandida vs ENADE ---
else:
    st.header("🔄 Análise Ementa Expandida vs ENADE")

    # 6D.1) Explode Ementa Contextualizada em frases
    df_ementa_ctx = df_ementas.rename(
        columns={"CONTEUDO_PROGRAMATICO": "CONTEUDO_PROGRAMATICO_CONTEXTUALIZADO"}
    )
    df_ementa_ctx['FRASE'] = (
        df_ementa_ctx['CONTEUDO_PROGRAMATICO_CONTEXTUALIZADO']
        .str.replace('\n', ' ')
        .str.split(r'[.;]')
    )
    df_ementa_ctx = (
        df_ementa_ctx
        .explode('FRASE')
        .assign(FRASE=lambda d: d['FRASE'].str.strip())
    )
    df_ementa_ctx = df_ementa_ctx[df_ementa_ctx['FRASE'].str.len() > 5].reset_index(drop=True)

    # 6D.2) Codifica e gera embeddings
    with st.spinner("🔮 Gerando embeddings para frases..."):
        emb_frases = model.encode(df_ementa_ctx['FRASE'].tolist(), convert_to_tensor=True)
        emb_enade  = model.encode(enade_expl['FRASE_ENADE'].tolist(),   convert_to_tensor=True)

    # 6D.3) Monta limiar
    limiar = st.sidebar.slider("Limiar de similaridade", 0.0, 1.0, 0.6, step=0.05)

    # 6D.4) Calcula matriz de similaridade
    sim_matrix = util.cos_sim(emb_enade, emb_frases).cpu().numpy()

    # 6D.5) Constrói registros
    records = []
    for idx_enade, row_enade in enade_expl.iterrows():
        sims    = sim_matrix[idx_enade]
        max_sim = float(sims.max())
        idx_max = int(sims.argmax())
        cod_max = df_ementa_ctx.loc[idx_max, 'COD_EMENTA']
        texto_max = df_ementa_ctx.loc[idx_max, 'CONTEUDO_PROGRAMATICO_CONTEXTUALIZADO']
        cods_acima = df_ementa_ctx.loc[sims >= limiar, 'COD_EMENTA'].unique().tolist()

        records.append({
            'FRASE_ENADE':        row_enade['FRASE_ENADE'],
            'DIMENSÃO':           row_enade['DIMENSÃO'],
            'MAX_SIMILARIDADE':   round(max_sim, 3),
            'COD_EMENTA_MAX':     cod_max,
            'CONTEUDO_ORIGEM':    texto_max,
            f'UCs_>={int(limiar*100)}%': "; ".join(map(str, cods_acima))
        })

    df_result = pd.DataFrame(records)

    # 6D.6) Exibe e oferece download
    st.dataframe(df_result)
    towrite = BytesIO()
    df_result.to_excel(towrite, index=False, sheet_name="analise")
    towrite.seek(0)
    st.download_button(
        "📥 Baixar Análise Expandida vs ENADE",
        data=towrite,
        file_name="analise_ementa_expandida_vs_enade.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
