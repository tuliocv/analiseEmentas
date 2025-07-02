# -*- coding: utf-8 -*-
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

# Configuração da página
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

                # extrai nome e código
                m = re.search(
                    r"UNIDADE CURRICULAR[:\s]*(.+?)\s*\(\s*(\d+)\s*\)",
                    texto, re.IGNORECASE | re.DOTALL
                )
                nome = m.group(1).strip() if m else fn
                cod  = m.group(2).strip() if m else fn

                # extrai conteúdo programático
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
# explode frases ENADE
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
    st.header("t-SNE das UCs")
    df_group = (
        df_ementas
        .groupby(['COD_EMENTA','NOME UC'])['CONTEUDO_PROGRAMATICO']
        .apply(lambda texts: " ".join(texts))
        .reset_index()
    )
    texts = df_group['CONTEUDO_PROGRAMATICO'].tolist()
    emb   = model.encode(texts, convert_to_tensor=False)
    n     = len(emb)
    perp  = st.slider("Perplexity", 2, max(2, n//3), value=min(30, max(2, n//3)))
    coords = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(emb)
    df_group['X'], df_group['Y'] = coords[:,0], coords[:,1]

    fig, ax = plt.subplots(figsize=(8,6))
    for label, grp in df_group.groupby('NOME UC'):
        ax.scatter(grp['X'], grp['Y'], label=label, s=40, alpha=0.7)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(bbox_to_anchor=(1,1))
    st.pyplot(fig)

# --- 6B) Matriz de Similaridade ---
elif analise == "Matriz de Similaridade ENADE × Ementas":
    st.header("Similaridade ENADE × Ementas")
    # explode ementas em frases
    ementa_expl = (
        df_ementas
        .assign(
            FRASE=lambda df: df['CONTEUDO_PROGRAMATICO']
                .str.replace('\n',' ')
                .str.split(r'[.;]')
        )
        .explode('FRASE')
        .assign(FRASE=lambda df: df['FRASE'].str.strip())
    )
    ementa_expl = ementa_expl[ementa_expl['FRASE'].str.len()>5]

    with st.spinner("Gerando embeddings…"):
        emb_e = model.encode(ementa_expl['FRASE'].tolist(), convert_to_tensor=True)
        emb_n = model.encode(enade_expl['FRASE_ENADE'].tolist(), convert_to_tensor=True)

    sim = util.cos_sim(emb_n, emb_e).cpu().numpy()
    rec = []
    idxs = ementa_expl.groupby('COD_EMENTA').indices
    for cod, sidx in idxs.items():
        for i,row in enade_expl.iterrows():
            rec.append({
                "COD_EMENTA": cod,
                "FRASE_ENADE": row['FRASE_ENADE'],
                "MAX_SIM": float(sim[i, sidx].max())
            })
    df_sim = (
        pd.DataFrame(rec)
        .pivot(index='COD_EMENTA', columns='FRASE_ENADE', values='MAX_SIM')
        .fillna(0)
    )
    st.dataframe(df_sim.style.background_gradient(cmap="RdYlGn"))
    buf = BytesIO()
    df_sim.to_excel(buf, index=True)
    buf.seek(0)
    st.download_button("⬇️ Baixar Similaridade", buf, "sim_enade_ementa.xlsx")

# --- 6C) Matriz de Redundância ---
elif analise == "Matriz de Redundância":
    st.header("Matriz de Redundância")
    df_group = (
        df_ementas
        .groupby('COD_EMENTA')['CONTEUDO_PROGRAMATICO']
        .apply(lambda txts: " ".join(txts))
        .reset_index()
    )
    emb = model.encode(df_group['CONTEUDO_PROGRAMATICO'].tolist(), convert_to_tensor=True)
    sim = util.cos_sim(emb, emb).cpu().numpy()
    df_red = pd.DataFrame(sim, index=df_group['COD_EMENTA'], columns=df_group['COD_EMENTA'])
    st.dataframe(df_red.style.background_gradient(cmap="RdYlGn_r"))
    buf = BytesIO()
    df_red.to_excel(buf, index=True)
    buf.seek(0)
    st.download_button("⬇️ Baixar Redundância", buf, "redundancia_uc.xlsx")

# --- 6D) Análise Ementa Expandida vs ENADE ---
else:
    st.header("🔄 Análise Ementa Expandida vs ENADE")

    # explode contextualizado em frases
    df_ctx = df_ementas.rename(
        columns={"CONTEUDO_PROGRAMATICO": "CONTEUDO_PROGRAMATICO"}
    )
    df_ctx['FRASE'] = (
        df_ctx['CONTEUDO_PROGRAMATICO']
        .str.replace('\n',' ')
        .str.split(r'[.;]')
    )
    df_ctx = (
        df_ctx
        .explode('FRASE')
        .assign(FRASE=lambda d: d['FRASE'].str.strip())
    )
    df_ctx = df_ctx[df_ctx['FRASE'].str.len()>5].reset_index(drop=True)

    limiar = st.slider("Limiar de similaridade", 0.0, 1.0, 0.6, step=0.05)

    with st.spinner("Calculando embeddings..."):
        emb_f = model.encode(df_ctx['FRASE'].tolist(), convert_to_tensor=True)
        emb_n = model.encode(enade_expl['FRASE_ENADE'].tolist(), convert_to_tensor=True)
    simm = util.cos_sim(emb_n, emb_f).cpu().numpy()

    records = []
    for i,row in enade_expl.iterrows():
        sims    = simm[i]
        max_sim = float(sims.max())
        idx_max = int(sims.argmax())
        cod_max = df_ctx.loc[idx_max,'COD_EMENTA']
        text_max= df_ctx.loc[idx_max,'CONTEUDO_PROGRAMATICO']
        above  = df_ctx.loc[sims>=limiar,'COD_EMENTA'].unique().tolist()

        records.append({
            "FRASE_ENADE":     row['FRASE_ENADE'],
            "DIMENSÃO":        row['DIMENSAO'],
            "MAX_SIM":         round(max_sim,3),
            "COD_MAX":         cod_max,
            "TEXTO_MAX":       text_max,
            f"UCs_>={int(limiar*100)}%": ";".join(map(str,above))
        })

    df_res = pd.DataFrame(records)
    st.dataframe(df_res)
    buf = BytesIO()
    df_res.to_excel(buf, index=False)
    buf.seek(0)
    st.download_button("📥 Baixar Análise Expandida vs ENADE", buf,
                       "analise_expandida_enade.xlsx")

