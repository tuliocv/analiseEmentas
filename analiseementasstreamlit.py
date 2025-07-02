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
from openpyxl import load_workbook
from openpyxl.formatting.rule import ColorScaleRule

# Configuração da página
st.set_page_config(layout="wide")
st.title("📂📑 Análise de Ementas via pasta .zip")

# --- 1) Upload do ZIP de PDFs de ementas ---
uploaded_zip = st.file_uploader(
    "Faça upload dos arquivos, em PDF das ementas, em uma pasta .zip",
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
    "Clusterização Ementas",
    "Matriz de Similaridade",
    "Matriz de Redundância",
    "Análise Ementa vs ENADE"
])

# --- 5) Carrega modelo em cache ---
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model = load_model()

# --- 6A) t-SNE das UCs ---
if analise == "Clusterização Ementas":
    st.header("Clusterização Ementas")
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
    ax.scatter(df_group['X'], df_group['Y'], s=40, alpha=0.7)
    # Anota cada ponto com o COD_EMENTA
    for _, row in df_group.iterrows():
        ax.text(
            row['X'] + 0.5,  # pequeno deslocamento para não sobrepor
            row['Y'] + 0.5,
            str(row['COD_EMENTA']),
            fontsize=8
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    st.pyplot(fig)

 # --- Botão de download da imagem ---
    buf_img = BytesIO()
    fig.savefig(buf_img, format="png", dpi=300, bbox_inches="tight")
    buf_img.seek(0)
    st.download_button(
        label="📥 Baixar Gráfico",
        data=buf_img,
        file_name="tsne_ucs.png",
        mime="image/png"
    )


# --- 6B) Matriz de Similaridade ---
elif analise == "Matriz de Similaridade":
    st.header("Matriz de Similaridade")
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

    # Exibe no Streamlit
    st.dataframe(df_sim.style.background_gradient(cmap="RdYlGn"))

    # 1) Grava DataFrame num buffer
    buf = BytesIO()
    df_sim.to_excel(buf, index=True, sheet_name="Similaridade")
    buf.seek(0)

    # 2) Carrega workbook do buffer
    wb = load_workbook(buf)
    ws = wb["Similaridade"]

    # 3) Define intervalo de células (coluna B até a última)
    min_col = 2
    max_col = ws.max_column
    max_row = ws.max_row
    col_letter = lambda idx: ws.cell(row=1, column=idx).column_letter
    range_str = f"{col_letter(min_col)}2:{col_letter(max_col)}{max_row}"

    # 4) Cria regra: vermelho em 0 → amarelo em 50% → verde em 1
    rule = ColorScaleRule(
        start_type='min', start_color='FF0000',
        mid_type='percentile', mid_value=50, mid_color='FFFF00',
        end_type='max', end_color='00FF00'
    )
    ws.conditional_formatting.add(range_str, rule)

    # 5) Salva de volta no buffer
    buf2 = BytesIO()
    wb.save(buf2)
    buf2.seek(0)

    # 6) Botão de download
    st.download_button(
        "⬇️ Baixar Matriz de Similaridade",
        data=buf2,
        file_name="sim_enade_ementa_colorido.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
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
    st.download_button("⬇️ Baixar Matriz de Redundância", buf, "redundancia_uc.xlsx")

# --- 6D) Análise Ementa Expandida vs ENADE ---
else:
    st.header("Análise Ementa vs ENADE")

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

