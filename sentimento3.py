import streamlit as st
import pandas as pd
import re
from transformers import pipeline

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentimento 2025 - Multi-Model",
    page_icon="🤖",
    layout="wide"
)

# ──────────────────────────────────────────────────────────────────────────────
# CATÁLOGO DE MODELOS
#
# Critérios de seleção para uso didático no Streamlit Cloud:
#   ✅ Gratuitos (Hugging Face Hub, sem API key)
#   ✅ Leves (DistilBERT ~270 MB ou BERT-base ~700 MB)
#   ✅ Suportam português sem fine-tuning adicional
#   ✅ Labels claras e mapeáveis
#   ✅ Carregam via pipeline() padrão sem configuração extra
# ──────────────────────────────────────────────────────────────────────────────
MODELOS = {

    # Modelo 1 — BERT multilíngue, escala 1–5 estrelas
    # Treinado em avaliações de produtos em 6 idiomas; generaliza bem para PT-BR.
    # Tamanho: ~700 MB | Labels: "1 star" … "5 stars"
    "⭐ BERT Multilingual — 1 a 5 Estrelas": {
        "id": "nlptown/bert-base-multilingual-uncased-sentiment",
        "tipo": "estrelas",
        "descricao": "Classifica em 1–5 estrelas. Treinado em avaliações de produtos "
                     "em inglês, alemão, holandês, francês, espanhol e italiano. "
                     "Boa generalização para português.",
    },

    # Modelo 2 — DistilBERT multilíngue, Positivo / Neutro / Negativo
    # Versão destilada (knowledge distillation) de um modelo mDeBERTa maior.
    # Tamanho: ~270 MB | Labels: "positive" / "neutral" / "negative"
    # Vantagem didática: demonstra que modelos menores podem ser tão bons quanto os maiores.
    "💬 DistilBERT Multilingual — Pos / Neu / Neg": {
        "id": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        "tipo": "polaridade",
        "descricao": "Modelo destilado (menor e mais rápido). Classifica em "
                     "Positivo, Neutro e Negativo. Suporta PT, EN, ES e outros.",
    },

    # Modelo 3 — DistilBERT multilíngue, escala 5 classes (Very Negative → Very Positive)
    # Fine-tuned com dados sintéticos gerados por LLMs, cobrindo português explicitamente.
    # Tamanho: ~270 MB | Labels: "Very Negative" / "Negative" / "Neutral" / "Positive" / "Very Positive"
    "🌐 DistilBERT 5 Classes — Muito Neg a Muito Pos": {
        "id": "tabularisai/multilingual-sentiment-analysis",
        "tipo": "cinco_classes",
        "descricao": "Fine-tuned com dados sintéticos em 22 idiomas incluindo português. "
                     "Escala de 5 níveis. Licença Apache 2.0.",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# MAPEAMENTO DE LABELS → EXIBIÇÃO PADRONIZADA
#
# Centralizar aqui evita lógica espalhada pelo código e facilita adicionar
# novos modelos no futuro: basta incluir o tipo e o mapeamento abaixo.
# ──────────────────────────────────────────────────────────────────────────────
MAPA_LABELS = {

    # Modelo 1: nlptown retorna "1 star", "2 stars", etc.
    "estrelas": {
        "1 star":  ("⭐☆☆☆☆", "Muito Negativo 😠"),
        "2 stars": ("⭐⭐☆☆☆", "Negativo 😞"),
        "3 stars": ("⭐⭐⭐☆☆", "Neutro / Misto 😐"),
        "4 stars": ("⭐⭐⭐⭐☆", "Positivo 🙂"),
        "5 stars": ("⭐⭐⭐⭐⭐", "Muito Positivo 😄"),
    },

    # Modelo 2: lxyuan retorna "positive", "neutral", "negative" (minúsculas)
    "polaridade": {
        "positive": ("🟢", "Positivo 🙂"),
        "neutral":  ("🟡", "Neutro 😐"),
        "negative": ("🔴", "Negativo 😞"),
    },

    # Modelo 3: tabularisai retorna "Very Negative", "Negative", etc.
    "cinco_classes": {
        "Very Negative": ("🔴🔴", "Muito Negativo 😠"),
        "Negative":      ("🔴",   "Negativo 😞"),
        "Neutral":       ("🟡",   "Neutro 😐"),
        "Positive":      ("🟢",   "Positivo 🙂"),
        "Very Positive": ("🟢🟢", "Muito Positivo 😄"),
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO DO MODELO COM CACHE
#
# @st.cache_resource cria uma entrada de cache por valor único de `model_id`.
# O modelo é baixado uma única vez e mantido em memória durante a sessão,
# mesmo que o script seja reexecutado a cada interação do usuário.
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_id: str):
    return pipeline(
        "sentiment-analysis",
        model=model_id,
        truncation=True,   # textos > 512 tokens são truncados; evita erros
        max_length=512,
    )

# ──────────────────────────────────────────────────────────────────────────────
# FUNÇÃO DE NORMALIZAÇÃO DE RESULTADO
#
# Recebe o dict retornado pelo pipeline ({"label": ..., "score": ...}) e
# o tipo do modelo, e devolve uma representação padronizada para exibição
# e para gravação no histórico.
# ──────────────────────────────────────────────────────────────────────────────
def normalizar_resultado(res: dict, tipo: str) -> dict:
    label_raw = res["label"]
    score     = res["score"]

    mapa = MAPA_LABELS.get(tipo, {})
    icone, descricao = mapa.get(label_raw, ("❓", label_raw))   # fallback seguro

    return {
        "label_raw": label_raw,
        "icone":     icone,
        "descricao": descricao,
        "score":     score,
    }

# ──────────────────────────────────────────────────────────────────────────────
# ESTADO DA SESSÃO
# ──────────────────────────────────────────────────────────────────────────────
if "historico" not in st.session_state:
    st.session_state.historico = []

# ──────────────────────────────────────────────────────────────────────────────
# BARRA LATERAL
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🛠️ Configurações")

nome_modelo = st.sidebar.selectbox(
    "Escolha o modelo:",
    list(MODELOS.keys())
)
config_modelo = MODELOS[nome_modelo]

# Carrega (ou recupera do cache) o modelo selecionado
analyzer = load_model(config_modelo["id"])

# Caixa de informação sobre o modelo ativo
st.sidebar.divider()
st.sidebar.markdown("**Modelo ativo:**")
st.sidebar.code(config_modelo["id"], language=None)
st.sidebar.caption(config_modelo["descricao"])

st.sidebar.divider()
if st.sidebar.button("🗑️ Limpar Histórico"):
    st.session_state.historico = []
    st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# INTERFACE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────
st.title("🧠 Dashboard de Análise Comparativa de Sentimentos")
st.caption(
    "Experimente o mesmo texto em modelos diferentes e compare os resultados. "
    "Todos os modelos são gratuitos e rodam diretamente no Streamlit Cloud."
)

aba_entrada, aba_historico = st.tabs(["📝 Análise de Texto", "📊 Histórico Comparativo"])

# ──────────────────────────────────────────────────────────────────────────────
# ABA 1 — ENTRADA E ANÁLISE
# ──────────────────────────────────────────────────────────────────────────────
with aba_entrada:
    st.subheader(f"Analisando com: {nome_modelo}")

    texto_input = st.text_area(
        "Cole suas frases abaixo (uma por linha):",
        placeholder="Ex:\nAdorei o atendimento, foi excelente!\nProduto chegou quebrado, péssima experiência.\nMais ou menos, esperava mais.",
        height=180,
    )

    if st.button("🚀 Executar Análise", type="primary"):
        linhas = [l.strip() for l in texto_input.split("\n") if l.strip()]

        if not linhas:
            st.warning("Insira ao menos uma frase antes de executar.")
        else:
            with st.spinner(f"Processando {len(linhas)} frase(s)..."):
                novos = []
                for linha in linhas:
                    res_raw     = analyzer(linha)[0]
                    normalizado = normalizar_resultado(res_raw, config_modelo["tipo"])
                    novos.append({
                        "Frase":     linha,
                        "Modelo":    nome_modelo,
                        "Resultado": f"{normalizado['icone']}  {normalizado['descricao']}",
                        "Confiança": f"{normalizado['score']:.1%}",
                    })
                    st.session_state.historico.append(novos[-1])

            # Exibe resultado imediato desta rodada
            st.success(f"✅ {len(novos)} frase(s) analisada(s). Resultado:")
            df_novo = pd.DataFrame(novos)
            st.dataframe(df_novo, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# ABA 2 — HISTÓRICO E COMPARATIVO
# ──────────────────────────────────────────────────────────────────────────────
with aba_historico:
    st.subheader("Resultados Consolidados")

    if not st.session_state.historico:
        st.info("Nenhuma análise realizada ainda. Use a aba de entrada para começar.")
    else:
        df = pd.DataFrame(st.session_state.historico)

        # Filtro por modelo — útil quando o aluno testa vários modelos na mesma sessão
        modelos_disponiveis = df["Modelo"].unique().tolist()
        filtro = st.multiselect(
            "Filtrar por modelo:",
            options=modelos_disponiveis,
            default=modelos_disponiveis,
        )
        df_filtrado = df[df["Modelo"].isin(filtro)]

        st.dataframe(df_filtrado, use_container_width=True, hide_index=True)

        # Métricas rápidas
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de frases", len(df_filtrado))
        col2.metric("Modelos usados", df_filtrado["Modelo"].nunique())
        col3.metric("Frases únicas", df_filtrado["Frase"].nunique())

        st.divider()

        # Download
        csv = df_filtrado.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Baixar CSV",
            data=csv,
            file_name="comparativo_sentimentos.csv",
            mime="text/csv",
        )
