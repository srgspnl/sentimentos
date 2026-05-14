import streamlit as st
import pandas as pd
import re
from transformers import pipeline

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Sentimento 2025 - Multi-Model", page_icon="🤖", layout="wide")

# --- DICIONÁRIO DE MODELOS DISPONÍVEIS ---
# Adicionei modelos que funcionam bem em Português
MODELOS = {
    "BERT Multilingual (1-5 Estrelas)": "nlptown/bert-base-multilingual-uncased-sentiment",
    "BERT Português (Pos/Neg/Neu)": "pabloms/bert-base-portuguese-lowercased-sentiment",
    "Twitter RoBERTa (Emoções)": "cardiffnlp/twitter-xlm-roberta-base-sentiment"
}

# --- CARREGAMENTO DO MODELO COM CACHE ---
@st.cache_resource
def load_selected_model(model_name):
    return pipeline("sentiment-analysis", model=model_name)

# --- FUNÇÕES DE APOIO ---
def extrair_score(res):
    label = res['label']
    # Lógica para converter estrelas em visualização (se o modelo for o nlptown)
    match = re.search(r"(\d)", label)
    if match:
        estrelas = int(match.group(1))
        return "⭐" * estrelas
    return label  # Retorna a label textual (ex: POSITIVE) para outros modelos

# --- ESTADO DA SESSÃO ---
if 'historico' not in st.session_state:
    st.session_state.historico = []

# --- INTERFACE LATERAL ---
st.sidebar.title("🛠️ Configurações de IA")
modelo_selecionado = st.sidebar.selectbox("Selecione o Modelo de NLP:", list(MODELOS.keys()))
endpoint_modelo = MODELOS[modelo_selecionado]

# Carrega o modelo escolhido
analyzer = load_selected_model(endpoint_modelo)

if st.sidebar.button("Limpar Histórico"):
    st.session_state.historico = []
    st.rerun()

st.sidebar.divider()
st.sidebar.info(f"**Modelo Ativo:**\n`{endpoint_modelo}`")

# --- INTERFACE PRINCIPAL ---
st.title("🧠 Dashboard de Análise Comparativa")
aba = st.tabs(["📝 Entrada de Texto", "📂 Histórico de Análises"])

# --- LÓGICA DE PROCESSAMENTO ---
def processar_texto(texto):
    linhas = [l.strip() for l in texto.split('\n') if l.strip()]
    if not linhas:
        st.warning("Insira ao menos uma frase.")
        return

    with st.spinner(f"Analisando com {modelo_selecionado}..."):
        for linha in linhas:
            res = analyzer(linha)[0]
            visualizacao = extrair_score(res)
            
            st.session_state.historico.append({
                "Frase": linha,
                "Modelo": modelo_selecionado,
                "Resultado": visualizacao,
                "Confiança": f"{res['score']:.2%}"
            })

# --- ABA 1: ENTRADA ---
with aba[0]:
    st.subheader("Análise de Sentimento em Massa")
    texto_input = st.text_area("Cole suas frases (uma por linha):", height=200)
    
    if st.button("Executar Análise"):
        processar_texto(texto_input)
        st.success("Processamento concluído! Verifique a aba de Histórico.")

# --- ABA 2: HISTÓRICO ---
with aba[1]:
    st.subheader("Resultados Consolidados")
    if st.session_state.historico:
        df = pd.DataFrame(st.session_state.historico)
        
        # Filtro para comparar modelos no histórico
        modelos_no_hist = df['Modelo'].unique()
        filtro = st.multiselect("Filtrar por Modelo:", modelos_no_hist, default=modelos_no_hist)
        
        df_filtrado = df[df['Modelo'].isin(filtro)]
        
        st.dataframe(df_filtrado, use_container_width=True, hide_index=True)
        
        # Download
        csv = df_filtrado.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Baixar CSV", data=csv, file_name="comparativo_sentimentos.csv")
    else:
        st.info("Nenhuma análise realizada ainda.")