import streamlit as st
import pandas as pd
import re
from transformers import pipeline

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Sentimento 2025 - On-Demand", page_icon="⚙️", layout="wide")

# --- DICIONÁRIO DE MODELOS ---
MODELOS = {
    "Nenhum (Aguardando Seleção)": None,
    "BERT Multilingual (1-5 Estrelas)": "nlptown/bert-base-multilingual-uncased-sentiment",
    "BERT Português (Pos/Neg/Neu)": "pabloms/bert-base-portuguese-lowercased-sentiment",
    "DistilBERT (Mais leve/Rápido)": "distilbert-base-uncased-finetuned-sst-2-english"
}

# --- FUNÇÃO DE CARREGAMENTO (LAZY LOADING) ---
@st.cache_resource(show_spinner="Baixando e carregando pesos do modelo...")
def get_analyzer(model_endpoint):
    if model_endpoint is None:
        return None
    # Usamos o device=-1 para forçar CPU, economizando memória no container
    return pipeline("sentiment-analysis", model=model_endpoint, device=-1)

# --- ESTADO DA SESSÃO ---
if 'historico' not in st.session_state:
    st.session_state.historico = []

# --- BARRA LATERAL ---
st.sidebar.title("🎮 Painel de Controle")

# Seleção do modelo (Inicia com Nenhum)
escolha = st.sidebar.selectbox("Escolha um modelo para iniciar:", list(MODELOS.keys()))
endpoint = MODELOS[escolha]

# Botão de Reset Total
if st.sidebar.button("♻️ Reset Total e Limpar Cache"):
    st.cache_resource.clear()
    st.session_state.historico = []
    st.rerun()

st.sidebar.divider()
if endpoint:
    st.sidebar.success(f"**Modelo Ativo:**\n{escolha}")
else:
    st.sidebar.warning("Nenhum modelo carregado.")

# --- INTERFACE PRINCIPAL ---
st.title("🧠 Análise de Sentimentos Sob Demanda")

if endpoint is None:
    st.info("👋 Bem-vindo! Para começar, selecione um modelo na barra lateral esquerda.")
    st.markdown("""
    Esta aplicação foi configurada para economizar recursos:
    1. O modelo só é baixado após a sua escolha.
    2. Você pode dar 'Reset' para liberar espaço em disco no servidor.
    """)
else:
    # Só tenta carregar o pipeline se houver um endpoint selecionado
    analyzer = get_analyzer(endpoint)
    
    aba1, aba2 = st.tabs(["📝 Processamento", "📊 Histórico"])

    with aba1:
        st.subheader(f"Analisando com: {escolha}")
        texto_input = st.text_area("Cole suas frases (uma por linha):", height=150)
        
        if st.button("Executar Análise"):
            if texto_input.strip():
                linhas = [l.strip() for l in texto_input.split('\n') if l.strip()]
                with st.spinner("Processando..."):
                    for linha in linhas:
                        res = analyzer(linha)[0]
                        # Lógica simples para visualização
                        label = res['label']
                        visualizacao = "⭐" * int(re.search(r"\d", label).group(1)) if "star" in label.lower() else label
                        
                        st.session_state.historico.append({
                            "Frase": linha,
                            "Modelo": escolha,
                            "Resultado": visualizacao,
                            "Confiança": f"{res['score']:.2%}"
                        })
                st.success("Concluído!")
            else:
                st.error("Por favor, insira algum texto.")

    with aba2:
        if st.session_state.historico:
            df = pd.DataFrame(st.session_state.historico)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 Exportar Resultados", data=csv, file_name="analise.csv")
        else:
            st.write("O histórico aparecerá aqui após o processamento.")