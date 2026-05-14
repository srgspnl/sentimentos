import streamlit as st
import pandas as pd
import re
from transformers import pipeline

# --- CONFIGURAÇÃO E MODELO ---
st.set_page_config(page_title="Sentimento 2025 - Multi-análise", page_icon="⭐", layout="wide")

@st.cache_resource
def load_model():
    # Modelo BERT multilíngue para 1-5 estrelas
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

analyzer = load_model()

def extrair_estrelas(label):
    match = re.search(r"(\d)", label)
    return int(match.group(1)) if match else 3

# --- ESTADO DA SESSÃO (Para persistir a lista de frases) ---
if 'historico' not in st.session_state:
    st.session_state.historico = []

# --- INTERFACE LATERAL ---
st.sidebar.title("Configurações")
if st.sidebar.button("Limpar Histórico"):
    st.session_state.historico = []
    st.rerun()

aba = st.radio("Escolha o método de entrada:", 
               ["Colar Múltiplas Frases", "Carregar Arquivo TXT", "Ver Histórico e Scores"])

# --- LÓGICA DE PROCESSAMENTO ---
def processar_texto(texto):
    # Divide por quebra de linha e remove vazios
    linhas = [l.strip() for l in texto.split('\n') if l.strip()]
    if not linhas:
        st.warning("Nenhum texto detectado.")
        return

    novos_resultados = []
    with st.spinner(f"Analisando {len(linhas)} frases..."):
        for linha in linhas:
            res = analyzer(linha)[0]
            estrelas = extrair_estrelas(res['label'])
            novos_resultados.append({
                "Frase": linha,
                "Score": estrelas,
                "Representação": "⭐" * estrelas,
                "Confiança": f"{res['score']:.2%}"
            })
    
    # Adiciona ao histórico da sessão
    st.session_state.historico.extend(novos_resultados)
    st.success(f"{len(linhas)} frases processadas com sucesso!")

# --- ABAS DE INTERFACE ---

if aba == "Colar Múltiplas Frases":
    st.subheader("Análise em Massa")
    texto_input = st.text_area("Cole aqui suas frases (uma por linha):", height=200, 
                               placeholder="Exemplo:\nO produto é ótimo\nNão gostei do atendimento\nEntrega demorada")
    
    if st.button("Analisar Tudo"):
        processar_texto(texto_input)

elif aba == "Carregar Arquivo TXT":
    st.subheader("Upload de Arquivo")
    arquivo = st.file_uploader("Escolha um arquivo .txt", type="txt")
    if arquivo:
        conteudo = arquivo.read().decode("utf-8")
        if st.button("Processar Arquivo"):
            processar_texto(conteudo)

elif aba == "Ver Histórico e Scores":
    st.subheader("Resultados Consolidados")
    
    if st.session_state.historico:
        df = pd.DataFrame(st.session_state.historico)
        
        # Exibição da Tabela com Estrelas
        st.dataframe(
            df,
            column_config={
                "Representação": st.column_config.TextColumn("Visualização Estrelas"),
                "Score": st.column_config.NumberColumn("Estrelas (1-5)", format="%d ⭐")
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Opção de Download
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Exportar para CSV", data=csv, file_name="analise_sentimento.csv")
    else:
        st.info("Ainda não há frases processadas. Use as outras abas para começar.")