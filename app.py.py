import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from transformers import pipeline
from collections import defaultdict

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Sentimento 2025 - Transformers", page_icon="🧠", layout="wide")

# --- CACHE DO MODELO ---
# O cache evita que o modelo seja recarregado a cada clique na interface
@st.cache_resource
def load_model():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    return pipeline("sentiment-analysis", model=model_name, truncation=True, max_length=512)

analyzer = load_model()

# --- MAPEAMENTOS ---
MAPA_ESTRELAS = {
    1: "Muito Negativo 😠",
    2: "Negativo 😞",
    3: "Neutro / Misto 😐",
    4: "Positivo 🙂",
    5: "Muito Positivo 😄",
}

def extrair_estrelas(label):
    match = re.search(r"(\d)", label)
    return int(match.group(1)) if match else 3

# --- BARRA LATERAL (MENUS) ---
st.sidebar.title("📌 Menu de Navegação")
aba = st.sidebar.radio("Escolha uma funcionalidade:", 
                         ["Sobre o Projeto", "Teste em Tempo Real", "Análise em Lote (Arquivo)"])

# --- CONTEÚDO DAS ABAS ---

if aba == "Sobre o Projeto":
    st.title("🧠 Análise de Sentimentos com Transformers")
    st.markdown("""
    Este projeto utiliza o modelo **BERT multilíngue** para classificar textos em uma escala de 1 a 5 estrelas.
    
    - **Modelo:** `nlptown/bert-base-multilingual-uncased-sentiment`
    - **Idiomas:** Treinado em 6 idiomas com ótima performance em Português.
    - **Tecnologias:** Hugging Face Transformers, PyTorch e Streamlit.
    """)
    st.info("💡 Dica: Use o menu lateral para alternar entre testes manuais ou carregamento de arquivos.")

elif aba == "Teste em Tempo Real":
    st.title("📝 Teste Rápido")
    frase_usuario = st.text_area("Digite uma frase para análise:", placeholder="Ex: O atendimento foi excelente, adorei!")
    
    if st.button("Analisar Sentimento"):
        if frase_usuario:
            with st.spinner("Processando..."):
                res = analyzer(frase_usuario)[0]
                estrelas = extrair_estrelas(res['label'])
                
                col1, col2 = st.columns(2)
                col1.metric("Classificação", f"{estrelas} Estrelas")
                col2.metric("Confiança", f"{res['score']:.2%}")
                
                st.subheader(f"Resultado: {MAPA_ESTRELAS[estrelas]}")
                st.write("⭐" * estrelas)
        else:
            st.warning("Por favor, digite algo antes de analisar.")

elif aba == "Análise em Lote (Arquivo)":
    st.title("📂 Análise em Lote")
    st.write("Faça upload de um arquivo `.txt` com uma frase por linha.")
    
    arquivo_upload = st.file_uploader("Escolha o arquivo TXT", type=["txt"])
    
    if arquivo_upload:
        conteudo = arquivo_upload.read().decode("utf-8")
        linhas = [linha.strip() for linha in conteudo.split("\n") if linha.strip()]
        
        if st.button(f"Processar {len(linhas)} frases"):
            progresso = st.progress(0)
            resultados_lote = []
            contagem = defaultdict(int)

            for i, linha in enumerate(linhas):
                res = analyzer(linha)[0]
                estrelas = extrair_estrelas(res['label'])
                contagem[estrelas] += 1
                resultados_lote.append({
                    "Texto": linha,
                    "Estrelas": estrelas,
                    "Sentimento": MAPA_ESTRELAS[estrelas],
                    "Confiança": f"{res['score']:.2%}"
                })
                progresso.progress((i + 1) / len(linhas))

            df = pd.DataFrame(resultados_lote)
            
            # --- DASHBOARD DE RESULTADOS ---
            st.success("Análise concluída!")
            
            c1, c2 = st.columns([1, 1])
            
            with c1:
                st.subheader("Distribuição")
                fig, ax = plt.subplots()
                cores = ["#d32f2f", "#f57c00", "#fbc02d", "#388e3c", "#1976d2"]
                counts = [contagem[i] for i in range(1, 6)]
                ax.bar([f"{i}★" for i in range(1, 6)], counts, color=cores)
                st.pyplot(fig)
            
            with c2:
                st.subheader("Dados Detalhados")
                st.dataframe(df)

            # --- DOWNLOAD ---
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 Baixar Resultados (CSV)", data=csv, file_name="resultados_sentimento.csv", mime="text/csv")