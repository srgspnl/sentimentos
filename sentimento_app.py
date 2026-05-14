"""
Sentimento 2025 — App Streamlit
Análise de sentimentos com BERT multilíngue (nlptown)
"""

import re
import io
from collections import defaultdict
from pathlib import Path

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Configuração da página ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentimento 2025",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS customizado ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    border-right: 1px solid #2a2a4a;
}
[data-testid="stSidebar"] * {
    color: #e0e0f0 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.03em;
}

/* Cards de resultado */
.resultado-card {
    background: #f8f8ff;
    border-left: 4px solid #6c63ff;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size: 0.92rem;
    box-shadow: 0 2px 8px rgba(108,99,255,0.07);
}
.resultado-card .frase {
    color: #1a1a2e;
    font-weight: 600;
    margin-bottom: 4px;
}
.resultado-card .detalhe {
    color: #555;
    font-size: 0.83rem;
}

/* Métricas */
.metric-box {
    background: linear-gradient(135deg, #1a1a2e, #2d2d5e);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: white;
}

/* Badge de sentimento */
.badge-muito-pos  { background:#1976d2; color:white; padding:3px 10px; border-radius:20px; font-size:0.78rem; }
.badge-pos        { background:#388e3c; color:white; padding:3px 10px; border-radius:20px; font-size:0.78rem; }
.badge-neutro     { background:#fbc02d; color:#333;  padding:3px 10px; border-radius:20px; font-size:0.78rem; }
.badge-neg        { background:#f57c00; color:white; padding:3px 10px; border-radius:20px; font-size:0.78rem; }
.badge-muito-neg  { background:#d32f2f; color:white; padding:3px 10px; border-radius:20px; font-size:0.78rem; }

div[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace;
    font-size: 2rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constantes ─────────────────────────────────────────────────────────────────
NOME_MODELO = "nlptown/bert-base-multilingual-uncased-sentiment"

MAPA_ESTRELAS = {
    1: ("Muito Negativo", "😠", "#d32f2f", "badge-muito-neg"),
    2: ("Negativo",       "😞", "#f57c00", "badge-neg"),
    3: ("Neutro / Misto", "😐", "#fbc02d", "badge-neutro"),
    4: ("Positivo",       "🙂", "#388e3c", "badge-pos"),
    5: ("Muito Positivo", "😄", "#1976d2", "badge-muito-pos"),
}

FRASES_DEMO = [
    "Estou muito feliz com o atendimento, foi excelente!",
    "O serviço foi horrível, estou muito insatisfeito.",
    "Não sei como me sinto sobre isso.",
    "Foi ok, mas poderia ser melhor.",
    "Péssima experiência, nunca mais compro aqui.",
    "Que móvel excelente! Melhor cadeira que já comprei.",
    "Tudo certo, conforme solicitado.",
]

# ── Cache do modelo ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def carregar_modelo():
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model=NOME_MODELO,
        truncation=True,
        max_length=512,
    )

# ── Funções auxiliares ─────────────────────────────────────────────────────────
def extrair_estrelas(label: str) -> int:
    match = re.search(r"(\d)", label)
    return int(match.group(1)) if match else 3

def limpar_frase(texto: str) -> str:
    return texto.strip().rstrip(".")

def analisar_frases(analyzer, frases: list[str]) -> list[dict]:
    resultados = []
    for frase in frases:
        if not frase.strip():
            continue
        try:
            r = analyzer(frase)[0]
            estrelas = extrair_estrelas(r["label"])
            resultados.append({
                "frase":    frase,
                "estrelas": estrelas,
                "score":    r["score"],
                "label":    MAPA_ESTRELAS[estrelas][0],
                "emoji":    MAPA_ESTRELAS[estrelas][1],
                "cor":      MAPA_ESTRELAS[estrelas][2],
                "badge":    MAPA_ESTRELAS[estrelas][3],
            })
        except Exception as e:
            resultados.append({
                "frase": frase, "estrelas": 0, "score": 0,
                "label": f"Erro: {e}", "emoji": "⚠️", "cor": "#999", "badge": "",
            })
    return resultados

def gerar_grafico(contagem: dict) -> plt.Figure:
    estrelas_lista = list(range(1, 6))
    totais         = [contagem.get(e, 0) for e in estrelas_lista]
    cores          = [MAPA_ESTRELAS[e][2] for e in estrelas_lista]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("#f8f8ff")

    # Barras
    bars = axes[0].bar(estrelas_lista, totais, color=cores, edgecolor="white",
                       width=0.6, linewidth=1.5)
    axes[0].set_facecolor("#f0f0fa")
    axes[0].set_xlabel("Estrelas", fontsize=10)
    axes[0].set_ylabel("Quantidade", fontsize=10)
    axes[0].set_title("Distribuição por Estrelas", fontsize=11, fontweight="bold")
    axes[0].set_xticks(estrelas_lista)
    axes[0].set_xticklabels([f"{e}⭐" for e in estrelas_lista])
    axes[0].spines[["top","right"]].set_visible(False)
    for bar, v in zip(bars, totais):
        if v > 0:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                str(v), ha="center", va="bottom", fontweight="bold", fontsize=10
            )

    # Pizza
    dados = [(e, t) for e, t in zip(estrelas_lista, totais) if t > 0]
    if dados:
        ep, tp = zip(*dados)
        cp = [MAPA_ESTRELAS[e][2] for e in ep]
        lp = [f"{e}⭐ — {MAPA_ESTRELAS[e][0]} ({t})" for e, t in dados]
        axes[1].pie(tp, colors=cp, autopct="%1.1f%%", startangle=90,
                    pctdistance=0.75, wedgeprops={"linewidth":2, "edgecolor":"white"})
        axes[1].legend(lp, loc="lower center", bbox_to_anchor=(0.5, -0.18),
                       fontsize=8, frameon=False, ncol=2)
        axes[1].set_facecolor("#f8f8ff")
    else:
        axes[1].text(0.5, 0.5, "Sem dados", ha="center", va="center")

    axes[1].set_title("Proporção por Categoria", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig

def exibir_resultados(resultados: list[dict]):
    """Exibe cards de resultado e métricas."""
    if not resultados:
        st.warning("Nenhum resultado para exibir.")
        return

    validos = [r for r in resultados if r["estrelas"] > 0]
    contagem = defaultdict(int)
    for r in validos:
        contagem[r["estrelas"]] += 1

    # Métricas resumo
    total = len(validos)
    media = sum(r["estrelas"] for r in validos) / total if total else 0
    positivos = sum(1 for r in validos if r["estrelas"] >= 4)
    negativos = sum(1 for r in validos if r["estrelas"] <= 2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Total analisado", total)
    c2.metric("⭐ Média de estrelas", f"{media:.1f}")
    c3.metric("🟢 Positivos (4-5⭐)", positivos)
    c4.metric("🔴 Negativos (1-2⭐)", negativos)

    st.markdown("---")

    # Gráfico
    st.subheader("📊 Visualização")
    fig = gerar_grafico(contagem)
    st.pyplot(fig)

    # Download do gráfico
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    st.download_button("⬇️ Baixar gráfico (PNG)", buf.getvalue(),
                       "distribuicao_sentimentos.png", "image/png")

    st.markdown("---")
    st.subheader("🗂️ Detalhes por frase")
    for r in resultados:
        badge_html = f'<span class="{r["badge"]}">{r["emoji"]} {r["label"]}</span>' if r["badge"] else r["label"]
        conf_html  = f'<span style="color:#888">Confiança: {r["score"]:.1%}</span>' if r["score"] else ""
        st.markdown(f"""
        <div class="resultado-card">
            <div class="frase">"{r['frase']}"</div>
            <div class="detalhe">{badge_html} &nbsp; {'⭐' * r['estrelas']} &nbsp; {conf_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # Download CSV
    import pandas as pd
    df = pd.DataFrame([{
        "frase": r["frase"],
        "estrelas": r["estrelas"],
        "sentimento": r["label"],
        "confianca_pct": round(r["score"] * 100, 2),
    } for r in validos])
    csv = df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("⬇️ Exportar CSV", csv,
                       "resultados_sentimento.csv", "text/csv")

# ── Sidebar — Menu de navegação ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Sentimento 2025")
    st.markdown("*Análise com BERT multilíngue*")
    st.markdown("---")

    pagina = st.radio(
        "Navegação",
        options=[
            "🏠  Início",
            "⚡  Análise Rápida",
            "📂  Análise por Arquivo",
            "✏️  Análise por Texto",
            "ℹ️  Sobre o Modelo",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Modelo:**")
    st.code("nlptown/bert-base\n-multilingual-uncased\n-sentiment", language=None)
    st.markdown("**Escala:** 1 ⭐ a 5 ⭐⭐⭐⭐⭐")

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: INÍCIO
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "🏠  Início":
    st.title("🧠 Sentimento 2025")
    st.markdown("#### Análise de sentimentos em português com BERT multilíngue")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**⚡ Análise Rápida**\n\nTeste com frases de exemplo pré-carregadas. Ideal para verificar o modelo rapidamente.")
    with col2:
        st.info("**📂 Análise por Arquivo**\n\nEnvie um `.txt` com uma opinião por linha e processe tudo de uma vez.")
    with col3:
        st.info("**✏️ Análise por Texto**\n\nDigite ou cole quantas frases quiser e analise em tempo real.")

    st.markdown("---")
    st.markdown("### Como funciona")
    st.markdown("""
    O modelo **`nlptown/bert-base-multilingual-uncased-sentiment`** classifica textos
    em uma escala de **1 a 5 estrelas**:

    | Estrelas | Sentimento | Emoji |
    |----------|------------|-------|
    | ⭐ | Muito Negativo | 😠 |
    | ⭐⭐ | Negativo | 😞 |
    | ⭐⭐⭐ | Neutro / Misto | 😐 |
    | ⭐⭐⭐⭐ | Positivo | 🙂 |
    | ⭐⭐⭐⭐⭐ | Muito Positivo | 😄 |
    """)

    st.markdown("---")
    st.markdown("*Selecione uma opção no menu lateral para começar.*")

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: ANÁLISE RÁPIDA
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "⚡  Análise Rápida":
    st.title("⚡ Análise Rápida")
    st.markdown("Frases de demonstração pré-carregadas para testar o modelo.")

    with st.expander("📝 Frases de teste (clique para ver/editar)", expanded=True):
        frases_editadas = []
        for i, f in enumerate(FRASES_DEMO):
            val = st.text_input(f"Frase {i+1}", value=f, key=f"demo_{i}")
            frases_editadas.append(val)

    if st.button("▶️ Executar análise rápida", type="primary", use_container_width=True):
        with st.spinner("⏳ Carregando modelo e analisando..."):
            analyzer = carregar_modelo()
            frases_limpas = [limpar_frase(f) for f in frases_editadas if f.strip()]
            resultados = analisar_frases(analyzer, frases_limpas)
            st.session_state["resultados_rapido"] = resultados
        st.success(f"✅ {len(resultados)} frase(s) analisada(s)!")

    if "resultados_rapido" in st.session_state:
        exibir_resultados(st.session_state["resultados_rapido"])

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: ANÁLISE POR ARQUIVO
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📂  Análise por Arquivo":
    st.title("📂 Análise por Arquivo")
    st.markdown("Envie um arquivo `.txt` com **uma opinião por linha**.")

    st.info("""
    **Formato esperado do arquivo:**
    ```
    O produto chegou no prazo e funcionou perfeitamente.
    Atendimento péssimo, vou reclamar no Reclame Aqui.
    Mais ou menos, esperava mais pelo preço.
    ```
    """)

    arquivo = st.file_uploader("Selecione o arquivo .txt", type=["txt"])

    if arquivo:
        conteudo = arquivo.read().decode("utf-8", errors="replace")
        frases_raw = conteudo.splitlines()
        frases = [limpar_frase(f) for f in frases_raw if f.strip()]

        st.success(f"✅ **{len(frases)} frase(s)** carregada(s) do arquivo `{arquivo.name}`")

        with st.expander("👁️ Pré-visualizar frases"):
            for i, f in enumerate(frases[:20], 1):
                st.markdown(f"`{i:03d}` {f}")
            if len(frases) > 20:
                st.caption(f"... e mais {len(frases)-20} frase(s).")

        col_a, col_b = st.columns(2)
        batch = col_a.number_input("Tamanho do lote (batch)", min_value=1,
                                   max_value=64, value=16, step=8,
                                   help="Frases processadas simultaneamente. Maior = mais rápido com GPU.")
        _ = col_b  # espaço reservado

        if st.button("▶️ Processar arquivo", type="primary", use_container_width=True):
            progress = st.progress(0, text="Inicializando modelo...")
            with st.spinner(""):
                analyzer = carregar_modelo()
                resultados = []
                total = len(frases)
                for i, frase in enumerate(frases):
                    r = analisar_frases(analyzer, [frase])
                    resultados.extend(r)
                    progress.progress((i + 1) / total,
                                      text=f"Processando {i+1}/{total}...")
            progress.empty()
            st.session_state["resultados_arquivo"] = resultados
            st.success(f"✅ Concluído! {len(resultados)} frase(s) analisada(s).")

    if "resultados_arquivo" in st.session_state:
        exibir_resultados(st.session_state["resultados_arquivo"])

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: ANÁLISE POR TEXTO
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "✏️  Análise por Texto":
    st.title("✏️ Análise por Texto")
    st.markdown("Digite ou cole frases abaixo — **uma por linha**.")

    texto = st.text_area(
        "Insira as frases aqui:",
        height=200,
        placeholder="Exemplo:\nAmei o produto, chegou antes do prazo!\nAtendimento deixou a desejar.\nNão tenho opinião formada ainda.",
    )

    col1, col2 = st.columns([1, 3])
    analisar = col1.button("▶️ Analisar", type="primary", use_container_width=True)
    limpar   = col2.button("🗑️ Limpar resultados", use_container_width=False)

    if limpar and "resultados_texto" in st.session_state:
        del st.session_state["resultados_texto"]
        st.rerun()

    if analisar:
        frases = [limpar_frase(f) for f in texto.splitlines() if f.strip()]
        if not frases:
            st.warning("⚠️ Digite ao menos uma frase.")
        else:
            with st.spinner(f"Analisando {len(frases)} frase(s)..."):
                analyzer = carregar_modelo()
                resultados = analisar_frases(analyzer, frases)
                st.session_state["resultados_texto"] = resultados
            st.success(f"✅ {len(resultados)} frase(s) analisada(s)!")

    if "resultados_texto" in st.session_state:
        exibir_resultados(st.session_state["resultados_texto"])

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: SOBRE O MODELO
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "ℹ️  Sobre o Modelo":
    st.title("ℹ️ Sobre o Modelo")

    st.markdown("""
    ### `nlptown/bert-base-multilingual-uncased-sentiment`

    Modelo BERT multilíngue fine-tuned para classificação de sentimentos em avaliações de produtos.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Características:**
        - 🌍 Treinado em **6 idiomas**: inglês, alemão, holandês, francês, espanhol e italiano
        - 🇧🇷 Boa generalização para **português**
        - 📏 Aceita até **512 tokens** por entrada
        - 💾 Tamanho: ~700 MB (baixado uma vez, fica em cache)
        - 🎯 Saída: escala de 1 a 5 estrelas
        """)
    with col2:
        st.markdown("""
        **Limitações:**
        - Não foi treinado nativamente em português
        - Textos > 512 tokens são truncados
        - Ironias e sarcasmo podem confundir o modelo
        - Domínio: avaliações de produtos (pode ter desvio em outros contextos)
        """)

    st.markdown("---")
    st.markdown("### 💡 Alternativas para melhorar os resultados")

    st.markdown("""
    | Modelo | Característica |
    |--------|---------------|
    | `HeyLucasLeao/bert-base-portuguese-cased-sentiment` | BERT treinado nativamente em português |
    | `pyabsa` | Análise por **aspecto** (produto, entrega, atendimento) |
    | Gradio | Interface alternativa mais simples de hospedar |

    **Otimizações de performance:**
    ```python
    # Usar GPU (se disponível)
    pipeline(..., device=0)

    # Processar em lote (muito mais rápido)
    pipeline(lista_de_frases, batch_size=32, truncation=True)
    ```
    """)

    st.markdown("---")
    st.markdown("### 📦 Dependências necessárias")
    st.code("""
pip install streamlit transformers torch matplotlib pandas
    """, language="bash")

    st.markdown("### ▶️ Como executar")
    st.code("""
streamlit run sentimento_app.py
    """, language="bash")
