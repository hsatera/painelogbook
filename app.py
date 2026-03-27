import streamlit as st
import pandas as pd
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import csv

# --- Proteção simples por senha ---
PASSWORD = "Residenci@MFC123!"

if "acesso_liberado" not in st.session_state:
    st.session_state["acesso_liberado"] = False

if not st.session_state["acesso_liberado"]:
    st.title("🔒 Acesso restrito")
    senha = st.text_input("Digite a senha para acessar o painel", type="password")
    if st.button("Entrar"):
        if senha == PASSWORD:
            st.session_state["acesso_liberado"] = True
            st.rerun()
        else:
            st.error("Senha incorreta")
    st.stop()

# --- Configurações ---
MONTHS_TRANSLATION = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março",
    4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro",
    10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

nltk.download('stopwords', quiet=True)

st.set_page_config(layout="wide", page_title="Discussões Clínicas", page_icon="📚")

# --- Residentes ---
residentes_dict = {
    'Anchieta': {'R1': ['Gabriella Rodrigues', 'Ana Clara Magalhães'], 'R2': ['Clarissa Goulardins', 'Paulo Okuda']},
    '31 de Março': {'R1': ['Mariana Ribeiro', 'Larissa Eri']},
    'Boa Vista': {'R1': ['Nathália Carmelo', 'André Binas'], 'R2': ['Ana Cecília Venâncio', 'Armindo Albuquerque']},
    'Eulina': {'R1': ['Sarah Ribeiro', 'Raquel Rios'], 'R2': ['Bárbara Nakamuta', 'Helena Lopes de Barros']},
    'Rosália': {'R1': ['Kin Shimabukuro'], 'R2': ['Alice Haddad', 'Letícia Dantas']},
    'San Martin': {'R1': ['Julia Bicas'], 'R2': ['José Camargo Júnior', 'Carolina Almeida']},
    'Santa Bárbara': {'R1': ['Felipe Fedrizzi', 'Julia Marcato'], 'R2': ['Marianna Freitas']},
    'São Marcos': {'R1': ['Ludmila Vilela', 'Clara Guimarães'], 'R2': ['Guilherme Bonelli', 'Mariana de Oliveira']},
    'Village': {'R1': ['Sarah Dariva'], 'R2': ['Mathias Machado', 'Murilo Castro']}
}

resident_metadata = {}
for ubs, anos in residentes_dict.items():
    for ano, nomes in anos.items():
        for nome in nomes:
            resident_metadata[nome] = {'UBS': ubs, 'Ano': ano}

# --- URL ---
GSHEETS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRaHBifgKX5-0Bi4DIBVRJMz2jcLdfLmBg4uvgWXZXwb5ziT6B_OwM7x3oofJWHoUdZQnxrbHHt9YUu/pub?output=csv"

# --- LOAD DATA ---
@st.cache_data(ttl=600)
def load_data(url, resident_metadata):
    try:
        df = pd.read_csv(url, quoting=csv.QUOTE_MINIMAL)
    except:
        st.error("Erro ao carregar dados.")
        st.stop()

    column_mapping = {
        'Carimbo de data/hora': 'Data',
        'Docente/Tutor/preceptor': 'Preceptor',
        'Residente envolvido(a)': 'Residente_antigo',
        'Residente(s) envolvidos(as) (se  houver mais de 1)': 'Residente_multi',
        'UBS': 'UBS',
        'Situação/caso discutida/o (sem identificação)': 'Situacao',
        'Pergunta norteadora da discussão': 'Questao',
        'Principal Módulo do GT Teórico relacionado à discussão': 'Modulo',
        'Referência(s) utilizadas/sugeridas': 'Referencia',
        'Pactuações/encaminhamentos/feedback realizado/procedimento realizado': 'Encaminhamento'
    }

    df.rename(columns={old: new for old, new in column_mapping.items() if old in df.columns}, inplace=True)

    # Garantir colunas
    for col in ['Data','Preceptor','Residente_antigo','Residente_multi','UBS','Situacao','Questao','Modulo','Referencia','Encaminhamento']:
        if col not in df.columns:
            df[col] = ""

    # Datas (Tratamento do código antigo incorporado)
    df['Data'] = df['Data'].astype(str).str.strip()
    df['Data'] = df['Data'].str.replace(r'(\d+)\/(\d+)\/(\d{4})\s', r'\3-\2-\1 ', regex=True)
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df = df.fillna("")

    # --- EXPANSÃO PARA MÚLTIPLOS RESIDENTES ---
    def parse_residents(value):
        if pd.isna(value) or str(value).strip() == "":
            return []
        return [p.strip() for p in re.split(r'[,;\n]+', str(value)) if p.strip()]

    rows = []
    for _, row in df.iterrows():
        nomes_1 = parse_residents(row.get('Residente_antigo', ''))
        nomes_2 = parse_residents(row.get('Residente_multi', ''))
        todos_nomes = list(dict.fromkeys(nomes_1 + nomes_2))
        
        if not todos_nomes:
            todos_nomes = ["Não informado"]

        for nome in todos_nomes:
            new_row = row.drop(labels=['Residente_antigo', 'Residente_multi'], errors='ignore').to_dict()
            new_row['Residente'] = nome
            rows.append(new_row)

    df = pd.DataFrame(rows)

    # Metadata e Mês formatado
    df['AnoResidencia'] = df['Residente'].apply(lambda x: resident_metadata.get(x, {}).get('Ano','Não informado'))
    df['UBS'] = df.apply(lambda row: resident_metadata.get(row['Residente'],{}).get('UBS',row['UBS']), axis=1)
    df['Mes'] = df['Data'].apply(lambda x: f"{MONTHS_TRANSLATION.get(x.month)} de {x.year}" if pd.notna(x) else 'Não informado')

    return df

df_raw = load_data(GSHEETS_URL, resident_metadata)
lista_residentes_validos = list(resident_metadata.keys())
df = df_raw[df_raw['Residente'].isin(lista_residentes_validos)].copy()

# --- SIDEBAR ---
st.sidebar.header("⚙️ Filtros")
selected_residents = st.sidebar.multiselect("Residente", sorted(lista_residentes_validos))
selected_ubs = st.sidebar.multiselect("UBS", sorted(list(residentes_dict.keys())))
selected_modulos = st.sidebar.multiselect("Módulo", sorted(df['Modulo'].unique()))
selected_months = st.sidebar.multiselect("Mês", df.sort_values('Data')['Mes'].unique())

filtered_df = df.copy()
if selected_residents: filtered_df = filtered_df[filtered_df['Residente'].isin(selected_residents)]
if selected_ubs: filtered_df = filtered_df[filtered_df['UBS'].isin(selected_ubs)]
if selected_modulos: filtered_df = filtered_df[filtered_df['Modulo'].isin(selected_modulos)]
if selected_months: filtered_df = filtered_df[filtered_df['Mes'].isin(selected_months)]

# --- GRÁFICOS (Com cores do código antigo) ---
def plot_monthly_chart(df_in):
    if df_in.empty: return None
    df_m = df_in.copy().sort_values('Data')
    df_m['Periodo'] = df_m['Data'].dt.to_period('M')
    counts = df_m.groupby('Periodo').size().reset_index(name='Qtd')
    counts['MesStr'] = counts['Periodo'].apply(lambda x: f"{MONTHS_TRANSLATION.get(x.month)} de {x.year}")
    counts['Acumulado'] = counts['Qtd'].cumsum()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Cores originais: Azul #2E86AB e Vermelho #E74C3C
    fig.add_trace(go.Bar(x=counts['MesStr'], y=counts['Qtd'], name='Mensal', marker_color='#2E86AB'), secondary_y=False)
    fig.add_trace(go.Scatter(x=counts['MesStr'], y=counts['Acumulado'], name='Acumulado', mode='lines+markers', line=dict(color='#E74C3C', width=3)), secondary_y=True)
    fig.update_layout(title='Evolução das Discussões', height=500)
    return fig

def gerar_wordcloud(texto):
    stopwords_pt = set(stopwords.words('portuguese'))
    # Stopwords customizadas do código antigo
    stopwords_custom = stopwords_pt.union(STOPWORDS).union({"qual", "como", "manejo", "conduta", "sobre", "pode", "deve", "fazer", "paciente"})
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = [t for t in tokenizer.tokenize(texto.lower()) if t not in stopwords_custom and len(t) >= 3]
    texto_limpo = " ".join(tokens)
    if not texto_limpo.strip(): return None
    # Estilo do código antigo (fundo branco, colormap magma)
    wc = WordCloud(width=1200, height=500, background_color='white', colormap='magma', max_words=100).generate(texto_limpo)
    fig, ax = plt.subplots(figsize=(12,6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    return fig

# --- UI ---
st.title("📚 Análise de Discussões Clínicas")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Visão Geral", "📈 Evolução", "🧠 Temas", "🔎 Detalhes"])

with tab1:
    st.header(f"Total: {len(filtered_df)}")
    # Cores automáticas por categoria como no antigo
    st.plotly_chart(px.bar(filtered_df['Modulo'].value_counts().reset_index(), x='Modulo', y='count', color='Modulo', title="Por Módulo"), use_container_width=True)
    st.plotly_chart(px.bar(filtered_df['UBS'].value_counts().reset_index(), x='UBS', y='count', color='UBS', title="Por UBS"), use_container_width=True)

with tab2:
    chart = plot_monthly_chart(filtered_df)
    if chart: st.plotly_chart(chart, use_container_width=True)

with tab3:
    st.header("Temas das Perguntas Norteadoras")
    fig_wc = gerar_wordcloud(" ".join(filtered_df['Questao'].astype(str)))
    if fig_wc: st.pyplot(fig_wc)
    else: st.info("Sem dados suficientes.")

with tab4:
    if not filtered_df.empty:
        for resident, group in filtered_df.groupby('Residente'):
            st.markdown(f"### 👩‍⚕️ {resident}")
            for _, row in group.sort_values('Data', ascending=False).iterrows():
                # Título do expander formatado como no antigo
                label = f"{row['Modulo']}: {row['Questao'][:80]}..." if row['Questao'] else row['Modulo']
                with st.expander(label):
                    dt_str = row['Data'].strftime('%d/%m/%Y') if pd.notna(row['Data']) else 'Não informada'
                    st.write(f"**Data:** {dt_str} | **Preceptor:** {row['Preceptor']} | **UBS:** {row['UBS']}")
                    
                    if row['Questao']: st.markdown(f"**Pergunta Norteadora:** {row['Questao']}")
                    if row['Situacao']: st.markdown(f"**Situação:** {row['Situacao']}")
                    if row['Referencia']: st.markdown(f"**Referência:** {row['Referencia']}")
                    if row['Encaminhamento']: st.markdown(f"**Encaminhamento:** {row['Encaminhamento']}")
            st.markdown("---")
    else:
        st.info("Nenhum dado encontrado.")
