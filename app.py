import streamlit as st
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import csv

# --- Configurações Iniciais ---
MONTHS_TRANSLATION = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março",
    4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro",
    10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

nltk.download('stopwords', quiet=True)
st.set_page_config(layout="wide", page_title="Dashboard Clínico Residência", page_icon="📚")

# --- Estilização customizada ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    div[data-testid="stExpander"] { border: none !important; box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important; background: white; }
    </style>
    """, unsafe_allow_html=True)

# --- Dicionário de Residentes ---
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

# --- Carregamento e Tratamento ---
GSHEETS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRaHBifgKX5-0Bi4DIBVRJMz2jcLdfLmBg4uvgWXZXwb5ziT6B_OwM7x3oofJWHoUdZQnxrbHHt9YUu/pub?output=csv" 

@st.cache_data(ttl=600)
def load_data(url, resident_metadata):
    try:
        df = pd.read_csv(url, quoting=csv.QUOTE_MINIMAL)
    except:
        st.error("Erro ao conectar com a planilha."); st.stop()
    
    mapping = {
        'Carimbo de data/hora': 'Data', 'Docente/Tutor/preceptor': 'Preceptor',
        'Residente envolvido(a)': 'Residente', 'UBS': 'UBS',
        'Situação/caso discutida/o (sem identificação)': 'Situacao',
        'Pergunta norteadora da discussão': 'Questao',
        'Principal Módulo do GT Teórico relacionado à discussão': 'Modulo',
        'Referência(s) utilizadas/sugeridas (com link)': 'Referencia',
        'Pactuações/encaminhamentos/feedback realizado/procedimento realizado': 'Encaminhamento'
    }
    df.rename(columns={k: v for k, v in mapping.items() if k in df.columns}, inplace=True)
    
    # Filtro imediato por residentes do dicionário
    df = df[df['Residente'].isin(resident_metadata.keys())].copy()
    
    df['Data'] = pd.to_datetime(df['Data'].astype(str).str.replace(r'(\d+)\/(\d+)\/(\d{4}).*', r'\3-\2-\1', regex=True), errors='coerce')
    df['AnoResidencia'] = df['Residente'].map(lambda x: resident_metadata.get(x, {}).get('Ano', 'N/I'))
    df['Mes'] = df['Data'].apply(lambda x: f"{MONTHS_TRANSLATION.get(x.month)} / {x.year}" if pd.notna(x) else 'N/I')
    return df.fillna('')

df = load_data(GSHEETS_URL, resident_metadata)

# --- Sidebar ---
st.sidebar.title("🎨 Painel de Controle")
st.sidebar.markdown("---")
sel_res = st.sidebar.multiselect("👩‍⚕️ Residente", sorted(df['Residente'].unique()))
sel_ubs = st.sidebar.multiselect("🏥 UBS", sorted(df['UBS'].unique()))
sel_mod = st.sidebar.multiselect("📘 Módulo", sorted(df['Modulo'].unique()))

filtered_df = df.copy()
if sel_res: filtered_df = filtered_df[filtered_df['Residente'].isin(sel_res)]
if sel_ubs: filtered_df = filtered_df[filtered_df['UBS'].isin(sel_ubs)]
if sel_mod: filtered_df = filtered_df[filtered_df['Modulo'].isin(sel_mod)]

# --- Layout Principal ---
st.title("📊 Análise de Discussões Clínicas")
st.markdown(f"Exibindo **{len(filtered_df)}** registros baseados nos filtros selecionados.")

# --- Metricas e Nuvem ---
col_stats, col_cloud = st.columns([1, 1.5])

with col_stats:
    st.subheader("Resumo Geral")
    st.metric("Total de Discussões", len(filtered_df))
    
    # Gráfico de Rosca Colorido
    fig_pie = px.pie(filtered_df, names='Modulo', hole=0.4, 
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_pie.update_layout(showlegend=False, height=300, margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_cloud:
    st.subheader("☁️ Temas mais comuns (Situações)")
    if not filtered_df.empty and filtered_df['Situacao'].str.cat().strip():
        text = " ".join(filtered_df['Situacao'].astype(str))
        stops = set(STOPWORDS).union(set(stopwords.words('portuguese'))).union({'paciente', 'quadro', 'caso', 'discutido', 'apresenta'})
        
        # Gerar nuvem em formato circular (mask aproximada pelo design)
        wc = WordCloud(background_color="white", width=800, height=400, 
                       colormap='viridis', stopwords=stops, border_color='white').generate(text)
        
        fig_wc, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_wc)
    else:
        st.info("Sem dados suficientes para gerar a nuvem.")

# --- Seção Opcional de Evolução Temporal ---
with st.expander("📈 Visualizar Evolução Temporal (Clique para abrir)"):
    df_temp = filtered_df.sort_values('Data')
    if not df_temp.empty:
        df_temp['Contagem'] = 1
        resumo_mes = df_temp.groupby('Mes', sort=False)['Contagem'].sum().reset_index()
        fig_temp = px.line(resumo_mes, x='Mes', y='Contagem', markers=True, 
                           title="Volume de Discussões por Mês",
                           color_discrete_sequence=['#FF4B4B'])
        fig_temp.update_layout(height=400, xaxis_title="", yaxis_title="Quantidade")
        st.plotly_chart(fig_temp, use_container_width=True)

# --- Gráficos de Barras Coloridos ---
st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    fig_ubs = px.bar(filtered_df['UBS'].value_counts().reset_index(), 
                     x='UBS', y='count', color='UBS',
                     title="Discussões por Unidade",
                     color_discrete_sequence=px.colors.qualitative.Bold)
    fig_ubs.update_layout(showlegend=False, height=450)
    st.plotly_chart(fig_ubs, use_container_width=True)

with c2:
    fig_ano = px.bar(filtered_df['AnoResidencia'].value_counts().reset_index(), 
                     x='AnoResidencia', y='count', color='AnoResidencia',
                     title="Distribuição R1 vs R2",
                     color_discrete_sequence=['#636EFA', '#EF553B'])
    fig_ano.update_layout(showlegend=False, height=450)
    st.plotly_chart(fig_ano, use_container_width=True)

# --- Detalhes (Lista) ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.header("🔎 Detalhamento das Consultas")
if not filtered_df.empty:
    for resident, group in filtered_df.groupby('Residente'):
        with st.container():
            st.markdown(f"### 👩‍⚕️ {resident}")
            for _, row in group.sort_values('Data', ascending=False).iterrows():
                with st.expander(f"📌 {row['Modulo']} - {row['Data'].strftime('%d/%m/%Y') if pd.notna(row['Data']) else 'Data N/I'}"):
                    st.write(f"**Questão:** {row['Questao']}")
                    st.info(f"**Situação:** {row['Situacao']}")
                    st.success(f"**Conduta/Encaminhamento:** {row['Encaminhamento']}")
                    if row['Referencia']: st.caption(f"📚 {row['Referencia']}")
else:
    st.warning("Selecione os filtros na lateral para visualizar os dados.")
