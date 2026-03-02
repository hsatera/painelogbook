import streamlit as st
import pandas as pd
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import csv

# --- Configuração Inicial ---
MONTHS_TRANSLATION = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março",
    4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro",
    10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

nltk.download('stopwords', quiet=True)

st.set_page_config(layout="wide", page_title="Análise de Discussões Clínicas")
st.title("📚 Análise de Discussões Clínicas da Residência")
st.markdown("Exibindo **todas as discussões históricas** para os residentes listados no dicionário atualizado.")

# --- Dicionário de Residentes Atualizado ---
residentes_dict = {
    'Anchieta': {
        'R1': ['Gabriella Rodrigues', 'Ana Clara Magalhães'],
        'R2': ['Clarissa Goulardins', 'Paulo Okuda']
    },
    '31 de Março': {
        'R1': ['Mariana Ribeiro', 'Larissa Eri']
    },
    'Boa Vista': {
        'R1': ['Nathália Carmelo', 'André Binas'],
        'R2': ['Ana Cecília Venâncio', 'Armindo Albuquerque']
    },
    'Eulina': {
        'R1': ['Sarah Ribeiro', 'Raquel Rios'],
        'R2': ['Bárbara Nakamuta', 'Helena Lopes de Barros']
    },
    'Rosália': {
        'R1': ['Kin Shimabukuro'],
        'R2': ['Alice Haddad', 'Letícia Dantas']
    },
    'San Martin': {
        'R1': ['Julia Bicas'],
        'R2': ['José Camargo Júnior', 'Carolina Almeida']
    },
    'Santa Bárbara': {
        'R1': ['Felipe Fedrizzi', 'Julia Marcato'],
        'R2': ['Marianna Freitas']
    },
    'São Marcos': {
        'R1': ['Ludmila Vilela', 'Clara Guimarães'],
        'R2': ['Guilherme Bonelli', 'Mariana de Oliveira']
    },
    'Village': {
        'R1': ['Sarah Dariva'],
        'R2': ['Mathias Machado', 'Murilo Castro']
    }
}

# Mapeamento reverso para busca rápida
resident_metadata = {}
for ubs, anos in residentes_dict.items():
    for ano, nomes in anos.items():
        for nome in nomes:
            resident_metadata[nome] = {'UBS': ubs, 'Ano': ano}

# --- Carregamento de Dados ---
GSHEETS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRaHBifgKX5-0Bi4DIBVRJMz2jcLdfLmBg4uvgWXZXwb5ziT6B_OwM7x3oofJWHoUdZQnxrbHHt9YUu/pub?output=csv" 

@st.cache_data(ttl=600)
def load_data(url, resident_metadata):
    try:
        df = pd.read_csv(url, quoting=csv.QUOTE_MINIMAL)
    except Exception as e:
        st.error("Erro ao carregar os dados. Verifique a conexão.")
        st.stop()
    
    column_mapping = {
        'Carimbo de data/hora': 'Data',
        'Docente/Tutor/preceptor': 'Preceptor',
        'Residente envolvido(a)': 'Residente',
        'UBS': 'UBS',
        'Situação/caso discutida/o (sem identificação)': 'Situacao',
        'Pergunta norteadora da discussão': 'Questao',
        'Principal Módulo do GT Teórico relacionado à discussão': 'Modulo',
        'Referência(s) utilizadas/sugeridas (com link)': 'Referencia',
        'Pactuações/encaminhamentos/feedback realizado/procedimento realizado': 'Encaminhamento'
    }
    
    df.rename(columns={old: new for old, new in column_mapping.items() if old in df.columns}, inplace=True)

    # Tratamento de Data (Lida com formatos DD/MM/YYYY ou YYYY-MM-DD)
    df['Data'] = df['Data'].astype(str).str.strip()
    df['Data'] = df['Data'].str.replace(r'(\d+)\/(\d+)\/(\d{4})\s', r'\3-\2-\1 ', regex=True)
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')

    # Limpeza de campos nulos
    cols_to_fix = ['Preceptor', 'Residente', 'UBS', 'Modulo', 'Situacao', 'Questao', 'Referencia', 'Encaminhamento']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].fillna('')

    # Atribuição de Metadados baseada no Dicionário Novo
    df['AnoResidencia'] = df['Residente'].apply(lambda x: resident_metadata.get(x, {}).get('Ano', 'Não informado'))
    df['UBS_Ref'] = df['Residente'].apply(lambda x: resident_metadata.get(x, {}).get('UBS', 'Não informado'))
    
    # Substitui a UBS original pela do dicionário se o residente for conhecido
    df['UBS'] = df.apply(lambda row: row['UBS_Ref'] if row['UBS_Ref'] != 'Não informado' else row['UBS'], axis=1)

    df['Mes'] = df['Data'].apply(
        lambda x: f"{MONTHS_TRANSLATION.get(x.month)} de {x.year}" if pd.notna(x) else 'Não informado'
    )

    return df

df = load_data(GSHEETS_URL, resident_metadata)

# --- Filtros (Barra Lateral) ---
st.sidebar.header("⚙️ Filtros")

# Filtramos o DF para conter apenas os nomes presentes no dicionário
allowed_names = list(resident_metadata.keys())
filtered_df = df[df['Residente'].isin(allowed_names)]

# Opções dos filtros baseadas nos dados reais e dicionário
selected_residents = st.sidebar.multiselect("Residente", sorted(allowed_names))
selected_ubs = st.sidebar.multiselect("UBS", sorted(list(residentes_dict.keys())))
selected_modulos = st.sidebar.multiselect("Módulo", sorted(df['Modulo'].unique().tolist()))
selected_months = st.sidebar.multiselect("Mês/Ano", df.sort_values('Data')['Mes'].unique().tolist())

# Aplicação dinâmica
if selected_residents:
    filtered_df = filtered_df[filtered_df['Residente'].isin(selected_residents)]
if selected_ubs:
    filtered_df = filtered_df[filtered_df['UBS'].isin(selected_ubs)]
if selected_modulos:
    filtered_df = filtered_df[filtered_df['Modulo'].isin(selected_modulos)]
if selected_months:
    filtered_df = filtered_df[filtered_df['Mes'].isin(selected_months)]

# --- Dashboards ---
st.header("📊 Panorama Geral")
st.metric("Total de Discussões", len(filtered_df))

def plot_monthly_chart(data_frame):
    if data_frame.empty: return None
    df_m = data_frame.copy().sort_values('Data')
    counts = df_m.groupby('Mes', sort=False).size().reset_index(name='Qtd')
    fig = px.line(counts, x='Mes', y='Qtd', title="Volume de Discussões ao Longo do Tempo", markers=True)
    return fig

col1, col2 = st.columns(2)
with col1:
    fig_ubs = px.pie(filtered_df, names='UBS', title="Distribuição por Unidade")
    st.plotly_chart(fig_ubs, use_container_width=True)
with col2:
    fig_mod = px.bar(filtered_df['Modulo'].value_counts().reset_index(), x='Modulo', y='count', title="Discussões por Módulo")
    st.plotly_chart(fig_mod, use_container_width=True)

st.plotly_chart(plot_monthly_chart(filtered_df), use_container_width=True)

# --- Listagem Detalhada ---
st.header("🔎 Detalhe das Discussões")
for resident, group in filtered_df.groupby('Residente'):
    with st.expander(f"👩‍⚕️ {resident} ({len(group)} discussões)"):
        for _, row in group.sort_values('Data', ascending=False).iterrows():
            st.markdown(f"**{row['Data'].strftime('%d/%m/%Y') if pd.notna(row['Data']) else 'S/D'} - {row['Modulo']}**")
            st.info(f"Questão: {row['Questao']}")
            if row['Situacao']: st.write(f"*Situação:* {row['Situacao']}")
            st.divider()
