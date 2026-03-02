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
import calendar

# Dicionário para traduzir os nomes dos meses para português
MONTHS_TRANSLATION = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março",
    4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro",
    10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

# Baixar recursos do NLTK se necessário.
nltk.download('stopwords', quiet=True)

# --- Configuração da Página Streamlit ---
st.set_page_config(layout="wide", page_title="Análise de Discussões Clínicas")

st.title("📚 Análise de Discussões Clínicas da Residência")
st.markdown("Filtro aplicado: Dados a partir de **Março de 2026**.")

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

# Inverte o dicionário para facilitar a busca por residente
resident_metadata = {}
for ubs, anos in residentes_dict.items():
    for ano, nomes in anos.items():
        for nome in nomes:
            resident_metadata[nome] = {'UBS': ubs, 'Ano': ano}

# --- Carregamento e Pré-processamento ---
GSHEETS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRaHBifgKX5-0Bi4DIBVRJMz2jcLdfLmBg4uvgWXZXwb5ziT6B_OwM7x3oofJWHoUdZQnxrbHHt9YUu/pub?output=csv" 

@st.cache_data(ttl=600)
def load_data(url, resident_metadata):
    try:
        df = pd.read_csv(url, quoting=csv.QUOTE_MINIMAL)
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
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

    # Conversão de Data
    df['Data'] = df['Data'].astype(str).str.strip()
    df['Data'] = df['Data'].str.replace(r'(\d+)\/(\d+)\/(\d{4})\s', r'\3-\2-\1 ', regex=True)
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')

    # --- FILTRO DE DATA: A partir de Março de 2026 ---
    data_corte = pd.Timestamp(2026, 3, 1)
    df = df[df['Data'] >= data_corte]

    # Preenchimento de Nulos
    for col in ['Preceptor', 'Residente', 'UBS', 'Modulo', 'Situacao', 'Questao', 'Referencia', 'Encaminhamento']:
        if col in df.columns:
            df[col] = df[col].fillna('')

    # Metadados
    df['AnoResidencia'] = df['Residente'].apply(lambda x: resident_metadata.get(x, {}).get('Ano', 'Não informado'))
    df['UBS'] = df.apply(lambda row: resident_metadata.get(row['Residente'], {}).get('UBS', row['UBS']), axis=1)
    
    df['Mes'] = df['Data'].apply(
        lambda x: f"{MONTHS_TRANSLATION.get(x.month)} de {x.year}" if pd.notna(x) else 'Não informado'
    )

    return df

df = load_data(GSHEETS_URL, resident_metadata)

# --- Barra Lateral de Filtros ---
st.sidebar.header("⚙️ Filtros")

# Filtra apenas residentes que estão no dicionário novo
residents_from_dict = sorted(list(resident_metadata.keys()))
all_ubs = sorted(list(residentes_dict.keys()))
all_residency_years = ["R1", "R2"]
all_modulos = sorted(df['Modulo'].unique().tolist()) if not df.empty else []

# Ordenação Cronológica de Meses
if not df.empty:
    sorted_months_df = df.sort_values('Data')
    all_months = sorted_months_df['Mes'].unique().tolist()
else:
    all_months = []

selected_residents = st.sidebar.multiselect("Residente", residents_from_dict)
selected_ubs = st.sidebar.multiselect("UBS", all_ubs)
selected_modulos = st.sidebar.multiselect("Módulo", all_modulos)
selected_months = st.sidebar.multiselect("Mês", all_months)
selected_residency_years = st.sidebar.multiselect("Ano de Residência", all_residency_years)

# --- Aplicação dos Filtros ---
filtered_df = df[df['Residente'].isin(residents_from_dict)]

if selected_residents:
    filtered_df = filtered_df[filtered_df['Residente'].isin(selected_residents)]
if selected_ubs:
    filtered_df = filtered_df[filtered_df['UBS'].isin(selected_ubs)]
if selected_modulos:
    filtered_df = filtered_df[filtered_df['Modulo'].isin(selected_modulos)]
if selected_months:
    filtered_df = filtered_df[filtered_df['Mes'].isin(selected_months)]
if selected_residency_years:
    filtered_df = filtered_df[filtered_df['AnoResidencia'].isin(selected_residency_years)]

# --- Visualizações ---
# (O restante do código de plotagem de gráficos e WordCloud permanece o mesmo do seu original)
st.header("📊 Resumo das Discussões (Desde Março/2026)")
st.subheader(f"Total de Discussões Encontradas: {len(filtered_df)}")

# [Inserir aqui as funções plot_bar_chart, plot_monthly_chart e geração de WordCloud do código original]
