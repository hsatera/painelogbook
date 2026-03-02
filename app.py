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

# --- Configurações Iniciais ---
MONTHS_TRANSLATION = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março",
    4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro",
    10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

nltk.download('stopwords', quiet=True)
st.set_page_config(layout="wide", page_title="Análise de Discussões Clínicas")

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

resident_metadata = {}
for ubs, anos in residentes_dict.items():
    for ano, nomes in anos.items():
        for nome in nomes:
            resident_metadata[nome] = {'UBS': ubs, 'Ano': ano}

# --- Carregamento de Dados (Blindado contra KeyError) ---
GSHEETS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRaHBifgKX5-0Bi4DIBVRJMz2jcLdfLmBg4uvgWXZXwb5ziT6B_OwM7x3oofJWHoUdZQnxrbHHt9YUu/pub?output=csv" 

@st.cache_data(ttl=600)
def load_data(url, resident_metadata):
    try:
        df = pd.read_csv(url, quoting=csv.QUOTE_MINIMAL)
    except:
        st.error("Erro ao carregar dados.")
        st.stop()
    
    # Mapeamento exato das colunas do formulário
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
    
    # Renomeia as que existem
    df.rename(columns={old: new for old, new in column_mapping.items() if old in df.columns}, inplace=True)

    # GARANTIA: Se a coluna não existir após o rename, cria ela vazia para evitar KeyError
    colunas_obrigatorias = ['Data', 'Preceptor', 'Residente', 'UBS', 'Situacao', 'Questao', 'Modulo', 'Referencia', 'Encaminhamento']
    for col in colunas_obrigatorias:
        if col not in df.columns:
            df[col] = ""

    # Tratamento de Data
    df['Data'] = df['Data'].astype(str).str.strip()
    df['Data'] = df['Data'].str.replace(r'(\d+)\/(\d+)\/(\d{4})\s', r'\3-\2-\1 ', regex=True)
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')

    # Limpeza de nulos e metadados
    for col in colunas_obrigatorias:
        df[col] = df[col].fillna('')

    df['AnoResidencia'] = df['Residente'].apply(lambda x: resident_metadata.get(x, {}).get('Ano', 'Não informado'))
    df['UBS'] = df.apply(lambda row: resident_metadata.get(row['Residente'], {}).get('UBS', row['UBS']), axis=1)
    df['Mes'] = df['Data'].apply(lambda x: f"{MONTHS_TRANSLATION.get(x.month)} de {x.year}" if pd.notna(x) else 'Não informado')

    return df

df = load_data(GSHEETS_URL, resident_metadata)

# --- Filtros ---
st.sidebar.header("⚙️ Filtros")
residents_from_dict = sorted(list(resident_metadata.keys()))
selected_residents = st.sidebar.multiselect("Residente", residents_from_dict)
selected_ubs = st.sidebar.multiselect("UBS", sorted(list(residentes_dict.keys())))
selected_modulos = st.sidebar.multiselect("Módulo", sorted(df['Modulo'].unique().tolist()))
selected_months = st.sidebar.multiselect("Mês", df.sort_values('Data')['Mes'].unique().tolist())

filtered_df = df[df['Residente'].isin(residents_from_dict)]
if selected_residents: filtered_df = filtered_df[filtered_df['Residente'].isin(selected_residents)]
if selected_ubs: filtered_df = filtered_df[filtered_df['UBS'].isin(selected_ubs)]
if selected_modulos: filtered_df = filtered_df[filtered_df['Modulo'].isin(selected_modulos)]
if selected_months: filtered_df = filtered_df[filtered_df['Mes'].isin(selected_months)]

# --- Visualizações ---
st.title("📚 Análise de Discussões Clínicas da Residência")
st.header(f"Total de Discussões: {len(filtered_df)}")

# Gráfico Mensal e Acumulado
def plot_monthly_chart(df_in):
    if df_in.empty: return None
    df_m = df_in.copy().sort_values('Data')
    df_m['Periodo'] = df_m['Data'].dt.to_period('M')
    counts = df_m.groupby('Periodo').size().reset_index(name='Qtd')
    counts['MesStr'] = counts['Periodo'].apply(lambda x: f"{MONTHS_TRANSLATION.get(x.month)} de {x.year}")
    counts['Acumulado'] = counts['Qtd'].cumsum()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=counts['MesStr'], y=counts['Qtd'], name='Mensal'), secondary_y=False)
    fig.add_trace(go.Scatter(x=counts['MesStr'], y=counts['Acumulado'], name='Acumulado', mode='lines+markers'), secondary_y=True)
    fig.update_layout(title='Evolução das Discussões', height=400)
    return fig

st.plotly_chart(plot_monthly_chart(filtered_df), use_container_width=True)

# Outros Gráficos
c1, c2 = st.columns(2)
with c1: st.plotly_chart(px.bar(filtered_df['Modulo'].value_counts().reset_index(), x='Modulo', y='count', title="Por Módulo"), use_container_width=True)
with c2: st.plotly_chart(px.bar(filtered_df['UBS'].value_counts().reset_index(), x='UBS', y='count', title="Por UBS"), use_container_width=True)

# --- Detalhes (Layout Original) ---
st.header("🔎 Detalhe das Discussões")
if not filtered_df.empty:
    for resident, group in filtered_df.groupby('Residente', sort=True):
        st.markdown("---")
        st.markdown(f"### 👩‍⚕️ Residente: {resident}")
        for _, row in group.sort_values('Data', ascending=False).iterrows():
            # Exibe Módulo e Questão como título
            titulo = f"**{row['Modulo']}**: {row['Questao']}" if row['Questao'] else f"**{row['Modulo']}**"
            st.markdown(titulo)
            
            with st.expander("Ver resumo completo"):
                st.markdown(f"**Data:** {row['Data'].strftime('%d/%m/%Y') if pd.notna(row['Data']) else 'Não informada'}")
                if row['Preceptor']: st.markdown(f"**Preceptor:** {row['Preceptor']}")
                if row['UBS']: st.markdown(f"**UBS:** {row['UBS']}")
                if row['Situacao']: st.markdown(f"**Situação:** {row['Situacao']}")
                if row['Referencia']: st.markdown(f"**Referência:** {row['Referencia']}")
                if row['Encaminhamento']: st.markdown(f"**Encaminhamento:** {row['Encaminhamento']}")
else:
    st.info("Nenhum dado encontrado.")
