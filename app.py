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
st.markdown("Use os filtros na barra lateral para explorar as discussões do logbook e veja os resumos nos gráficos e nuvem de palavras.")

# --- Dicionário de Residentes ---
residentes_dict = {
    'Anchieta': {
        'R1': ['Clarissa Goulardins', 'Paulo Okuda'],
        'R2': ['Nathalia Rodrigues', 'Olgata SIlva']
    },
    'Barão': {
        'R1': ['Mathias Machado'],
        'R2': ['Luiz Fernando Silva', 'Rafael Augusto', 'Raquel Rocha']
    },
    'Boa Vista': {
        'R1': ['Ana Cecília Venâncio', 'Armindo Albuquerque'],
        'R2': ['Daniel Fonseca', 'Isabela Almeida']
    },
    'Eulina': {
        'R1': ['Bárbara Nakamuta', 'Helena Lopes de Barros']
    },
    'Rosália': {
        'R1': ['Alice Haddad', 'Letícia Dantas'],
        'R2': ['Luiza Kassar', 'Nathalia Braido', 'Rebeca de Barros']
    },
    'San Martin': {
        'R1': ['Carolina Almeida', 'José Camargo Júnior', 'Henrique Sater'],
        'R2': ['Natalia Bergamo']
    },
    'Santa Bárbara': {
        'R1': ['Marianna Freitas'],
        'R2': ['Débora Roveron', 'Maria Victória Vargas']
    },
    'São Marcos': {
        'R1': ['Guilherme Bonelli', 'Mariana de Oliveira', 'Mariana Macabú'],
        'R2': ['Giovana Miho', 'Laura Carvalho']
    },
    'Village': {
        'R1': ['Murilo Castro'],
        'R2': ['Ana Luisa Chen']
    }
}

# Inverte o dicionário para facilitar a busca por residente
resident_metadata = {}
for ubs, anos in residentes_dict.items():
    for ano, nomes in anos.items():
        for nome in nomes:
            resident_metadata[nome] = {'UBS': ubs, 'Ano': ano}

# --- Carregamento e Pré-processamento dos Dados do Google Sheets ---
GSHEETS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRaHBifgKX5-0Bi4DIBVRJMz2jcLdfLmBg4uvgWXZXwb5ziT6B_OwM7x3oofJWHoUdZQnxrbHHt9YUu/pub?output=csv" 

@st.cache_data(ttl=600)
def load_data(url, resident_metadata):
    """
    Carrega os dados do Google Sheets, trata as colunas e associa metadados do residente.
    """
    try:
        df = pd.read_csv(url, quoting=csv.QUOTE_MINIMAL)
    except Exception as e:
        st.error(f"Erro ao carregar os dados do Google Sheets. Verifique o URL ou as configurações de compartilhamento.")
        st.stop()
    
    df.rename(columns={
        'Carimbo de data/hora': 'Data',
        'Docente/Tutor/preceptor': 'Preceptor',
        'Residente envolvido(a)': 'Residente',
        'UBS': 'UBS',
        'Situação/caso discutida/o (sem identificação)': 'Situacao',
        'Pergunta norteadora da discussão': 'Questao',
        'Principal Módulo do GT Teórico relacionado à discussão': 'ModuloPrincipal',
        'Se desejar, marque Outros Módulos do GT teórico relacionados à situação': 'OutrosModulos',
        'Referência(s) utilizadas/sugeridas (com link)': 'Referencia',
        'Pactuações/encaminhamentos/feedback realizado/procedimento realizado': 'Encaminhamento'
    }, inplace=True)
    
    df['Data'] = df['Data'].astype(str).str.strip()
    df['Data'] = df['Data'].str.replace(r'(\d+)\/(\d+)\/(\d{4})\s', r'\3-\2-\1 ', regex=True)
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')

    df['Preceptor'] = df['Preceptor'].fillna('Não informado')
    df['Residente'] = df['Residente'].fillna('Não informado')
    df['UBS'] = df['UBS'].fillna('Não informado')
    df['ModuloPrincipal'] = df['ModuloPrincipal'].fillna('Não informado')
    df['OutrosModulos'] = df['OutrosModulos'].fillna('Não informado')
    df['Situacao'] = df['Situacao'].fillna('')
    df['Questao'] = df['Questao'].fillna('')
    df['Referencia'] = df['Referencia'].fillna('')
    df['Encaminhamento'] = df['Encaminhamento'].fillna('')
    
    # Lógica para o nome do mês
    df['Mes'] = df['Data'].apply(
        lambda x: f"{MONTHS_TRANSLATION.get(x.month)} de {x.year}" if pd.notna(x) else 'Não informado'
    )
    
    df['AnoResidencia'] = df['Residente'].apply(lambda x: resident_metadata.get(x, {}).get('Ano', 'Não informado'))
    df['UBS'] = df.apply(lambda row: resident_metadata.get(row['Residente'], {}).get('UBS', 'Não informado') if row['UBS'] == 'Não informado' else row['UBS'], axis=1)

    return df

df = load_data(GSHEETS_URL, resident_metadata)

# --- Cria a lista completa de módulos para o filtro
df_modulos_temp = df.copy()
df_modulos_temp['OutrosModulos'] = df_modulos_temp['OutrosModulos'].astype(str).str.split(',').apply(lambda x: [s.strip() for s in x])
df_modulos_temp['TodosModulos'] = df_modulos_temp.apply(
    lambda row: [row['ModuloPrincipal']] + row['OutrosModulos'] if row['ModuloPrincipal'] != 'Não informado' else row['OutrosModulos'],
    axis=1
)
df_modulos_temp = df_modulos_temp.explode('TodosModulos')
all_modulos = sorted(df_modulos_temp['TodosModulos'].unique().tolist())
if 'Não informado' in all_modulos:
    all_modulos.remove('Não informado')

# --- Barra Lateral de Filtros ---
st.sidebar.header("⚙️ Filtros")

residents_from_dict = list(resident_metadata.keys())
all_residents = sorted(residents_from_dict)
all_ubs = sorted(list(set(meta['UBS'] for meta in resident_metadata.values())))
all_residency_years = sorted(list(set(meta['Ano'] for meta in resident_metadata.values())))
all_months = sorted(df['Mes'].unique().tolist())

selected_residents = st.sidebar.multiselect("Residente", all_residents)
selected_ubs = st.sidebar.multiselect("UBS", all_ubs)
selected_modulos = st.sidebar.multiselect("Módulo", all_modulos)
selected_months = st.sidebar.multiselect("Mês", all_months)
selected_residency_years = st.sidebar.multiselect("Ano de Residência", all_residency_years)

# --- Lógica de Filtragem ---
filtered_df = df.copy()
filtered_df = filtered_df[filtered_df['Residente'].isin(residents_from_dict)]

if selected_residents:
    filtered_df = filtered_df[filtered_df['Residente'].isin(selected_residents)]
if selected_ubs:
    filtered_df = filtered_df[filtered_df['UBS'].isin(selected_ubs)]
if selected_months:
    filtered_df = filtered_df[filtered_df['Mes'].isin(selected_months)]
if selected_residency_years:
    filtered_df = filtered_df[filtered_df['AnoResidencia'].isin(selected_residency_years)]
if selected_modulos:
    # Filtra o dataframe principal para que as discussões contenham os módulos selecionados
    df_modulos_combinado = filtered_df.copy()
    df_modulos_combinado['OutrosModulos'] = df_modulos_combinado['OutrosModulos'].astype(str).str.split(',').apply(lambda x: [s.strip() for s in x])
    df_modulos_combinado['TodosModulos'] = df_modulos_combinado.apply(
        lambda row: [row['ModuloPrincipal']] + row['OutrosModulos'] if row['ModuloPrincipal'] != 'Não informado' else row['OutrosModulos'],
        axis=1
    )
    df_modulos_combinado = df_modulos_combinado.explode('TodosModulos')
    
    # Encontra os índices das discussões que contêm os módulos selecionados
    indices_com_modulo = df_modulos_combinado[df_modulos_combinado['TodosModulos'].isin(selected_modulos)].index
    filtered_df = filtered_df.loc[indices_com_modulo].drop_duplicates().copy()

# Cria o dataframe de módulos para o gráfico, a partir do df já filtrado
df_modulos_combinado = filtered_df.copy()
df_modulos_combinado['OutrosModulos'] = df_modulos_combinado['OutrosModulos'].astype(str).str.split(',').apply(lambda x: [s.strip() for s in x])
df_modulos_combinado['TodosModulos'] = df_modulos_combinado.apply(
    lambda row: [row['ModuloPrincipal']] + row['OutrosModulos'] if row['ModuloPrincipal'] != 'Não informado' else row['OutrosModulos'],
    axis=1
)
df_modulos_combinado = df_modulos_combinado.explode('TodosModulos')
df_modulos_combinado = df_modulos_combinado[df_modulos_combinado['TodosModulos'] != '']
df_modulos_combinado = df_modulos_combinado[df_modulos_combinado['TodosModulos'] != 'Não informado']


# --- Seção Principal ---
st.header("📊 Resumo das Discussões Filtradas")
st.subheader(f"Total de Discussões Encontradas: {len(filtered_df)}")

# --- Funções para Gerar Gráficos ---
def plot_bar_chart(data_frame, column, title):
    if not data_frame.empty:
        counts = data_frame[column].value_counts().reset_index()
        counts.columns = [column, 'Contagem']
        fig = px.bar(
            counts, 
            x=column, 
            y='Contagem', 
            title=title,
            labels={'Contagem': 'Número de Discussões', column: ''},
            color=column,
            color_continuous_scale=px.colors.qualitative.Plotly,
            height=300
        )
        fig.update_layout(xaxis_title_text='', yaxis_title_text='Número de Discussões')
        fig.update_traces(texttemplate='%{y}', textposition='outside')
        return fig
    return None

def plot_monthly_chart(data_frame):
    if data_frame.empty:
        return None

    df_monthly = data_frame.copy()
    
    df_monthly['DataOrdenacao'] = pd.to_datetime(df_monthly['Data'])
    
    monthly_counts = df_monthly.groupby(df_monthly['DataOrdenacao'].dt.to_period('M')).size().reset_index(name='Contagem')
    monthly_counts['Mes'] = monthly_counts['DataOrdenacao'].apply(
        lambda x: f"{MONTHS_TRANSLATION.get(x.month)} de {x.year}" if pd.notna(x) else 'Não informado'
    )
    monthly_counts['Acumulado'] = monthly_counts['Contagem'].cumsum()
    monthly_counts.sort_values(by='DataOrdenacao', inplace=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=monthly_counts['Mes'],
            y=monthly_counts['Contagem'],
            name='Discussões Mensais',
            marker_color=px.colors.qualitative.Plotly[0]
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=monthly_counts['Mes'],
            y=monthly_counts['Acumulado'],
            name='Contagem Acumulada',
            mode='lines+markers',
            marker_color=px.colors.qualitative.Plotly[1]
        ),
        secondary_y=True,
    )
    
    fig.update_layout(
        title_text='Discussões por Mês (Mensal e Acumulado)',
        height=450,
        xaxis_title_text='Mês',
        yaxis_title_text='Número de Discussões',
        yaxis2_title_text='Contagem Acumulada',
        legend=dict(x=0.01, y=0.99)
    )

    return fig

# Exibe os gráficos de distribuição, um em cima do outro.
st.header("Gráficos de Distribuição")

# Gráfico de Mês aprimorado
fig_mes = plot_monthly_chart(filtered_df)
if fig_mes:
    st.plotly_chart(fig_mes, use_container_width=True)

# Demais gráficos
chart_cols = ['UBS', 'AnoResidencia', 'Residente']
for chart_col_name in chart_cols:
    fig = plot_bar_chart(filtered_df, chart_col_name, f'Discussões por {chart_col_name}')
    if fig:
        st.plotly_chart(fig, use_container_width=True)

# Gráfico de Módulos (agora combinado)
fig_modulos = plot_bar_chart(df_modulos_combinado, 'TodosModulos', 'Discussões por Módulo')
if fig_modulos:
    st.plotly_chart(fig_modulos, use_container_width=True)


# WordCloud
st.header("☁️ Nuvem de Palavras das Situações Discutidas")
PALAVRAS_EXCLUIR = set(stopwords.words('portuguese')).union({
    'do', 'da', 'dos', 'das', 'no', 'na', 'nos', 'nas', 'um', 'uma', 'uns', 'umas',
    'pra', 'pelo', 'pela', 'pelos', 'pelas', 'etc', 'pode'
})

def filtrar_palavras(texto):
    tokenizer = RegexpTokenizer(r'\b\w+\b')
    palavras = tokenizer.tokenize(texto.lower())
    return [
        palavra for palavra in palavras
        if palavra.isalpha() and len(palavra) >= 4 and palavra not in PALAVRAS_EXCLUIR
    ]

def gerar_wordcloud(palavras_filtradas):
    if not palavras_filtradas:
        return None
    texto_filtrado = ' '.join(palavras_filtradas)
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS,
                          max_words=200, colormap='viridis', collocations=False).generate(texto_filtrado)
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=None)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig

all_situacoes_text = " ".join(filtered_df['Situacao'].dropna().tolist())
if all_situacoes_text.strip():
    palavras_filtradas = filtrar_palavras(all_situacoes_text)
    if palavras_filtradas:
        wordcloud_fig = gerar_wordcloud(palavras_filtradas)
        if wordcloud_fig:
            st.pyplot(wordcloud_fig)
            plt.close(wordcloud_fig)
        else:
            st.info("Nenhuma palavra significativa encontrada para gerar a nuvem de palavras.")
    else:
        st.info("Nenhuma palavra significativa encontrada para gerar a nuvem de palavras após a filtragem.")
else:
    st.info("Nenhuma situação discutida disponível para gerar a nuvem de palavras com os filtros selecionados.")

# --- Tabela de Dados Agrupada por Residente ---
st.header("🔎 Detalhe das Discussões por Residente")

if not filtered_df.empty:
    for resident, group in filtered_df[filtered_df['Residente'].isin(residents_from_dict)].groupby('Residente'):
        st.markdown("---")
        st.markdown(f"### 👩‍⚕️ Residente: {resident}")
        
        for _, row in group.iterrows():
            st.markdown(f"{row['ModuloPrincipal']}: {row['Questao']}")
            
            with st.expander("Ver resumo completo"):
                st.markdown(f"**Data:** {row['Data'].strftime('%d/%m/%Y') if pd.notna(row['Data']) else 'Não informada'}")
                st.markdown(f"**Preceptor:** {row['Preceptor']}")
                st.markdown(f"**UBS:** {row['UBS']}")
                st.markdown(f"**Situação:** {row['Situacao']}")
                st.markdown(f"**Módulo Principal:** {row['ModuloPrincipal']}")
                st.markdown(f"**Outros Módulos:** {row['OutrosModulos']}")
                st.markdown(f"**Referência:** {row['Referencia']}")
                st.markdown(f"**Encaminhamento:** {row['Encaminhamento']}")
else:
    st.info("Nenhuma discussão encontrada com os filtros selecionados. Ajuste os filtros na barra lateral.")
