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

# --- Configurações Iniciais e Tradução ---
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

# --- Carregamento e Processamento ---
GSHEETS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRaHBifgKX5-0Bi4DIBVRJMz2jcLdfLmBg4uvgWXZXwb5ziT6B_OwM7x3oofJWHoUdZQnxrbHHt9YUu/pub?output=csv" 

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
        'Residente envolvido(a)': 'Residente',
        'UBS': 'UBS',
        'Situação/caso discutida/o (sem identificação)': 'Situacao',
        'Pergunta norteadora da discussão': 'Questao',
        'Principal Módulo do GT Teórico relacionado à discussão': 'Modulo',
        'Referência(s) utilizadas/sugeridas (com link)': 'Referencia',
        'Pactuações/encaminhamentos/feedback realizado/procedimento realizado': 'Encaminhamento'
    }
    df.rename(columns={old: new for old, new in column_mapping.items() if old in df.columns}, inplace=True)

    # Tratamento de Data
    df['Data'] = df['Data'].astype(str).str.strip()
    df['Data'] = df['Data'].str.replace(r'(\d+)\/(\d+)\/(\d{4})\s', r'\3-\2-\1 ', regex=True)
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')

    # Preenchimento de nulos
    for col in ['Preceptor', 'Residente', 'UBS', 'Modulo', 'Situacao', 'Questao', 'Referencia', 'Encaminhamento']:
        if col in df.columns: df[col] = df[col].fillna('')

    # Metadados do Dicionário
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

# Aplicar filtros (Sempre limitando aos residentes do dicionário)
filtered_df = df[df['Residente'].isin(residents_from_dict)]
if selected_residents: filtered_df = filtered_df[filtered_df['Residente'].isin(selected_residents)]
if selected_ubs: filtered_df = filtered_df[filtered_df['UBS'].isin(selected_ubs)]
if selected_modulos: filtered_df = filtered_df[filtered_df['Modulo'].isin(selected_modulos)]
if selected_months: filtered_df = filtered_df[filtered_df['Mes'].isin(selected_months)]

# --- Cabeçalho Principal ---
st.title("📚 Análise de Discussões Clínicas da Residência")
st.header("📊 Resumo das Discussões Filtradas")
st.subheader(f"Total de Discussões Encontradas: {len(filtered_df)}")

# --- Gráfico Mensal e Acumulado ---
def plot_monthly_chart(data_frame):
    if data_frame.empty: return None
    df_m = data_frame.copy().sort_values('Data')
    df_m['Periodo'] = df_m['Data'].dt.to_period('M')
    monthly_counts = df_m.groupby('Periodo').size().reset_index(name='Contagem')
    monthly_counts['MesStr'] = monthly_counts['Periodo'].apply(lambda x: f"{MONTHS_TRANSLATION.get(x.month)} de {x.year}")
    monthly_counts['Acumulado'] = monthly_counts['Contagem'].cumsum()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=monthly_counts['MesStr'], y=monthly_counts['Contagem'], name='Mensal'), secondary_y=False)
    fig.add_trace(go.Scatter(x=monthly_counts['MesStr'], y=monthly_counts['Acumulado'], name='Acumulado', mode='lines+markers'), secondary_y=True)
    fig.update_layout(title='Evolução das Discussões', height=400)
    return fig

st.plotly_chart(plot_monthly_chart(filtered_df), use_container_width=True)

# --- Outros Gráficos ---
col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(px.bar(filtered_df['Modulo'].value_counts().reset_index(), x='Modulo', y='count', title="Por Módulo"), use_container_width=True)
with col_b:
    st.plotly_chart(px.bar(filtered_df['UBS'].value_counts().reset_index(), x='UBS', y='count', title="Por UBS"), use_container_width=True)

# --- WordCloud ---
st.header("☁️ Nuvem de Palavras (Questões)")
def gerar_wordcloud(texto):
    if not texto.strip(): return None
    sw = set(stopwords.words('portuguese')).union({'do', 'da', 'de', 'para', 'na', 'no', 'em', 'com'})
    wc = WordCloud(width=800, height=300, background_color='white', stopwords=sw).generate(texto)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    return fig

texto_questoes = " ".join(filtered_df['Questao'].tolist())
fig_wc = gerar_wordcloud(texto_questoes)
if fig_wc: st.pyplot(fig_wc)

# --- DETALHE DAS DISCUSSÕES (Layout Original Retomado) ---
st.header("🔎 Detalhe das Discussões por Residente")
if not filtered_df.empty:
    for resident, group in filtered_df.groupby('Residente', sort=True):
        st.markdown("---")
        st.markdown(f"### 👩‍⚕️ Residente: {resident}")
        
        for _, row in group.sort_values('Data', ascending=False).iterrows():
            # Título da discussão (Módulo + Questão)
            label = f"**{row['Modulo']}**: {row['Questao']}" if row['Questao'] else f"**{row['Modulo']}**"
            st.markdown(label)
            
            with st.expander("Ver resumo completo"):
                st.markdown(f"**Data:** {row['Data'].strftime('%d/%m/%Y') if pd.notna(row['Data']) else 'Não informada'}")
                
                if row['Preceptor'] and row['Preceptor'] != 'Não informado':
                    st.markdown(f"**Preceptor:** {row['Preceptor']}")
                
                if row['UBS'] and row['UBS'] != 'Não informado':
                    st.markdown(f"**UBS:** {row['UBS']}")

                if row['Situacao']:
                    st.markdown(f"**Situação:** {row['Situacao']}")

                if row['Referencia']:
                    st.markdown(f"**Referência:** {row['Referencia']}")

                if row['Encaminhamento']:
                    st.markdown(f"**Encaminhamento:** {row['Encaminhamento']}")
else:
    st.info("Nenhum dado encontrado para os filtros selecionados.")
