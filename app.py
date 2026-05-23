import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Dashboard Analisis San Francisco 311",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

/* =========================================================
BACKGROUND ANIMATION
========================================================= */

.stApp {
    background: linear-gradient(
        -45deg,
        #020617,
        #0f172a,
        #111827,
        #1e1b4b
    );
    background-size: 400% 400%;
    animation: gradientBG 18s ease infinite;
    overflow-x: hidden;
}

@keyframes gradientBG {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}

/* Floating blur circles */

.stApp::before {
    content: "";
    position: fixed;
    width: 500px;
    height: 500px;
    background: rgba(99,102,241,0.18);
    filter: blur(120px);
    border-radius: 50%;
    top: -150px;
    left: -120px;
    animation: float1 12s ease-in-out infinite;
    z-index: -1;
}

.stApp::after {
    content: "";
    position: fixed;
    width: 450px;
    height: 450px;
    background: rgba(56,189,248,0.15);
    filter: blur(120px);
    border-radius: 50%;
    bottom: -120px;
    right: -100px;
    animation: float2 15s ease-in-out infinite;
    z-index: -1;
}

@keyframes float1 {
    0% {
        transform: translateY(0px) translateX(0px);
    }
    50% {
        transform: translateY(40px) translateX(50px);
    }
    100% {
        transform: translateY(0px) translateX(0px);
    }
}

@keyframes float2 {
    0% {
        transform: translateY(0px) translateX(0px);
    }
    50% {
        transform: translateY(-50px) translateX(-30px);
    }
    100% {
        transform: translateY(0px) translateX(0px);
    }
}

/* =========================================================
SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.75);
    backdrop-filter: blur(18px);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* =========================================================
TEXT
========================================================= */

h1, h2, h3, h4 {
    color: #f8fafc;
    font-weight: 600;
}

p, label, span {
    color: #cbd5e1;
}

/* =========================================================
HERO SECTION
========================================================= */

.hero {
    background: rgba(15,23,42,0.55);
    backdrop-filter: blur(18px);
    padding: 32px;
    border-radius: 24px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 28px;
    box-shadow: 0 0 30px rgba(99,102,241,0.18);
}

.hero h1 {
    color: white;
    margin-bottom: 10px;
    font-size: 2rem;
}

.hero p {
    color: #cbd5e1;
    font-size: 15px;
}

/* =========================================================
METRIC CARD
========================================================= */

[data-testid="metric-container"] {
    background: rgba(17,24,39,0.60);
    backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.06);
    transition: 0.3s ease;
}

[data-testid="metric-container"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 0 20px rgba(99,102,241,0.25);
}

[data-testid="metric-container"] label {
    color: #94a3b8 !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: white;
}

/* =========================================================
INFO BOX
========================================================= */

.info-box {
    background: rgba(17,24,39,0.60);
    backdrop-filter: blur(14px);
    padding: 18px;
    border-radius: 18px;
    border-left: 4px solid #6366f1;
    margin-bottom: 22px;
}

.info-box p {
    color: #e2e8f0;
    margin: 0;
}

/* =========================================================
TABLES
========================================================= */

[data-testid="stDataFrame"] {
    background: rgba(17,24,39,0.55);
    border-radius: 18px;
    overflow: hidden;
}

/* =========================================================
BUTTON
========================================================= */

.stButton > button {
    background: linear-gradient(
        135deg,
        #6366f1,
        #8b5cf6
    );
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 22px;
    font-weight: 600;
    transition: 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 18px rgba(99,102,241,0.4);
}

/* =========================================================
INPUTS
========================================================= */

.stTextInput input,
.stTextArea textarea,
[data-baseweb="select"] {
    background-color: rgba(17,24,39,0.70) !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* =========================================================
SCROLLBAR
========================================================= */

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #0f172a;
}

::-webkit-scrollbar-thumb {
    background: #6366f1;
    border-radius: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():

    # =====================================================
    # PAKAI CSV EXPORT DARI COLAB
    # =====================================================
    df = pd.read_csv("san_francisco_311_clean.csv")

    return df

df = load_data()

# =========================================================
# DATA PREPARATION
# =========================================================
df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
df['closed_date'] = pd.to_datetime(df['closed_date'], errors='coerce')

df['resolution_days'] = (
    df['closed_date'] - df['created_date']
).dt.days

df['year'] = df['created_date'].dt.year
df['month'] = df['created_date'].dt.month
df['month_name'] = df['created_date'].dt.month_name()
df['day_of_week'] = df['created_date'].dt.day_name()
df['hour'] = df['created_date'].dt.hour

df = df.drop_duplicates()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("""
    # 📊 Urban Service Intelligence
    """)
    
    st.caption("""
    San Francisco 311 Analytics • BigQuery + Streamlit
    """)

    menu = st.radio(
        "Menu",
        [
            "Overview",
            "EDA & Visualisasi",
            "Heatmap",
            "Analisis Waktu Penyelesaian",
            "WordCloud",
            "Topic Modeling (LDA)",
            "Eksplorasi Data"
        ]
    )

    st.write("---")

    year_list = sorted(df['year'].dropna().unique())

    selected_year = st.selectbox(
        "Filter Tahun",
        year_list
    )

    df = df[df['year'] == selected_year]

# =========================================================
# OVERVIEW
# =========================================================
if menu == "Overview":

    st.markdown("""
    <div class="hero">
        <h1>Dashboard Analisis Layanan Publik San Francisco 311</h1>
        <p>
        Analisis data berbasis cloud menggunakan Google BigQuery,
        Google Colab, GitHub, dan Streamlit Cloud.
        </p>
    </div>
    """, unsafe_allow_html=True)

    total_reports = len(df)

    total_category = df['category'].nunique()

    avg_resolution = round(
        df['resolution_days'].dropna().mean(),
        2
    )

    closed_percent = round(
        (df['status'] == 'Closed').mean() * 100,
        2
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Laporan", f"{total_reports:,}")
    c2.metric("Jumlah Kategori", total_category)
    c3.metric("Rata-rata Penyelesaian", f"{avg_resolution} hari")
    c4.metric("Persentase Closed", f"{closed_percent}%")

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Top 10 Kategori Keluhan")

        top_cat = (
            df['category']
            .value_counts()
            .head(10)
            .reset_index()
        )

        top_cat.columns = ['Kategori', 'Jumlah']

        fig = px.bar(
            top_cat,
            x='Jumlah',
            y='Kategori',
            orientation='h',
            color='Jumlah',
            color_continuous_scale='blues'
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

    st.subheader("Distribusi Status")

    status_counts = (
        df['status']
        .value_counts()
        .reset_index()
    )

    status_counts.columns = ['Status', 'Jumlah']

    fig2 = go.Figure(
        data=[
            go.Pie(
                labels=status_counts['Status'],
                values=status_counts['Jumlah'],
                hole=0.65,
                marker=dict(
                    colors=[
                        "#6366f1",
                        "#38bdf8",
                        "#4ade80",
                        "#facc15",
                        "#f87171"
                    ],
                    line=dict(color="#0b1120", width=3)
                ),
                textinfo='percent',
                textfont=dict(size=14, color='white'),
                pull=[0.03]*len(status_counts),
            )
        ]
    )

    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        showlegend=True,
        legend=dict(
            orientation="h",
            y=-0.1
        ),
        margin=dict(t=20, b=20, l=20, r=20)
    )

    st.markdown("""
    <style>
    .glow-chart {
        animation: rotateGlow 12s linear infinite;
        border-radius: 50%;
        padding: 12px;
        box-shadow:
            0 0 20px rgba(99,102,241,0.4),
            0 0 40px rgba(56,189,248,0.25),
            0 0 60px rgba(99,102,241,0.15);
    }

    @keyframes rotateGlow {
        0% {
            transform: rotate(0deg);
        }
        100% {
            transform: rotate(360deg);
        }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glow-chart">', unsafe_allow_html=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# EDA
# =========================================================
elif menu == "EDA & Visualisasi":

    st.title("Exploratory Data Analysis")

    st.subheader("Distribusi Laporan per Bulan")

    monthly = (
        df.groupby('month_name')
        .size()
        .reset_index(name='Jumlah')
    )

    month_order = [
        'January','February','March','April',
        'May','June','July','August',
        'September','October','November','December'
    ]

    monthly['month_name'] = pd.Categorical(
        monthly['month_name'],
        categories=month_order,
        ordered=True
    )

    monthly = monthly.sort_values('month_name')

    fig = px.line(
        monthly,
        x='month_name',
        y='Jumlah',
        markers=True
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribusi Kategori")

    top8 = (
        df['category']
        .value_counts()
        .head(8)
        .reset_index()
    )

    top8.columns = ['Kategori', 'Jumlah']

    fig2 = px.bar(
        top8,
        x='Kategori',
        y='Jumlah',
        color='Jumlah'
    )

    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )

    st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# HEATMAP
# =========================================================
elif menu == "Heatmap":

    st.title("Heatmap Hari vs Kategori")

    pivot = (
        df.groupby(['day_of_week', 'category'])
        .size()
        .unstack(fill_value=0)
    )

    day_order = [
        'Monday',
        'Tuesday',
        'Wednesday',
        'Thursday',
        'Friday',
        'Saturday',
        'Sunday'
    ]

    pivot = pivot.reindex(day_order)

    top_cat = df['category'].value_counts().head(6).index

    pivot = pivot[top_cat]

    pivot = pivot.fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(
        pivot,
        annot=True,
        fmt='d',
        cmap='YlOrRd',
        linewidths=0.5,
        ax=ax
    )

    ax.set_title("Jumlah Laporan per Hari dan Kategori")

    st.pyplot(fig)

# =========================================================
# ANALISIS WAKTU PENYELESAIAN
# =========================================================
elif menu == "Analisis Waktu Penyelesaian":

    st.title("Analisis Waktu Penyelesaian")

    st.subheader("Distribusi Waktu Penyelesaian")

    fig = px.histogram(
        df,
        x='resolution_days',
        nbins=40
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rata-rata Waktu Penyelesaian per Kategori")

    avg_res = (
        df.groupby('category')['resolution_days']
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig2 = px.bar(
        avg_res,
        x='resolution_days',
        y='category',
        orientation='h',
        color='resolution_days'
    )

    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )

    st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# WORDCLOUD
# =========================================================
elif menu == "WordCloud":

    st.title("WordCloud Deskripsi Keluhan")

    text_column = 'descriptor' if 'descriptor' in df.columns else 'complaint_type'

    text = " ".join(
        df[text_column]
        .dropna()
        .astype(str)
    )

    if text.strip() == "":

        st.warning("Data descriptor kosong")

    else:

        wordcloud = WordCloud(
            width=1200,
            height=500,
            background_color='white'
        ).generate(text)

        fig, ax = plt.subplots(figsize=(15, 6))

        ax.imshow(wordcloud)

        ax.axis('off')

        st.pyplot(fig)

# =========================================================
# TOPIC MODELING
# =========================================================
elif menu == "Topic Modeling (LDA)":

    st.title("Topic Modeling (LDA)")

    st.markdown("""
    <div class="info-box">
        <p>
        LDA digunakan untuk menemukan topik utama
        berdasarkan teks descriptor laporan masyarakat.
        </p>
    </div>
    """, unsafe_allow_html=True)

    text_column = 'descriptor' if 'descriptor' in df.columns else 'complaint_type'

    text_series = (
        df[text_column]
        .dropna()
        .astype(str)
    )

    if len(text_series) == 0:

        st.warning("Data descriptor kosong")

    else:

        vectorizer = CountVectorizer(
            stop_words='english',
            max_df=0.95,
            min_df=2
        )

        dtm = vectorizer.fit_transform(text_series)

        lda = LatentDirichletAllocation(
            n_components=5,
            random_state=42
        )

        lda.fit(dtm)

        words = vectorizer.get_feature_names_out()

        for idx, topic in enumerate(lda.components_):

            st.subheader(f"Topik {idx+1}")

            top_words = [
                words[i]
                for i in topic.argsort()[-10:]
            ]

            st.write(", ".join(top_words))

# =========================================================
# DATA EXPLORATION
# =========================================================
elif menu == "Eksplorasi Data":

    st.title("Eksplorasi Dataset")

    search = st.text_input(
        "Cari kategori atau complaint"
    )

    display_df = df.copy()

    text_column = 'descriptor' if 'descriptor' in df.columns else 'complaint_type'

    if search:

        mask = (
            display_df['category']
            .astype(str)
            .str.contains(search, case=False, na=False)
        ) | (
            display_df[text_column]
            .astype(str)
            .str.contains(search, case=False, na=False)
        )

        display_df = display_df[mask]

    st.caption(
        f"{len(display_df):,} data ditampilkan"
    )

    st.dataframe(
        display_df[
            [
                'created_date',
                'closed_date',
                'status',
                'category',
                'complaint_type',
                'resolution_days'
            ]
        ],
        use_container_width=True
    )
