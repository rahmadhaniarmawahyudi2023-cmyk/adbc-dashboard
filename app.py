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

.stApp {
    background-color: #0b1120;
}

section[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid rgba(255,255,255,0.06);
}

h1, h2, h3, h4 {
    color: #e2e8f0;
}

p, label {
    color: #94a3b8;
}

.block-container {
    padding-top: 1.5rem;
}

.hero {
    background: linear-gradient(135deg, #1e293b, #312e81);
    padding: 28px;
    border-radius: 16px;
    margin-bottom: 24px;
}

.hero h1 {
    color: white;
    margin-bottom: 10px;
}

.hero p {
    color: #cbd5e1;
}

[data-testid="metric-container"] {
    background: #111827;
    border-radius: 12px;
    padding: 18px;
    border: 1px solid rgba(255,255,255,0.05);
}

[data-testid="metric-container"] label {
    color: #94a3b8 !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: white;
}

.info-box {
    background: #111827;
    padding: 16px;
    border-radius: 12px;
    border-left: 4px solid #6366f1;
    margin-bottom: 20px;
}

.info-box p {
    color: #cbd5e1;
    margin: 0;
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

        fig2 = px.pie(
            status_counts,
            names='Status',
            values='Jumlah',
            hole=0.5
        )

        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )

        st.plotly_chart(fig2, use_container_width=True)

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
