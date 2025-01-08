import streamlit as st
import pandas as pd
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Segmentasi Pelanggan Berdasarkan Pola Pembelian")
st.write("Menggunakan Algoritma Clustering (K-Means)")

# Sidebar untuk upload file
uploaded_file = st.sidebar.file_uploader("Upload File Dataset (CSV)", type="csv")

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Awal")
    st.write(df.head())

    # Preparation
    df = df.dropna(subset=['CustomerID'])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # Recency
    reference_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    recency = df.groupby('CustomerID')['InvoiceDate'].max().reset_index()
    recency['Recency'] = (reference_date - recency['InvoiceDate']).dt.days

    # Frequency
    frequency = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    frequency.rename(columns={'InvoiceNo': 'Frequency'}, inplace=True)

    # Monetary
    monetary = df.groupby('CustomerID')['TotalPrice'].sum().reset_index()
    monetary.rename(columns={'TotalPrice': 'Monetary'}, inplace=True)

    # Gabung fitur RFM
    rfm = recency.merge(frequency, on='CustomerID').merge(monetary, on='CustomerID')

    st.subheader("Data RFM")
    st.write(rfm.head())

    # Standarisasi Data RFM
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    # Tentukan Jumlah Cluster dengan Metode Elbow
    sse = []
    for k in range(1, 11):  # mencoba 1 hingga 10 cluster
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(rfm_scaled)
        sse.append(kmeans.inertia_)

    # Plot Elbow Method
    st.subheader("Metode Elbow")
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), sse, marker='o')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('SSE')
    ax.set_title('Elbow Method')
    st.pyplot(fig)

    # Input jumlah cluster dari pengguna
    optimal_clusters = st.sidebar.slider("Pilih Jumlah Cluster (k)", 2, 10, 4)

    # Terapkan K-Means
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    st.subheader("Hasil Clustering")
    st.write(rfm.head())

    # Analisis Karakteristik Cluster
    cluster_analysis = rfm.groupby('Cluster').mean()
    st.subheader("Analisis Karakteristik Cluster")
    st.write(cluster_analysis)

    # Visualisasi Cluster
    st.subheader("Visualisasi Cluster (Frequency vs Monetary)")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=rfm['Frequency'],
        y=rfm['Monetary'],
        hue=rfm['Cluster'],
        palette='viridis',
        ax=ax
    )
    ax.set_title("Cluster Visualization (Frequency vs Monetary)")
    st.pyplot(fig)
