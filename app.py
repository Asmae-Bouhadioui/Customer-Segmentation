import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from fpdf import FPDF

# Configuration de la page
st.set_page_config(page_title="Segmentation des Clients", layout="wide")

st.title("📊 Segmentation des Clients avec K-Means")

# Importation du dataset
uploaded_file = st.file_uploader("📂 Importer un fichier CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Aperçu des données :")
    st.write(df.head())

    # Sélectionner les colonnes numériques
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) < 2:
        st.error("Pas assez de colonnes numériques pour afficher la matrice de corrélation.")
    else:
        # ---------------------------
        # Exploration des données
        st.sidebar.header("📌 Analyse et Pré-traitement")
        if st.sidebar.checkbox("📊 Afficher les statistiques descriptives"):
            st.write("### Statistiques descriptives")
            st.write(df.describe())

        if st.sidebar.checkbox("📉 Afficher la matrice de corrélation"):
            # Calculer la matrice de corrélation
            correlation_matrix = df[numeric_cols].corr().values
            labels = list(numeric_cols)

            # Créer une heatmap interactive
            fig = ff.create_annotated_heatmap(
                z=correlation_matrix,
                x=labels,
                y=labels,
                colorscale="Viridis",
                showscale=True
            )

            st.write("### 📊 Matrice de corrélation interactive")
            st.plotly_chart(fig)
        # ---------------------------
        # Choix du nombre de clusters
        st.sidebar.header("🔢 Paramètres du modèle")
        K = st.sidebar.slider("Choisissez le nombre de clusters (K)", min_value=2, max_value=10, value=3)

        # ---------------------------
        # Entraînement du modèle
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[numeric_cols])
        
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(df_scaled)

        df["Cluster"] = clusters

        # Sauvegarder le modèle
        joblib.dump(kmeans, "KM.pkl")

        # ---------------------------
        # Visualisation des clusters
        st.subheader("📊 Visualisation des Segments")
        if len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], color=df["Cluster"].astype(str), title="Clusters des clients")
            st.plotly_chart(fig)

        # ---------------------------
        # Génération du rapport PDF
        def generate_pdf():
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", style="B", size=16)
            pdf.cell(200, 10, "Rapport de Segmentation des Clients", ln=True, align="C")

            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, f"Nombre de clusters choisi: {K}", ln=True, align="L")

            pdf.cell(200, 10, "Statistiques descriptives:", ln=True, align="L")
            for col in numeric_cols:
                pdf.cell(200, 10, f"{col} - Moyenne: {df[col].mean():.2f}, Écart-type: {df[col].std():.2f}", ln=True)

            pdf.cell(200, 10, "Centres des clusters:", ln=True, align="L")
            centers = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_cols)
            for i, row in centers.iterrows():
                pdf.cell(200, 10, f"Cluster {i}: {row.to_dict()}", ln=True)

            pdf.output("rapport_segmentation.pdf")

        if st.button("📄 Générer un rapport PDF"):
            generate_pdf()
            st.success("📄 Rapport généré avec succès ! Téléchargez-le ci-dessous :")
            with open("rapport_segmentation.pdf", "rb") as f:
                st.download_button("📥 Télécharger le rapport", f, file_name="rapport_segmentation.pdf")
