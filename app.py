# Corriger le bloc de génération de code : éviter les f-strings à l'intérieur du formatage
# Utiliser des guillemets triples et .format() pour insérer les noms de fichiers

dashboard_code = """
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# --- Chargement des données ---
df = pd.read_excel("tests_par_semaine_antibiotiques_2024.xlsx")
df = df[df["Semaine"].apply(lambda x: str(x).isdigit())].copy()
df["Semaine"] = df["Semaine"].astype(int)

# --- Détection des colonnes de % de résistance ---
df.columns = [col.strip() for col in df.columns]
percentage_cols = [col for col in df.columns if col.startswith('%') or ' %' in col]

st.title("📊 Dashboard - Résistance aux antibiotiques 2024")

# --- Sélecteur des antibiotiques ---
selected_antibiotics = st.multiselect(
    "🧪 Sélectionner les antibiotiques à afficher",
    options=percentage_cols,
    default=percentage_cols
)

# --- Slider pour plage de semaines ---
min_week, max_week = df["Semaine"].min(), df["Semaine"].max()
week_range = st.slider("📆 Plage de semaines", min_week, max_week, (min_week, max_week))

# --- Filtrage des données ---
filtered_df = df[(df["Semaine"] >= week_range[0]) & (df["Semaine"] <= week_range[1])]

# --- Création du graphique ---
fig = go.Figure()

for ab in selected_antibiotics:
    data = filtered_df[["Semaine", ab]].copy()
    data[ab] = pd.to_numeric(data[ab], errors='coerce')

    q1 = np.percentile(data[ab].dropna(), 25)
    q3 = np.percentile(data[ab].dropna(), 75)
    iqr = q3 - q1
    lower = max(q1 - 1.5 * iqr, 0)
    upper = q3 + 1.5 * iqr

    fig.add_trace(go.Scatter(x=data["Semaine"], y=data[ab],
                             mode='lines+markers',
                             name=ab))
    fig.add_trace(go.Scatter(x=data["Semaine"], y=[upper]*len(data),
                             mode='lines', name=ab + " seuil haut",
                             line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=data["Semaine"], y=[lower]*len(data),
                             mode='lines', name=ab + " seuil bas",
                             line=dict(dash='dot')))

fig.update_layout(
    title="Évolution des % de résistance (avec seuils Tukey)",
    xaxis_title="Semaine",
    yaxis_title="Résistance (%)",
    yaxis=dict(range=[0, 40]),
    hovermode="x unified"
)

# --- Affichage ---
st.plotly_chart(fig, use_container_width=True)
"""

# Enregistrer le code dans un fichier Python
dashboard_path = "/mnt/data/app_streamlit_2024.py"
with open(dashboard_path, "w") as f:
    f.write(dashboard_code)

dashboard_path
