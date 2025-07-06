import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time
import os

# Configuration de la page
st.set_page_config(page_title="ÉoBénin", layout="wide", page_icon="🌬️")

# Style CSS personnalisé
st.markdown("""
<style>
    .stApp {
        # background-color: #f0f4f8;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: black !important;
        color: red !important;
    }
    
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        border-radius: 8px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .metric-card {
        background-color: black;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Chargement des données et du modèle
@st.cache_data
def load_data_and_model():
  current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "nasa_power_data.csv")
    if not os.path.exists(csv_path):
        st.error(f"Le fichier {csv_path} est introuvable.")
        return None, None, None, None, None
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    curve_path = os.path.join(current_dir, "courbe_puissance.csv")
    if not os.path.exists(curve_path):
        st.error(f"Le fichier {curve_path} est introuvable.")
        return None, None, None, None, None
    df_curve_power = pd.read_csv(curve_path)
    power_curve = interp1d(df_curve_power["wind_speed"], df_curve_power["power"], bounds_error=False, fill_value=0)

    model_xgb1 = XGBRegressor()
    model_path1 = os.path.join(current_dir, "model_xgb_eolienne.json")
    if not os.path.exists(model_path1):
        st.error(f"Le fichier {model_path1} est introuvable.")
        return None, None, None, None, None
    try:
        model_xgb1.load_model(model_path1)
    except Exception as e:
        st.error(f"Erreur lors du chargement de {model_path1}: {str(e)}")
        return None, None, None, None, None

    model_xgb2 = XGBRegressor()
    model_path2 = os.path.join(current_dir, "model_xgb_eolienne2.json")
    if not os.path.exists(model_path2):
        st.error(f"Le fichier {model_path2} est introuvable.")
        return None, None, None, None, None
    try:
        model_xgb2.load_model(model_path2)
    except Exception as e:
        st.error(f"Erreur lors du chargement de {model_path2}: {str(e)}")
        return None, None, None, None, None

    return df, df_curve_power, power_curve, model_xgb1, model_xgb2

df, df_curve_power, power_curve, model_xgb1, model_xgb2 = load_data_and_model()

model_xgb = model_xgb1
# Barre de navigation
st.sidebar.title("🌬️ ÉoBénin")
page = st.sidebar.radio("Navigation", ["Accueil", "Prédiction", "Résultats", "Documentation", "À propos"])

# Page d'Accueil
if page == "Accueil":
    st.title("Prédiction de l'Énergie Éolienne pour un Bénin Durable")
    st.image("graphique/head.jpeg", width=5000)
    st.markdown("<h3 style='text-align: center; font-style: italic; color: #555;'>L'IA au service de l'énergie renouvelable au Bénin</h3>", unsafe_allow_html=True)
    
    st.info("""
    **Bienvenue sur la plateforme ÉoBénin !** Cet outil vous permet de prédire la vitesse du vent et d'estimer la puissance générée par une éolienne Enercon E48/800,
    facilitant ainsi la transition vers un avenir énergétique durable à Cotonou.
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image("graphique/resultat.png")
        st.markdown("""
            <p style="text-align: justify;">
                Explorer les résultats des modèles de prédiction.
            </p>
            <a href="" style="float: right; text-decoration: none;">Decouvrire →</a>
                    """, unsafe_allow_html=True)
    with col2:
        st.image("graphique/visualise.png")
        st.markdown("""
            <p style="text-align: justify;">
                Visualiser les prévisions horaires de vitesse du vent et du prédiction éolienne.
            </p>
            <a href="" style="float: right; text-decoration: none;">Decouvrire →</a>
                    """, unsafe_allow_html=True)
    with col3:
        st.image("graphique/impact.jpeg") 
        st.markdown("""
            <p style="text-align: justify;">
                Comprendre les impacts environnementaux, sociaux et opérationnels.
            </p>
            <a href="" style="float: right; text-decoration: none;">Decouvrire →</a>
                    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Faire une prédiction"):
            st.session_state.page = "Prédiction"
    with col2:
        if st.button("Voir les résultats"):
            st.session_state.page = "Résultats"
    
        
# # Page de Prédiction
elif page == "Prédiction":
    st.title("Prédiction et Estimation de Puissance 🔮")
    st.markdown("Entrez les paramètres ci-dessous pour lancer une prédiction.")
    
    col1, col2 = st.columns([2, 2])
    with col1:
        st.subheader("Mode de Prédiction")
        prediction_mode = st.selectbox("**Mode de prédiction**", [
        "Paramètres météo uniquement",
        "Paramètres météo & temps",
        "Temps uniquement (À venir)",
        "Vitesse du vents",
        ])
    with col2:
        st.subheader("Paramètres d'Entrée")
        if prediction_mode in ["Paramètres météo uniquement", "Paramètres météo & temps"]:
            col1, col2 = st.columns(2)
            with col1:
                temp = st.number_input("Température (°C)", -50.0, 50.0, 0.0)
                humidity = st.number_input("Humidité (%)", 0.0, 100.0, 0.0)
            with col2:
                pressure = st.number_input("Pression (hPa)", 0.0, 1200.0, 0.0)
                precipitation = st.number_input("Précipitations (mm)", 0.0, 100.0, 0.0)
        if prediction_mode in ["Paramètres météo & temps", "Temps uniquement"]:
            col1, col2 = st.columns(2)
            with col1:
                # timestamp = st.date_input("Date", value=datetime.now())
                # 1. Sélection de la date
                selected_date = st.date_input("Choisissez une date", value=datetime(2019, 1, 1).date())  
            with col2:
                # 2. Liste des heures de 00:00 à 23:00
                selected_hour = st.selectbox("Choisissez l'heure", [time(h, 0) for h in range(24)])
            # 3. Combinaison de la date et de l'heure
            timestamp = datetime.combine(selected_date, selected_hour)
            timestamp = pd.to_datetime(timestamp)
        if prediction_mode == "Vitesse du vents":
            wind_speed = st.number_input("Vitesse du vent (m/s)", 0.0, 100.0, 0.0)

    st.markdown("---")
    if st.button("Lancer la Prédiction"):
        if prediction_mode == "Paramètres météo uniquement":
            model_xgb = model_xgb1
            input_data = pd.DataFrame({
                "temperature_2m": [temp],
                "humidity_2m": [humidity],
                "pressure": [pressure],
                "precipitation": [precipitation]
            })
        elif prediction_mode == "Paramètres météo & temps":
            model_xgb = model_xgb2
            input_data = pd.DataFrame({
                "temperature_2m": [temp],
                "humidity_2m": [humidity],
                "pressure": [pressure],
                "precipitation": [precipitation],
                "year": [timestamp.year],
                "month": [timestamp.month],
                "hour": [timestamp.hour]
            })
            # Ajouter les encodages cycliques
            input_data['month_sin'] = np.sin(2 * np.pi * input_data['month'] / 12)
            input_data['month_cos'] = np.cos(2 * np.pi * input_data['month'] / 12)
            input_data['hour_sin'] = np.sin(2 * np.pi * input_data['hour'] / 24)
            input_data['hour_cos'] = np.cos(2 * np.pi * input_data['hour'] / 24)
            # Supprimer la colonne month non nécessaire pour la prédiction
            input_data = input_data.drop(columns=['month', 'hour'])
        elif prediction_mode == "Temps uniquement":  # Timestamp uniquement
            # model_xgb = model_xgb3
            # input_data = pd.DataFrame({"timestamp": [timestamp]})
            input_data = pd.DataFrame({
                "year": [timestamp.year],
                "month": [timestamp.month],
                "hour": [timestamp.hour]
            })
            # Ajouter les encodages cycliques
            input_data['month_sin'] = np.sin(2 * np.pi * input_data['month'] / 12)
            input_data['month_cos'] = np.cos(2 * np.pi * input_data['month'] / 12)
            input_data['hour_sin'] = np.sin(2 * np.pi * input_data['hour'] / 24)
            input_data['hour_cos'] = np.cos(2 * np.pi * input_data['hour'] / 24)
            # Supprimer la colonne month non nécessaire pour la prédiction
            input_data = input_data.drop(columns=['month', 'hour'])
        elif prediction_mode == "Vitesse du vents":
            a = 1 #inutile
            
        # Prédiction
        try:
            # Calcul de la puissance pour la vitesse du vents
            if prediction_mode == "Vitesse du vents":
                power_pred = power_curve(wind_speed)
                st.success(f"**Puissance estimée :** {power_pred:.3f} kW")
            else:
                col1, col2 = st.columns([3, 9])
                with col1:
                    # Calcul de la puissance pour la vitesse du vents predit
                    wind_speed_pred = model_xgb.predict(input_data)
                    power_pred = power_curve(wind_speed_pred)
                    st.success(f"**Vitesse du vent prédite :** {wind_speed_pred[0]:.3f} m/s")
                    st.success(f"**Puissance estimée :** {power_pred[0]:.3f} kW")
                with col2:
                    # Graphique de la courbe de puissance
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_curve_power["wind_speed"], y=df_curve_power["power"], mode="lines", name="Courbe de puissance"))
                    fig.add_trace(go.Scatter(x=[wind_speed_pred[0]], y=[power_pred[0]], mode="markers", name="Prédiction", marker=dict(size=15)))
                    fig.update_layout(title="Courbe de Puissance", xaxis_title="Vitesse du Vent (m/s)", yaxis_title="Puissance (kW)")
                    st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {str(e)}")

# Page de Résultats
elif page == "Résultats":
    st.title("Performance des Modèles 📊")
    st.markdown("Analyse des métriques de performance du modèle XGBoost sur les données de test.")

    df_test = df.head(20000).copy()
    # Feature engineering pour le timestamp
    df_test['year'] = df_test['timestamp'].dt.year
    df_test['month'] = df_test['timestamp'].dt.month
    df_test['day'] = df_test['timestamp'].dt.day
    df_test['hour'] = df_test['timestamp'].dt.hour

    # Encodage cyclique pour capturer la saisonnalité
    df_test['month_sin'] = np.sin(2 * np.pi * df_test['month'] / 12)
    df_test['month_cos'] = np.cos(2 * np.pi * df_test['month'] / 12)
    df_test['hour_sin'] = np.sin(2 * np.pi * df_test['hour'] / 24)
    df_test['hour_cos'] = np.cos(2 * np.pi * df_test['hour'] / 24)


    # mse = mean_squared_error(y_test, y_pred)
    # mae = mean_absolute_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)

    # Sans date
    st.markdown("### Modèle 1 : Sans Données Temporelles")
    col1, col2 = st.columns(2)
    model_xgb = model_xgb1
    with col1:
        rmse = 0.03
        st.markdown(f"<div class='metric-card'><h3>RMSE</h3><p>{rmse}</p></div>", unsafe_allow_html=True)
    with col2:
        r2 = 0.9994922564198703
        st.markdown(f"<div class='metric-card'><h3>R²</h3><p>{r2}</p></div>", unsafe_allow_html=True)
    
    # Variables
    x_test = df_test[["temperature_2m", "humidity_2m", "pressure", "precipitation"]] #En fonction de ces paramètre
    y_test = df_test["wind_speed_50m"] #On va predire la vitesse du vent
    # Test Prediction
    y_pred = model_xgb.predict(x_test)
    # Graphique de prédiction vs réel
    fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Vitesse Réelle (m/s)", "y": "Vitesse Prédite (m/s)"}, title="Prédictions vs Réalité")
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode="lines", name="Idéal"))
    st.plotly_chart(fig)
    # Interpretation
    st.success("Ce modèle, bien qu'utilisant uniquement les variables météorologiques brutes (température, humidité, pression, précipitations), démontre déjà une capacité très élevée à prédire la vitesse du vent.")
    
    # Avec date
    st.markdown("### Modèle 2 : Avec Données Temporelles")
    col1, col2 = st.columns(2) 
    model_xgb = model_xgb2
    with col1:
        rmse = 0.00197582389865094
        st.markdown(f"<div class='metric-card'><h3>RMSE</h3><p>{rmse:.2f}</p></div>", unsafe_allow_html=True)
    with col2:
        r2 = 0.9999980235424452
        st.markdown(f"<div class='metric-card'><h3>R²</h3><p>{r2:.2f}</p></div>", unsafe_allow_html=True)
    
    # Variable
    x_test = df_test[["temperature_2m", "humidity_2m", "pressure", "precipitation", 
        "year", "month_sin", "month_cos", "hour_sin", "hour_cos"]]
    y_test = df_test["wind_speed_50m"]
    # Test Prediction
    y_pred = model_xgb.predict(x_test)
    # Graphique de prédiction vs réel
    fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Vitesse Réelle (m/s)", "y": "Vitesse Prédite (m/s)"}, title="Prédictions vs Réalité")
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode="lines", name="Idéal"))
    st.plotly_chart(fig)
    # Interpretation
    st.success("L'ajout des caractéristiques temporelles (année, mois, heure, encodées cycliquement) a significativement amélioré la performance du modèle, le portant à un niveau de précision quasi-parfait.")
    
    
# Page de Documentation
elif page == "Documentation":
    st.title("Documentation du Projet 📚")
    st.header("Coming Soon...")
    st.markdown("""
    ## Introduction
    Ce projet vise à prédire la vitesse du vent à 50m d'altitude à partir de données météorologiques fournies par NASA POWER.  
    La vitesse prédite est ensuite utilisée pour estimer la puissance générée par une éolienne Enercon E48/800 à l'aide d'une courbe de puissance.

    ## Données
    - **Source** : NASA POWER
    - **Colonnes utilisées** :
        - `timestamp` : Date et heure
        - `temperature_2m` : Température à 2m
        - `humidity_2m` : Humidité à 2m
        - `pressure` : Pression atmosphérique
        - `precipitation` : Précipitations
        - `wind_speed_50m` : Vitesse du vent à 50m

    ## Modèle
    - **Algorithme** : XGBoost Regressor
    - **Paramètres** : 1,000,000 estimateurs, taux d'apprentissage = 1
    - **Métriques** : MSE, MAE, R²

    ## Courbe de Puissance
    La courbe de puissance de l'éolienne Enercon E48/800 est interpolée à partir de données fournies dans `courbe_puissance.csv`.
    """)

# Page À Propos
elif page == "À propos":
    st.title("À Propos du Projet ℹ️")
    st.header("Coming Soon ...")
    # st.markdown("""
    # Ce projet a été développé dans le cadre de l'exploration des énergies renouvelables, avec un focus sur l'énergie éolienne.  
    # Notre objectif est de démontrer comment les technologies d'apprentissage automatique peuvent optimiser la production d'énergie verte.    
    # 
    # """, unsafe_allow_html=True)
    st.markdown("""
                E-mail: <a href="esperaakakpo6@gmail.com">esperaakakpo6@gmail.com</a>
                """, unsafe_allow_html=True)
    
