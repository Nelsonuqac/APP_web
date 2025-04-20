import streamlit as st
import pickle
import numpy as np

# ========== CONFIGURATION DE LA PAGE ==========
st.set_page_config(page_title="Prédiction de prix Airbnb", page_icon="🏠", layout="centered")

# ========== STYLE CSS PERSONNALISÉ ==========
st.markdown("""
    <style>
    .main-title {
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #6c757d;
        text-align: center;
        margin-bottom: 30px;
    }
    .model-box {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .model-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .model-result {
        font-size: 18px;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# ========== TITRE ==========
st.markdown("<div class='main-title'>🏠 Prédiction de catégorie de prix Airbnb</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Prédisez si une annonce a un <b>prix bas</b> ou <b>prix élevé</b> à l’aide de trois modèles d’apprentissage automatique.</div>", unsafe_allow_html=True)

# ========== CHARGEMENT DES MODÈLES ==========
with open("logistic_model.pkl", "rb") as f:
    logistic_model = pickle.load(f)
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ========== ENTRÉES UTILISATEUR ==========
st.sidebar.header("🎯 Caractéristiques de l'annonce Airbnb")
minimum_nights = st.sidebar.number_input("Nombre minimum de nuits", min_value=1, value=3)
number_of_reviews = st.sidebar.number_input("Nombre de commentaires", min_value=0, value=10)
reviews_per_month = st.sidebar.number_input("Commentaires par mois", min_value=0.0, value=1.2, step=0.1)

# ========== TRANSFORMATION ==========
user_input = np.array([[minimum_nights, number_of_reviews, reviews_per_month]])
user_input_scaled = scaler.transform(user_input)

# ========== PRÉDICTIONS ==========
classes = {0: "Prix bas", 1: "Prix élevé"}

prediction_log = logistic_model.predict(user_input_scaled)[0]
prediction_rf = rf_model.predict(user_input_scaled)[0]
prediction_svm = svm_model.predict(user_input_scaled)[0]

# ========== AFFICHAGE DES RÉSULTATS ==========
st.markdown("## 📊 Résultats de classification")

st.markdown(f"""
<div class='model-box'>
    <div class='model-title'>🧮 Logistic Regression</div>
    <div class='model-result'>{classes[prediction_log]}</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class='model-box'>
    <div class='model-title'>🌲 Random Forest</div>
    <div class='model-result'>{classes[prediction_rf]}</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class='model-box'>
    <div class='model-title'>🧭 SVM</div>
    <div class='model-result'>{classes[prediction_svm]}</div>
</div>
""", unsafe_allow_html=True)

# ========== AFFICHAGE DES PROBABILITÉS (OPTIONNEL) ==========
if st.checkbox("🔍 Afficher les probabilités (si disponibles)"):
    st.subheader("📈 Probabilités des modèles")
    try:
        proba_log = logistic_model.predict_proba(user_input_scaled)[0]
        st.write(f"Logistic Regression : **{round(proba_log[1] * 100, 2)}%** de probabilité d’un prix élevé")
    except:
        st.warning("Probabilités non disponibles pour Logistic Regression.")

    try:
        proba_rf = rf_model.predict_proba(user_input_scaled)[0]
        st.write(f"Random Forest : **{round(proba_rf[1] * 100, 2)}%** de probabilité d’un prix élevé")
    except:
        st.warning("Probabilités non disponibles pour Random Forest.")
