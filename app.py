import pandas as pd
import streamlit as st
import numpy as np
import joblib

####################################### Modele 1 ##############################################################

# Charger le modèle et l'encodeur
model = joblib.load('best_model.pkl')
encoder = joblib.load('label_encoder.pkl')

# Charger les noms des caractéristiques
feature_names = joblib.load('feature_names.pkl')

# Titre de l'application
st.title('Prédiction de Prix Pour Les Produits Oriflame')

# Afficher les catégories connues par l'encodeur
known_categories = encoder.classes_

# Entrée pour le rating du produit
rating = st.slider('Note du produit', min_value=0.0, max_value=5.0, step=0.1)

# Entrée pour le nom de la marque
brand_name = st.selectbox('Nom de la marque', known_categories)

# Assurez-vous que la marque sélectionnée est dans les catégories connues
if brand_name not in known_categories:
    st.error(f'La marque {brand_name} n\'est pas reconnue. Veuillez en choisir une autre.')
else:
    # Encoder la marque avec LabelEncoder
    encoded_brand = encoder.transform([brand_name])[0]  # Cela donne un entier

    # Créer le DataFrame pour les données d'entrée avec les colonnes appropriées
    input_data = pd.DataFrame([[rating, encoded_brand]], columns=['rating', 'brand_name_encoded'])

    # Réindexer les colonnes pour correspondre à ce que le modèle attend
    input_data = input_data[['rating', 'brand_name_encoded']]



    # Faire la prédiction
    if st.button('Prédire le prix'):
        try:
            # Prédire le prix
            predicted_price = model.predict(input_data)[0]

            # Afficher le prix prédit
            st.write(f'Le prix prédit pour le produit avec une note de {rating} et de la marque {brand_name} est {predicted_price:.2f}.')
        except Exception as e:
            st.error(f'Une erreur est survenue : {e}')


####################################### Modele 2 ##############################################################


# Charger le modèle et l'encodeur
model_2 = joblib.load('best_model_2.pkl')
encoder_2 = joblib.load('label_encoder_2.pkl')

# Charger les noms des caractéristiques
feature_names_2 = joblib.load('feature_names_2.pkl')

# Titre du modéle 2
st.title('Prédiction de la note d\'évaluation d\'un produit (Oriflame)')

# Afficher les catégories connues par l'encodeur du modèle 2
known_categories_2 = encoder_2.classes_

# Entrée pour le prix du produit
product_price = st.number_input('Entrer le prix du produit', min_value=0.0, step=1.0)

# Entrée pour le nom de la marque
brand_name_2 = st.selectbox('Nom de la marque ', known_categories_2)

# Assurez-vous que la marque sélectionnée est dans les catégories connues
if brand_name_2 not in known_categories_2:
    st.error(f'La marque {brand_name_2} n\'est pas reconnue. Veuillez en choisir une autre.')
else:
    # Encoder la marque avec LabelEncoder
    encoded_brand_2 = encoder_2.transform([brand_name_2])[0]  # Cela donne un entier

    # Créer le DataFrame pour les données d'entrée avec les colonnes appropriées
    input_data_2 = pd.DataFrame([[product_price, encoded_brand_2]], columns=['product_price', 'brand_name_encoded'])

    # Réindexer les colonnes pour correspondre à ce que le modèle attend
    input_data_2 = input_data_2[['product_price', 'brand_name_encoded']]


    # Faire la prédiction
    if st.button('Prédire la note d\'évaluation'):
        try:
            # Prédire le rating
            predicted_rating = model_2.predict(input_data_2)[0]

            # Afficher le prix prédit
            st.write(f'La note prédite pour le produit avec un prix de {product_price} et de la marque {brand_name_2} est {predicted_rating:.2f}.')
        except Exception as e:
            st.error(f'Une erreur est survenue : {e}')