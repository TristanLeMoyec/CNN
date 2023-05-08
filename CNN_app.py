import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page
from keras.utils.vis_utils import model_to_dot
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import cv2

st.set_page_config(
    page_title="CNN_app",
    page_icon="fox_face",
)

@st.cache_data
def import_data(filename):
    df = pd.read_csv(filename)
    return df

@st.cache_resource
def import_model(filenamemodel):
    loaded_model = pickle.load(open(filenamemodel, 'rb'))
    return loaded_model

df_test =import_data("test.csv")
model = import_model("model.pickle")
with st.container():

    with st.container():
        selected = option_menu(
            menu_title=None,
            options=["Accueil", "Prédictions tests", "Prédictions manuelles"],
            icons=['house', 'display', "pencil-square"],
            menu_icon="cast",
            orientation="horizontal",
            styles={
                "nav-link": {
                    "text-align": "left",
                    "--hover-color": "#076aff",
                },

                "nav-link-selected": {"background-color": "##076aff"},


            }
        )

        if selected == "Accueil":
            st.title('Bienvenue !')
            st.header('Ceci est un réseau de neurones convolutionnels permettant reconnaissance de chiffres :')
            model = import_model("model.pickle")
            model_graph = model_to_dot(model,show_layer_names=True, show_layer_activations= True)
            model_graph = str(model_graph)
            st.graphviz_chart(model_graph, use_container_width=True)

        if selected == "Prédictions tests":
            st.title(' ')


            def make_pred():
                row_test = df_test.sample(1)
                arr = row_test.values.reshape(28, 28)
                pred = model.predict(row_test.values.reshape(1, 28, 28, 1)).argmax()
                return row_test,arr,pred



            row_test,arr,pred= make_pred()

            col1, col2, col3 = st.columns([0.5,0.2,1])



            with col1:
                st.title('Le chiffre')
                st.image(arr,use_column_width = "always")

            with col3:
                st.title(f'Le modèle prédit {pred}')
                col4,col5=st.columns(2)

                with col4 :
                    st.button('Correct')


                with col5 :
                    st.button('Incorrect')

        if selected == "Prédictions manuelles":

                    col1,col2,col3 = st.columns([1,0.4,0.8])


                    def pred():
                        return model.predict(processed_img_array.reshape(1, 28, 28, 1)).argmax()

                    def reset_canvas():
                        return st_canvas(
                        height=280,
                        width=280,
                        background_color = "#000000",
                        stroke_color = "#FFFFFF",
                        initial_drawing= None
                                        )

                    with col1:
                        st.title("Canvas")

                        img_resized = Image.fromarray(reset_canvas().image_data.astype('uint8')).resize((28, 28))
                        # Convert the image to grayscale
                        img_gray = img_resized.convert('L')
                        # Convertir l'image en array numpy
                        img_array = np.array(img_gray)
                        # Traiter l'image comme nécessaire (ex: la normaliser)
                        processed_img_array = img_array / 255.0
                        # Stocker l'image dans une variable
                        image = np.expand_dims(processed_img_array, axis=0)

                    with col3:
                        st.title(f'Le modèle prédit {pred()}')
                        col22,col23=st.columns(2)

                        with col22 :
                            st.button('Correct')

                        with col23 :
                            st.button('Incorrect')

                    def reset_canvas():
                            return st_canvas(
                            height=280,
                            width=280,
                            background_color = "#000000",
                            stroke_color = "#FFFFFF",
                            initial_drawing= None
                        )
