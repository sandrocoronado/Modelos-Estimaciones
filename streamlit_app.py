import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Inicio", page_icon="🏠")

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        "Main Menu", ["Inicio", "Modelo de Estimación"],
        icons=["house", "graph-up"], menu_icon="cast", default_index=0)

# Main page content
if selected == "Inicio":
    st.title("Bienvenido a la Aplicación de Streamlit")
    st.write("Utiliza el menú de la barra lateral para navegar a diferentes páginas.")

elif selected == "Modelo de Estimación":
    with open("Modelo Estimacion.py") as f:
        exec(f.read())

