import streamlit as st

# Main page content
selected = st.sidebar.selectbox("Selecciona una página", ["Inicio", "Estimación Perfil", "Modelo de Estimación"])

if selected == "Inicio":
    st.title("Bienvenido a la Aplicación de Streamlit")
    st.write("Utiliza el menú de la barra lateral para navegar a diferentes páginas.")

elif selected == "Estimación Perfil":
    try:
        with open("Estimacion Perfil.py") as f:
            exec(f.read())
    except FileNotFoundError:
        st.error("El archivo 'Estimacion Perfil.py' no se encuentra en el directorio.")
    except Exception as e:
        st.error(f"Ocurrió un error al ejecutar 'Estimacion Perfil.py': {e}")

elif selected == "Modelo de Estimación":
    try:
        with open("Modelo Estimacion.py") as f:
            exec(f.read())
    except FileNotFoundError:
        st.error("El archivo 'Modelo Estimacion.py' no se encuentra en el directorio.")
    except Exception as e:
        st.error(f"Ocurrió un error al ejecutar 'Modelo Estimacion.py': {e}")


