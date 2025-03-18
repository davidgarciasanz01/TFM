import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from Retriever import get_fine_retriever
from LLM_Guia import run_guide_llm
from LLM import info_run_guide_llm  # Importa las funciones del LLM general
import os

def chat_interface():
    st.sidebar.image("images/logoMAN.png",  use_container_width =True)
    
    # Crear dos columnas
    col1, col2 = st.columns(2)

    # Imagen en la primera columna
    col1.image("images/bifaz.jpeg", caption="Bifaz del Manzanares")

    # Imagen en la segunda columna
    col2.image("images/damaElche.jpeg",  caption="Dibujo de la Dama de Elche")

    st.title("Asistente del Museo Arqueológico Nacional de Madrid")
    st.write("Hola, soy el asistente virtual del Museo, estoy especializado en las obras y recorridos del museo. "
             "Puedes consultarme información de las obras que se encuentran en el MAN."
             "¿En qué puedo ayudarte hoy?")

    # Inicializar el historial de chat si no está presente
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Inicializar la lista de mensajes si no está presente
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Manejo del campo de texto
    user_input = st.text_input("Tu pregunta:", "", key="user_input")

    # Función para enviar el mensaje y obtener respuesta
    def send_message(user_input):
        if user_input:
            # Agregar la interacción al historial (pregunta arriba)
            st.session_state.history.append({"role": "user", "content": user_input})

            
            # Obtener respuesta y mantener el historial
            response = run_guide_llm(user_input, st.session_state.history)

            

            # Agregar la respuesta del asistente al historial
            st.session_state.history.append({"role": "assistant", "content": response})

            # Limpiar el campo de texto (esto se hace solo cuando se envía el mensaje)
            return ""

        return user_input

    # Botón de enviar
    if st.button("Enviar") or user_input:
        user_input = send_message(user_input)

    # Mostrar el historial completo de la conversación (últimos mensajes primero)
    for message in reversed(st.session_state.history):  # Mostrar primero los últimos mensajes
        if message['role'] == 'user':
            st.write(f"**Tú**: {message['content']}")
        else:
            st.write(f"**Asistente**: {message['content']}")

def general_info_interface():
    st.title("Información General del Museo Arqueológico Nacional")
    st.sidebar.image("images/logoMAN.png",  use_container_width =True)
    #st.image("C:\\Users\\David\\Desktop\\MSC DATA SCIENCE\\2 CUATRIMESTRE\\TFM\\MAN\\images\\mapa.jpeg",caption="Mapa del MAN")
    st.write(
        "Bienvenido a la sección de información general. Aquí encontrarás datos interesantes "
        "sobre tarifas, horarios, servicios, y más detalles generales sobre el Museo Arqueológico Nacional."
    )
    chat_history = ChatMessageHistory()  # Inicializar historial vacío

    # Imagen del Museo en la sección de información general

    if 'history' not in st.session_state:
        st.session_state.history = []

    # Manejo del campo de texto
    user_input = st.text_input("Haz tu consulta sobre información general:", key="general_input")

    # Función para enviar el mensaje y obtener respuesta
    if st.button("Consultar") or user_input:
        if user_input:
            response = info_run_guide_llm(user_input,chat_history)  # Llamar al LLM general
            st.write(f"**Respuesta**: {response}")

def main():
    # Imagen del Museo en la barra lateral (sidebar)
    #st.sidebar.image("C:\\Users\\David\\Desktop\\MSC DATA SCIENCE\\2 CUATRIMESTRE\\TFM\\MAN\\images\\ministeriopng.png", caption="Museo Arqueológico Nacional", use_container_width =True)

    st.sidebar.image("images/descarga.jpeg", caption="Museo Arqueológico Nacional", use_container_width =True)

    st.sidebar.title("Elige tu experiencia")
    choice = st.sidebar.radio(
        "¿Qué te gustaría explorar?",
        ("Información General", "Interactuar con el Guía del Museo"),
    )

    if choice == "Información General":
        general_info_interface()
    elif choice == "Interactuar con el Guía del Museo":
        chat_interface()

if __name__ == "__main__":
    main()