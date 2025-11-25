import streamlit as st
from rag_agent import rag_agent

# Configuración general de la página
st.set_page_config(
    page_title="Asistente UNAD IA",
    page_icon="🎓",
    layout="centered"
)

st.title("🤖 Asistente UNAD IA")
st.caption("Consulta programas académicos, políticas y reglamentos de la UNAD.")

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Mostrar el historial del chat
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada del usuario
if prompt := st.chat_input("Escribe tu pregunta sobre la UNAD..."):
    # Mostrar la pregunta del usuario
    st.chat_message("user").markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                # Invocar al agente RAG
                response = rag_agent.invoke(prompt)
                st.markdown(response)
            except Exception as e:
                response = f"❌ Error al procesar tu solicitud: {e}"
                st.error(response)

    st.session_state["messages"].append({"role": "assistant", "content": response})
