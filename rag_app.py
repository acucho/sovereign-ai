import streamlit as st
import ollama
from pypdf import PdfReader
import io

st.set_page_config(
    page_title="Asistente Medico Privado",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Asistente Medico Privado")
st.caption("Powered by IA local · Sus datos nunca salen de esta red · RAG activado")

with st.sidebar:
    st.header("Documentos")
    st.caption("Sube los protocolos del hospital")

    archivos = st.file_uploader(
        "Cargar PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if archivos:
        st.success(f"{len(archivos)} documento(s) cargado(s)")
        for f in archivos:
            st.write(f"OK: {f.name}")
    else:
        st.info("Sube al menos un PDF para empezar")

    st.divider()
    st.caption("LLM: llama3.2:3b")
    st.caption("Modo: 100% privado")

# Extraer texto del PDF directamente
def extraer_texto(archivos):
    texto_completo = ""
    for archivo in archivos:
        reader = PdfReader(io.BytesIO(archivo.read()))
        for pagina in reader.pages:
            texto_completo += pagina.extract_text() + "\n"
    return texto_completo

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

if "texto_pdf" not in st.session_state:
    st.session_state.texto_pdf = ""

# Procesar PDF cuando se sube
if archivos and not st.session_state.texto_pdf:
    with st.spinner("Procesando documentos..."):
        st.session_state.texto_pdf = extraer_texto(archivos)
        st.sidebar.success("Documento procesado.")

# Mostrar historial
for msg in st.session_state.mensajes:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not archivos:
    st.warning("Sube un PDF en el panel izquierdo para activar el asistente")
else:
    if pregunta := st.chat_input("Consulta sobre los documentos cargados..."):

        st.session_state.mensajes.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        with st.chat_message("assistant"):
            with st.spinner("Analizando documento..."):

                # Pasar el texto completo del PDF como contexto
                prompt_sistema = f"""Eres un asistente medico especializado.
Tienes acceso al siguiente documento medico:

=== INICIO DEL DOCUMENTO ===
{st.session_state.texto_pdf}
=== FIN DEL DOCUMENTO ===

INSTRUCCIONES:
- Responde UNICAMENTE basandote en el documento anterior
- Si la informacion esta en el documento, respondela detalladamente en español
- Cita la seccion del documento donde encontraste la informacion
- Si NO esta en el documento, di exactamente: "Esta informacion no esta en el documento"
- NUNCA uses conocimiento externo al documento"""

                respuesta = ollama.chat(
                    model="llama3.2:3b",
                    messages=[
                        {"role": "system", "content": prompt_sistema},
                        {"role": "user", "content": pregunta}
                    ]
                )
                texto = respuesta["message"]["content"]
                st.markdown(texto)

        st.session_state.mensajes.append({"role": "assistant", "content": texto})

if st.session_state.texto_pdf:
    if st.sidebar.button("Limpiar y cargar nuevos documentos"):
        st.session_state.texto_pdf = ""
        st.session_state.mensajes = []
        st.rerun()
