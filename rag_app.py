import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import os
import tempfile

st.set_page_config(
    page_title="Asistente Medico Privado — RAG",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Asistente Medico Privado")
st.caption("Powered by IA local · Sus datos nunca salen de esta red · RAG activado")

# llama3.2:3b = 2 GB — suficiente para seguir instrucciones RAG correctamente
# nomic-embed-text = 274 MB para buscar en documentos
# Total RAM: ~2.3 GB — cabe perfectamente en tu PC
Settings.llm = Ollama(model="llama3.2:3b", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Prompt explicito — le dice exactamente que usar el documento
PROMPT_RAG = PromptTemplate(
    "Eres un asistente medico especializado. "
    "Usa UNICAMENTE la siguiente informacion del documento para responder. "
    "Si la respuesta esta en el documento, respondela detalladamente en español. "
    "Si genuinamente no esta en el documento, di: No encontre esa informacion en el documento.\n\n"
    "Informacion del documento:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Pregunta: {query_str}\n"
    "Respuesta detallada en español:"
)

with st.sidebar:
    st.header("Documentos")
    st.caption("Sube los protocolos o documentos del hospital")

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
    st.caption("LLM: llama3.2:3b (2.0 GB)")
    st.caption("Embeddings: nomic-embed-text (274 MB)")
    st.caption("Total RAM: ~2.3 GB")
    st.caption("Modo: 100% privado")

if "mensajes_rag" not in st.session_state:
    st.session_state.mensajes_rag = []

if "indice" not in st.session_state:
    st.session_state.indice = None

if archivos and st.session_state.indice is None:
    with st.spinner("Procesando documentos..."):
        carpeta_temp = tempfile.mkdtemp()
        for archivo in archivos:
            ruta = os.path.join(carpeta_temp, archivo.name)
            with open(ruta, "wb") as f:
                f.write(archivo.getbuffer())

        documentos = SimpleDirectoryReader(carpeta_temp).load_data()
        st.session_state.indice = VectorStoreIndex.from_documents(documentos)
        st.success("Documentos procesados. Ya puedes hacer preguntas.")

for msg in st.session_state.mensajes_rag:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not archivos:
    st.warning("Sube un PDF en el panel izquierdo para activar el asistente")
else:
    if pregunta := st.chat_input("Consulta sobre los documentos cargados..."):

        st.session_state.mensajes_rag.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        with st.chat_message("assistant"):
            with st.spinner("Buscando en documentos..."):
                motor = st.session_state.indice.as_query_engine(
                    similarity_top_k=5,
                    text_qa_template=PROMPT_RAG
                )
                resultado = motor.query(pregunta)
                texto = str(resultado)
                st.markdown(texto)

        st.session_state.mensajes_rag.append({"role": "assistant", "content": texto})

if st.session_state.indice is not None:
    if st.sidebar.button("Limpiar y cargar nuevos documentos"):
        st.session_state.indice = None
        st.session_state.mensajes_rag = []
        st.rerun()
