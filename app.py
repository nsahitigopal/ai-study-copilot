import streamlit as st

from dotenv import load_dotenv

from utils.pdf_processing import (
    load_pdf
)

from utils.chunking import (
    create_chunks
)

from utils.embeddings import (
    generate_embeddings
)

from utils.vector_store import (
    create_faiss_index
)

from utils.retrieval import (
    retrieve_context
)

from utils.llm import (
    generate_answer
)

load_dotenv()

st.set_page_config(
    page_title="AI Study Copilot",
    layout="wide"
)

st.title(
    "📚 AI Study Copilot"
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    if "index" not in st.session_state:

        with open(
            "temp.pdf",
            "wb"
        ) as f:

            f.write(
                uploaded_file.getbuffer()
            )

        with st.spinner(
            "Processing PDF..."
        ):

            documents = load_pdf(
                "temp.pdf"
            )

            chunks = create_chunks(
                documents
            )

            embeddings = (
                generate_embeddings(
                    chunks
                )
            )

            index = (
                create_faiss_index(
                    embeddings
                )
            )

            st.session_state[
                "chunks"
            ] = chunks

            st.session_state[
                "index"
            ] = index

        st.success(
            "PDF processed successfully!"
        )

    question = st.text_input(
        "Ask a question about the PDF"
    )

    if (
        st.button("Ask")
        and question
    ):

        with st.spinner(
            "Thinking..."
        ):

            context = (
                retrieve_context(
                    question,
                    st.session_state[
                        "index"
                    ],
                    st.session_state[
                        "chunks"
                    ]
                )
            )

            answer = (
                generate_answer(
                    question,
                    context
                )
            )

        st.subheader(
            "Answer"
        )

        st.write(
            answer
        )