from langchain_community.document_loaders import (
    PyMuPDFLoader
)


def load_pdf(
    pdf_path
):

    loader = PyMuPDFLoader(
        pdf_path
    )

    documents = loader.load()

    return documents