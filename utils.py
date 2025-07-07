from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

def process_document(uploaded_file):
    suffix = ".pdf" if uploaded_file.name.lower().endswith(".pdf") else ".txt"

    # Reset file pointer before reading just in case
    uploaded_file.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Load the document using appropriate loader
    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path)

    raw_docs = loader.load()
    print(f"🔍 Loaded {len(raw_docs)} raw document chunks")  # debug

    if not raw_docs:
        print("Warning: No documents loaded from file!")
        return []

    # Split the documents into smaller chunks for processing
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(raw_docs)
    print(f"🔍 Split into {len(docs)} document chunks")  # debug

    return docs
