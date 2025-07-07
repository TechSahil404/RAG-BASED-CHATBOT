from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_qa_chain():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1,
        max_new_tokens=200,
        temperature=1,
    )
    model = HuggingFacePipeline(pipeline=pipe)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return model, embeddings

def answer_question(question, docs):
    model, embeddings = load_qa_chain()
    
    # Docs ko chhote chunks me todna zaroori hai taaki retrieval achha ho
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_docs = text_splitter.split_documents(docs)

    db = FAISS.from_documents(chunked_docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever)
    result = qa_chain.run(question)
    return result, "Retrieved from top chunks" 