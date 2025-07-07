from langchain.chains.summarize import load_summarize_chain
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

def summarize_document(docs):
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1,
        max_new_tokens=200,
        temperature=1,
    )
    model = HuggingFacePipeline(pipeline=pipe)
    chain = load_summarize_chain(model, chain_type="map_reduce")
    summary = chain.invoke(docs)  # .run() ki jagah .invoke() use karo
    return summary
