from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load the HuggingFacePipeline model with Flan-T5 Small
def _get_model():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1,  # Use CPU #Falcon model requir 12gb+ RAM 
        max_new_tokens=200,
        temperature=0.5
    )
    return HuggingFacePipeline(pipeline=pipe)

# Generate logic-based questions from the document
def generate_questions(docs):
    if not docs:
        print("Error: No documents found for question generation.")
        return []

    # Use first chunk for question generation
    context = docs[0].page_content[:1000]
    prompt = PromptTemplate.from_template(
        "Read the context and generate 3 logic-based questions:\n\n{context}"
    )

    chain = LLMChain(llm=_get_model(), prompt=prompt)
    questions = chain.run(context)

    # Split questions line-by-line and clean them
    return [q.strip() for q in questions.split("\n") if q.strip()]

# Evaluate user answers based on the document
def evaluate_answers(questions, user_answers, docs):
    if not docs:
        print("Error: No documents found for answer evaluation.")
        return []

    # Use first 2 chunks as context for evaluation
    context = "\n".join([doc.page_content for doc in docs[:2]])
    model = _get_model()
    feedbacks = []

    for q, a in zip(questions, user_answers):
        prompt = f"""You are a logical evaluator.
Question: {q}
User Answer: {a}
Context: {context}

Give constructive feedback about whether the answer is correct, partially correct, or incorrect, and why. \n
Keep it short and informative."""

        feedback = model(prompt)  # Fixed: returns string directly
        feedbacks.append(feedback.strip())

    return feedbacks
