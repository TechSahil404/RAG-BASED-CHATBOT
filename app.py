import streamlit as st
from backend import answer_question
from challenge import generate_questions, evaluate_answers
from summarizer import summarize_document
from utils import process_document

st.title("Smart Assistant for Research Summarization")

uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file is not None:
    st.success("Document uploaded successfully!")

    docs = process_document(uploaded_file)

    # Debug print to check what docs contains
    st.write("Debug - Loaded docs:", docs)

    if not docs or len(docs) == 0:
        st.error("No content found in the uploaded document. Please upload a valid PDF or TXT file.")
    else:
        st.subheader("📄 Document Summary")
        summary = summarize_document(docs)
        st.write(summary)

        mode = st.radio("Choose Mode", ["Ask Anything", "Challenge Me"])

        if mode == "Ask Anything":
            question = st.text_input("Ask a question from the document")
            if question.strip() != "":
                answer, source = answer_question(question, docs)
                st.markdown(f"**Answer:** {answer}")
                st.markdown(f"*Source:* {source}")

        elif mode == "Challenge Me":
            questions = generate_questions(docs)
            if not questions or len(questions) == 0:
                st.warning("No questions could be generated. Try uploading a document with more content.")
            else:
                user_answers = []
                st.write("Answer the following questions:")
                for i, q in enumerate(questions):
                    ans = st.text_input(f"Q{i+1}: {q}")
                    user_answers.append(ans)

                # Check if all questions have answers before evaluating
                if len(user_answers) == len(questions) and all(ans.strip() != "" for ans in user_answers):
                    feedback = evaluate_answers(questions, user_answers, docs)
                    for i, fb in enumerate(feedback):
                        st.markdown(f"**Q{i+1} Feedback:** {fb}")
