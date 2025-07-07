**RAG-based Chatbot System**

An intelligent document-aware assistant built using Retrieval-Augmented Generation (RAG) architecture.  
This AI-powered tool reads PDF/TXT documents, generates concise summaries, answers user questions, and challenges users with logic-based questions — all grounded in document context.

## Features

-  Upload structured documents (PDF or TXT)
-  Auto Summary (within 150 words)
-  **Ask Anything** mode: Ask any question based on document content
-  **Challenge Me** mode: System-generated logic/comprehension-based questions with feedback
-  Justification for every answer (e.g., "This is supported by paragraph 3...")
-  Uses a local lightweight open-source model — no internet required


# Model Choice & Efficiency


Originally, the project used the **Falcon-7B-Instruct** model for its high-quality generation capabilities.  
However, due to its **14GB download size** and requirement of **12GB+ RAM**, it was replaced with a **lighter open-source model** suitable for local use on modest systems.

This ensures better efficiency and local deployment without needing high-end GPUs or paid APIs.


**Future Enhancement Suggestion:**  
For even better speed and quality, this system can be upgraded to use:
- OpenAI (ChatGPT)
- Gemini (Google)
- Claude (Anthropic)  
These APIs can significantly enhance performance if system requirements or budget allow.

---

##  Architecture

- **Frontend**: Streamlit (for clean, interactive UI)
- **Backend**: Python + LangChain + HuggingFace Transformers
- **Model**: Open-source `text2text-generation` model (can be replaced with Falcon, OpenAI, etc.)
- **Vector Store**: FAISS for semantic document retrieval
- **Document Splitter**: RecursiveCharacterTextSplitter
- **File Reader**: PyPDFLoader / TextLoader (LangChain)


## Interface Modes

### 1. Ask Anything
User asks any question → System searches document → Answers with context + reference.

### 2. Challenge Me
System generates 3 logic-based questions → User attempts answers → System evaluates and explains.



