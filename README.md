# GenieDoc – AI-Powered Document Assistant
### 🔗 Video Demo

Watch the walkthrough here:  
[ GenieDoc Loom Demo](https://www.loom.com/share/0a26bccb2c3740ffae4bda01cc75eea9?sid=a9d6eb66-ca5d-4af6-833f-22524d3ea8fd)

GenieDoc is a Streamlit-based document assistant that enables users to interact with PDF or TXT files using natural language. The application offers features like document-based Q&A, summarization, and self-testing with feedback, powered by language models and vector-based search.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/ms224141/GeniDoc.git
cd GeniDoc
```

2. Create a Virtual Environment and Activate It
python -m venv venv
source venv/bin/activate         # For Linux/Mac
venv\Scripts\activate            # For Windows


3. Install Dependencies
Make sure your requirements.txt includes the following:
streamlit
requests
PyPDF2
python-dotenv
transformers
sentence-transformers
langchain
faiss-cpu
langchain-community
openai

Then install the packages:
pip install -r requirements.txt

4. Run the Application
streamlit run app_mark3.py

Architecture / Reasoning Flow
1. User Interface Layer
Built using Streamlit.

Sidebar provides four interactive modes:

Home – Welcome and feature overview

Test Me – Asks MCQ-style questions and provides feedback

Q&A Assistant – Ask questions based on uploaded documents

Summarizer – Generates concise summaries from long documents

2. File Handling
Supports both .pdf and .txt formats.

Uses PyPDF2 for PDF extraction and UTF-8 decoding for .txt files.

3. Text Preprocessing
Uses CharacterTextSplitter to break long text into chunks.

Each chunk is ~1000 characters with ~200-character overlap for better context retention.

4. Embedding & Vector Store
Embeddings are generated using sentence-transformers/all-MiniLM-L6-v2.

FAISS is used for efficient similarity search on embedded chunks.

5. LLM API (OpenRouter)
Connects to the mistralai/mistral-7b-instruct model via OpenRouter.

Handles:

Answer generation

Summarization

Feedback and follow-up questions

Prompts are dynamically constructed using the most relevant document chunks.

6. Answer Justification
All answers include short justifications pulled from the document — either quoted lines or referenced sections.

Notes
A .env-based secure key management approach was attempted, but due to technical issues with .env loading in Streamlit, the API key was hardcoded directly into the script for demonstration purposes.

For production use, it is strongly recommended to load sensitive data like API keys using secure environment variables











