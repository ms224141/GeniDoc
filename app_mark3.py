import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.callbacks.manager import get_openai_callback
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Together
from langchain.chains.question_answering import load_qa_chain
from transformers import pipeline
import os
from langchain.chains.question_answering import load_qa_chain
import requests



import streamlit as st
st.set_page_config(page_title ="GenieDoc App")
st.title("GenieDoc")
#  now we will   try to create a side bar so that we can do different different functions on different different pages
page=st.sidebar.radio(
    "Choose operation you want to perform on the document",
    ["Home","Test me","Q&A Assistent","Summarizer"]
)
if page=="Home":
    st.header("Welcome to GenieDoc yor document helper")
    st.markdown("""This is your one_stop AI assistent which will help you for :
    \n- Document Q&A
    \n- PDF Summarization
    \n- Your Knowledge tester""")
elif page=="Test me":
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    API_KEY = "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"

    # Set page config
    st.set_page_config(page_title="One-by-One Q&A", layout="centered")
    st.title("Doc Interactive Q&A")

    # Session variables
    for key in ["text", "qa_pairs", "current_question", "user_answer", "feedback"]:
        if key not in st.session_state:
            st.session_state[key] = None if key != "qa_pairs" else []

    # Upload file
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    # Extract text from file
    if uploaded_file and st.session_state.text is None:
        text = ""
        if uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")

        st.session_state.text = text.strip()


    # Function to call LLM
    def query_llm(prompt, temp=0.5):
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mistralai/mistral-7b-instruct:free",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "max_tokens": 500
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()
        return result['choices'][0]['message']['content'] if 'choices' in result else "Error from API"


    # Ask first question
    if st.session_state.text and st.session_state.current_question is None:
        prompt = f"Read this document and ask the first simple question from it:\n\n{st.session_state.text[:3000]}"
        st.session_state.current_question = query_llm(prompt)

    # Display current question
    if st.session_state.current_question:
        st.markdown(f"**Q: {st.session_state.current_question}**")
        answer = st.text_input("Your Answer:")

        if st.button("Submit Answer") and answer.strip():
            # Feedback generation
            feedback_prompt = f"""
            Check the user's answer and give feedback.
            Question: {st.session_state.current_question}
            Answer: {answer}
            Context: {st.session_state.text[:3000]}
            """
            feedback = query_llm(feedback_prompt, temp=0.4)

            # Save this round
            st.session_state.qa_pairs.append({
                "question": st.session_state.current_question,
                "answer": answer,
                "feedback": feedback
            })
            st.session_state.feedback = feedback
            st.session_state.user_answer = answer

    # Show feedback and prepare follow-up question
    if st.session_state.feedback:
        st.markdown(f"**Feedback:** {st.session_state.feedback}")

        if st.button("Next Question"):
            followup_prompt = f"""
            Based on this question and user's answer, ask a relevant follow-up question.
            Previous Q: {st.session_state.current_question}
            User's Answer: {st.session_state.user_answer}
            Context: {st.session_state.text[:3000]}
            """
            followup_q = query_llm(followup_prompt)
            st.session_state.current_question = followup_q
            st.session_state.user_answer = None
            st.session_state.feedback = None


elif page == "Q&A Assistent":
    api_key = "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"
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    st.set_page_config(page_title="GenieDoc")
    st.header("GenieDoc")
    #  now we are going to upload file for further operations
    user_pdf = st.file_uploader("upload your file here", type=["pdf", "txt"])
    if user_pdf is not None:
        text = ""
        if user_pdf.type == "text/plain":
            text = user_pdf.read().decode("utf-8")
        elif user_pdf.type == "application/pdf":
            pdf_reader = PdfReader(user_pdf)
            for page in pdf_reader.pages:
                content = page.extract_text()
                if content:
                    text += content
        #   now we will try to devide the whole text in small small parts to make things easy

        if not text.strip():
            st.warning("No text extracted from the uploaded file. Please try another file.")
        else:
            # here we are spliting the text of file in small small parts
            splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = splitter.split_text(text)

            if not chunks:
                st.warning("Could not split the text into chunks. Try a different document.")
            else:
                #     now we will convert the text data into machine understandable format by using embedding

                embedding = embedding = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    cache_folder="hf_cache",  # Optional
                    model_kwargs={'device': 'cpu'}
                )
                insights = FAISS.from_texts(chunks, embedding)
        #  now we will take input from user

        question = st.text_input("You may ask any questions here related to the file you have uploaded")
        if question:
            document = insights.similarity_search(question)
            top_docs = document[:2]  # Limiting to top 2 most relevant chunks
            context = "\n".join([doc.page_content for doc in top_docs])
            system_prompt = "your role is to answer question in short from the doucument provided, just provide answer and give the part of document from where you have taken the answer to justify it"
            payload = {
                "model": "mistralai/mistral-7b-instruct:free",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant answering questions from documents. Each answer must be concise and include a brief justification from the document (e.g., 'This is supported by paragraph 3 of section 1' or quote the relevant line)."},

                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
                ],
                "max_tokens": 500
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            response = requests.post(API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                try:
                    result = response.json()
                    answer = result['choices'][0]['message']['content']
                    st.write(answer)
                except Exception as e:
                    st.write(f"Error in processing response: {e}")
                    st.write(f"Raw Response: {response.text}")
            else:
                st.write(f"Request failed with status code: {response.status_code}")
                st.write(f"Response content: {response.text}")
        else:
            st.write("No relevant document found.")

elif page == "Summarizer":
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    api_key = "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"
    st.header("PDF/Text Summarizer")
    user_pdf = st.file_uploader("upload your file here", type=["pdf", "txt"])

    def call_deepseek(prompt):
        payload = {
            "model": "mistralai/mistral-7b-instruct:free",
            "messages": [
                {"role": "system", "content": "You are a professional summarizer."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.4,
            "max_tokens": 1500
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"


    text = ""
    if user_pdf is not None:
        if user_pdf.type == "text/plain":
            text = user_pdf.read().decode("utf-8")
        elif user_pdf.type == "application/pdf":
            pdf_reader = PdfReader(user_pdf)
            for page in pdf_reader.pages:
                content = page.extract_text()
                if content:
                    text += content
#      now we will make the main objective that will do the actual summarization part
#      since we are making the model for summarizing  a big or small file so our model will not be able to do the task in one go
    def split_text(text,max_chunk_size=1000):
        lst=[]
        for i in range(0, len(text), max_chunk_size):
            lst.append(text[i:i+max_chunk_size])
        return lst
#     now write a function for summarization of the text
    def summarization(text):
        parts=split_text(text,max_chunk_size=3000)
        summaries=[]
        for j in parts:
            prompt=f"summarize the following:\n\n{j}"
            summary=call_deepseek(prompt)
            summaries.append(summary)
#      now we have summaries of every 3000 words chunk to reduce it into 150 words we have to again summarize the summary fo different  different chunks
        summary_for_last_summary="\n".join(summaries)[:3000]
        last_prompt=f"Combine and summarize the following summaries into a concise final version:\n\n{summary_for_last_summary}"
        last_summary=call_deepseek(last_prompt)
        return last_summary
    if user_pdf is not None and text.strip()!="":
        final_summary = summarization(text)
        st.header("your final summary is: \n")
        st.write(final_summary)
        st.download_button("Download your files summary", data=final_summary, file_name="summary.txt")

    else:
        st.warning("Please upload a valid file to generate summary")






