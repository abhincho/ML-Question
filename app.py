import streamlit as st
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers,LlamaCpp
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_chat import message
import speech_recognition as sr
from langchain.agents import Tool
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import PythonREPL
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import initialize_agent


DB_FAISS_PATH = "vectorestores/db_faiss"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
db = FAISS.load_local(DB_FAISS_PATH,embeddings,allow_dangerous_deserialization=True)

st.set_page_config(page_title="HealthCare Chatbot",layout="wide")

st.title("Question Answering  Chatbot")
#create llm
llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin",model_type="llama",
                    config={'max_new_tokens':128,'temperature':0.01})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type='stuff',
                                              retriever=db.as_retriever(search_kwargs={"k":2}),
                                              memory=memory)
wikipedia = WikipediaAPIWrapper()
duckduckgo = DuckDuckGoSearchRun()

def conversation_chat(query):
    result = chain({'question':query,"chat_history":st.session_state['history']})
    st.session_state['history'].append((query, result['answer']))

    return result['answer']

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! I am here to solve queries!!"]
    
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey"]


def listen_for(seconds: int):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio_data = r.record(source, duration=seconds)
    try:
        text = r.recognize_google(audio_data)
        st.write(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        st.warning("Sorry, I could not understand your speech.")
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
    return ""

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask your queries", key="input")
            use_voice_input = st.checkbox("Use voice input")
            submit_button = st.form_submit_button(label="Submit")

            if submit_button:
                if use_voice_input:
                    text = listen_for(10)
                    if not text:
                        return
                    user_input = text
                output = conversation_chat(user_input)

                wiki_result = wikipedia.run(user_input)
                duck_result = duckduckgo.run(user_input)

                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)
                st.session_state['generated'].append("Wikipedia Result: " + wiki_result)
                st.session_state['generated'].append("DuckDuckGo Result: " + duck_result)
                st.session_state['history'] = st.session_state['past'] + st.session_state['generated']

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                if i % 4 == 0:  # Changed to 4 to accommodate new messages
                    message(st.session_state["past"][i // 4], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                else:
                    message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

initialize_session_state()
# Chat history
display_chat_history()
