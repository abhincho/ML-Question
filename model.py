from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.embeddings import HuggingFaceHubEmbeddings,HuggingFaceEmbeddings
from langchain.llms import CTransformers,LlamaCpp
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from langchain.agents import Tool
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import PythonREPL
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import initialize_agent

DB_FAISS_PATH = "vectorestores/db_faiss"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
vectore_store = FAISS.load_local(DB_FAISS_PATH,embeddings)
#create llm 

llm  = CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin",model_type="llama",
                     config = {'max_new_tokens':128,'temperature':0.01})
memory = ConversationBufferMemory(memory_key="chat_history",return_messages = True)
# create the chain
chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type="stuff",
                                              retriever=vectore_store.as_retriever(search_kwargs={"k":2}),
                                              memory = memory)
# Function to handle user input and get the response
def conversation_chat(query):
    result = chain({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    return result["answer"]

# Initialize chat history
chat_history = []


user_input = input("Question: ")
response = conversation_chat(user_input)
print(f"Response: {response}")


wikipedia = WikipediaAPIWrapper()
duckduckgo = DuckDuckGoSearchRun()

# Continue taking user input and generating responses
while True:
    user_input = input("Question: ")
    if user_input.lower() == "exit":
        break
    response = conversation_chat(user_input)
    print(f"Response: {response}")
    wiki_result = wikipedia.run(user_input)
    duck_result = duckduckgo.run(user_input)

    print("Wikipedia Output : " ,wiki_result)
    print("DuckDuckGo Output : ", duck_result)


