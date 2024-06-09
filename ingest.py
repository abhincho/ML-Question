from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.embeddings import HuggingFaceHubEmbeddings,HuggingFaceEmbeddings
from langchain.llms import CTransformers,LlamaCpp
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

# Load the dataset
with open('news.article.json', 'r') as f:
    articles = json.load(f)
    
# Load documents
documents = [article['articleBody'] for article in articles]

# Split the documents into smaller chunks
print("Splitting documents into smaller chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = [text_splitter.split_text(doc) for doc in documents]
flattened_chunks = [item for sublist in chunks for item in sublist]
print(f"Total chunks created: {len(flattened_chunks)}")
#create the embeddings 

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2",model_kwargs={'device':'cpu'})

DB_FAISS_PATH = "vectorestores/db_faiss"

#create a vectore store 

vectore_store = FAISS.from_texts(flattened_chunks,embeddings)
vectore_store.save_local(DB_FAISS_PATH)

