
## Implementation

### 1. Set up:

Let's start by installing the following packages

```bash
  !pip install langchain-community langchain openai
  !pip install sentence_transformers chromadb
  !pip install jq
  !pip install tqdm
```
Alternatively, you can directly install the requred dependencies using

```bash
  !pip install -r requirements.txt

```
Now we import the necessary libraries and modules for this project

```bash
from tqdm import tqdm
from langchain_community.chat_models import ChatAnyscale
from langchain.llms import Anyscale
import os
from langchain_community.document_loaders import JSONLoader
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
import json
from pathlib import Path
from pprint import pprint

```
These include tools for progress bars (tqdm), language models (ChatAnyscale, Anyscale), file handling (os, pathlib), document handling (JSONLoader, Document), vector storage (Chroma), embeddings (HuggingFaceEmbeddings), and creating retrieval chains (ConversationalRetrievalChain).

### 2. Anyscale:

Anyscale is a company that provides infrastructure and tools to build, deploy, and manage scalable AI and machine learning applications.

#### > Get you Anyscale api key:
Create an account at:
https://app.endpoints.anyscale.com/welcome

Once your account has been created, go to: https://app.endpoints.anyscale.com/credentials

Then go to your profile and click on api_keys, from where you can create you api key.

I have uploaded my api key as a text file on my drive and then mounted the drive in colab.
```bash
from google.colab import drive
drive.mount('/content/gdrive')

# Read the API key from the file
api_key_file_path = '/content/gdrive/MyDrive/api_key.txt'# Adjust the path accordingly
with open(api_key_file_path, 'r') as f:
    api_key = f.read().strip()

# Set the environment variable
import os
os.environ['ANYSCALE_API_KEY'] = api_key

```
Set up the API key and base URL for the Anyscale API.
```bash
os.environ['ANYSCALE_API_BASE'] = 'https://app.endpoints.anyscale.com/v1'
os.environ['ANYSCALE_API_KEY'] = ANYSCALE_API_KEY

```


### 3. Model and Embeddings Initialization:
Initializes the language model (ChatAnyscale) with a specific model, i.e, Meta-Llama-3-8B-Instruc and also initializes the embeddings model (HuggingFaceEmbeddings) with a specific model, i.e, e5-small-v2
```bash
model = 'meta-llama/Meta-Llama-3-8B-Instruct'
LLM = ChatAnyscale(model_name=model)
embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

```
### 4. JASON file:
First I have uploaded the JSON dataset to my drive. Then we set the path to the JSON file and load it.
```bash
json_path = "/content/drive/MyDrive/news.article.json"
with open(json_path, 'r') as f:
    data = json.load(f)

```
Since the JSON file is huge with approximately 37000 news articles, I have split the data into smaller chunks of 1000 records each and saves each chunk into a separate JSON file.
```bash
chunk_size = 1000 
output_dir = Path("/content/split_json_files")
output_dir.mkdir(parents=True, exist_ok=True)
for i in range(0, len(data), chunk_size):
    chunk = data[i:i + chunk_size]
    chunk_file_path = output_dir / f"chunk_{i // chunk_size + 1}.json"
    with open(chunk_file_path, 'w') as chunk_file:
        json.dump(chunk, chunk_file, indent=2)
    print(f"Saved {len(chunk)} records to {chunk_file_path}")

```
Next, processes each JSON file and convert the JSON data into a list of Document objects. Then create a Chroma vector store in batches of 1000 documents and stores the embeddings persistently.
```bash
for json_file in tqdm(split_json_dir.glob("*.json")):
    with open(json_file, 'r') as f:
        data = json.load(f)
    documents = [Document(page_content=item['articleBody']) for item in data]
    batch_size = 1000
    for i in tqdm(range(0, len(documents), batch_size), leave=False):
        batch_documents = documents[i:i + batch_size]
        vectordb = Chroma.from_documents(batch_documents, embedding=embeddings_model, persist_directory="/content/embeddings/json_embeddings")
        vectordb.persist()

```
Create a ConversationalRetrievalChain that uses the language model (LLM) and the vector store retriever (vectordb). The retriever is configured to return the top 6 relevant documents (search_kwargs={'k': 6}).
```bash
qa = ConversationalRetrievalChain.from_llm(LLM,
                                           retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
                                           return_source_documents=True,
                                           verbose=False)

```
### 5. Query the Model
Define a query (For example: What happened at Al-Shifa Hospital?).
Use the qa chain to process the query and retrieve relevant information and print the answer retrieved by the chain.
```bash
query = 'What happened at Al-Shifa Hospital?'
result = qa({"question": query, "chat_history": []})
print(result['answer'])

```