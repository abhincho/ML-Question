import json
import re
from transformers import pipeline
from rank_bm25 import BM25Okapi
import wikipedia
import gdown
import os

# Define the URL from Google Drive (make sure it's set to 'Anyone with the link can view')
url = 'https://drive.google.com/file/d/1q6KVw4LD_rnXKVViVkBpdTyedHdQvtkk/view?usp=sharing '
output = 'news.article.json'

# Download the file if not already present
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# Loading the dataset
with open(output, 'r') as file:
    articles = json.load(file)

# Filter relevant articles
def filter_relevant_articles(articles, keywords):
    relevant_articles = []
    for article in articles:
        if any(keyword in article['content'].lower() for keyword in keywords):
            relevant_articles.append(article)
    return relevant_articles

keywords = ['israel', 'hamas', 'gaza', 'idf', 'palestine']
filtered_articles = filter_relevant_articles(articles, keywords)

# Cleaning text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

for article in filtered_articles:
    article['content'] = clean_text(article['content'])

# Load a pre-trained BERT QA model
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Function to get answers from the model
def get_answer(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Prepare documents for BM25
documents = [article['content'] for article in filtered_articles]
tokenized_docs = [doc.split(" ") for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# Retrieve relevant articles
def retrieve_articles(question, top_n=5):
    tokenized_question = question.split(" ")
    scores = bm25.get_scores(tokenized_question)
    top_n_indices = scores.argsort()[-top_n:][::-1]
    return [filtered_articles[i] for i in top_n_indices]

# Building the pipeline
def answer_question(question):
    relevant_articles = retrieve_articles(question)
    context = " ".join([article['content'] for article in relevant_articles])
    answer = get_answer(question, context)
    return answer

# Function to augment with Wikipedia
def augment_with_wikipedia(question):
    try:
        summary = wikipedia.summary(question, sentences=5)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return wikipedia.summary(e.options[0], sentences=5)
    except Exception as e:
        return ""

# Function to answer question with augmentation
def answer_question_with_augmentation(question):
    relevant_articles = retrieve_articles(question)
    context = " ".join([article['content'] for article in relevant_articles])
    answer = get_answer(question, context)
    additional_info = augment_with_wikipedia(question)
    
    return f"Answer: {answer}\n\nAdditional Information: {additional_info}"

# Example usage
if __name__ == "__main__":
    question = "What happened at the Al-Shifa Hospital?"
    final_answer = answer_question_with_augmentation(question)
    print(final_answer)
