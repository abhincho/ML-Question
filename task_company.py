import json
from datetime import datetime
from transformers import pipeline
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load a pre-trained BERT model for text classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def load_articles(file_path, limit=1000):
    print("Loading articles...")
    with open(file_path, 'r', encoding='utf-8') as file:
        articles = json.load(file)
    return articles[:limit]

def is_related_bert(article_body):
    candidate_labels = ["Israel", "Hamas", "Gaza", "Palestine"]
    result = classifier(article_body, candidate_labels)
    print(f"Classification result: {result}")  # Debugging statement
    return any(score > 0.5 for score in result['scores'])  # Threshold for relevance

def create_timeline_bert(articles, max_entries=100):
    timeline = []
    for index, article in enumerate(articles):
        print(f"Processing article {index + 1}/{len(articles)}")  # Progress indication
        date = article.get("dateModified", {}).get("$date", "")
        title = article.get("title", "")
        description = article.get("articleBody", "").split("\n\n")[0]
        
        if date and title and is_related_bert(description):
            date_formatted = datetime.fromisoformat(date.replace("Z", "+00:00")).strftime("%Y-%m-%d")
            timeline.append({
                "date": date_formatted,
                "title": title,
                "description": description
            })
    timeline.sort(key=lambda x: x["date"])
    
    # Limit the number of entries in the timeline
    if max_entries and len(timeline) > max_entries:
        timeline = timeline[:max_entries]
    
    return timeline

def print_timeline(timeline):
    for entry in timeline:
        print(f"{entry['date']}: {entry['title']} - {entry['description']}")

def save_timeline(timeline, file_path):
    with open(file_path, 'w', encoding='utf-8') as output_file:
        for entry in timeline:
            output_file.write(f"{entry['date']}: {entry['title']} - {entry['description']}\n")

def main():
    file_path = r'C:\Users\email\Downloads\yutr.json'
    articles = load_articles(file_path, limit=1000)  # Limit to 1000 articles
    print("Creating timeline...")
    timeline = create_timeline_bert(articles, max_entries=100)  # Adjust max_entries as needed
    print("Printing timeline...")
    print_timeline(timeline)
    print("Saving timeline...")
    save_timeline(timeline, 'timeline_summarization.txt')
    print("Done.")

if __name__ == "__main__":
    main()


