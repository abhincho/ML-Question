import json
import gdown
from datetime import datetime
from transformers import TFBartForConditionalGeneration, BartTokenizer


def download_file_from_google_drive(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data[:100]

def filter_articles_by_keyword(articles, keyword):
    filtered_articles = []
    for article in articles:
        if keyword in article['articleBody'].lower() or keyword in article['title'].lower():
            filtered_articles.append(article)
    return filtered_articles

def summarize_text(text, model, tokenizer, max_length=150, min_length=40):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def generate_timeline(articles, model, tokenizer):
    events = []
    current_event = None
    combined_summary = ""
    for article in articles:
        if 'dateModified' not in article or '$date' not in article['dateModified']:
            continue
        article_date = datetime.strptime(article['dateModified']['$date'], "%Y-%m-%dT%H:%M:%S.%fZ").date()
        
        # Group articles by date and summarize
        if not current_event or article_date != current_event['end_date']:
            if current_event:
                combined_summary = summarize_text(combined_summary, model, tokenizer)
                current_event['description'] = combined_summary
                events.append(current_event)
            current_event = {'start_date': article_date, 'end_date': article_date, 'description': article['title']}
            combined_summary = article['articleBody']
        else:
            current_event['end_date'] = article_date
            combined_summary += " " + article['articleBody']
    
    if current_event:
        combined_summary = summarize_text(combined_summary, model, tokenizer)
        current_event['description'] = combined_summary
        events.append(current_event)

    return events

def print_timeline(timeline):
    for event in timeline:
        print(f"{event['start_date']} - {event['end_date']}: {event['description']}\n")

def main():
    file_id = '1q6KVw4LD_rnXKVViVkBpdTyedHdQvtkk'  # GDrive file ID
    file_path = 'data.json'  # Destination path to save the dataset
    keyword = 'israel'

    # Download and load data
    download_file_from_google_drive(file_id, file_path)
    data = load_json(file_path)

    # Filter relevant articles
    relevant_articles = filter_articles_by_keyword(data, keyword)
    
    # Initialize summarization model and tokenizer
    model = TFBartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

    # Generate and print timeline summarization
    timeline = generate_timeline(relevant_articles, model, tokenizer)
    print_timeline(timeline)

if __name__ == "__main__":
    main()
