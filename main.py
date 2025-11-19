import numpy as np
import pandas as pd
from transformers import AutoTokenizer, pipeline
from sklearn.metrics import accuracy_score
import os

# Optional: prevent tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer=tokenizer)

# Sample text
news_1 = "I hated the wait times"
news_2 = "Very happy with the service"
news_3 = "Wait time on the call center was too long"
news_4 = "At first I was happy but then the agent was rude to me"
news_5 = "My name was mispelled on my file and my payment did not come on time"
news_6 = "I kept having errors trying to use the online service"
news_7 = "Load times on the application were not great"
news_8 = "Adequate service at OK speed"
news_9 = "I was pleasantly surprised"
news_10 = "Service was fast, efficient, and accurate"

print("News 1 Sentiment:", sentiment_pipeline(news_1))
print("News 2 Sentiment:", sentiment_pipeline(news_2))
print("News 3 Sentiment:", sentiment_pipeline(news_3))
print("News 4 Sentiment:", sentiment_pipeline(news_4))
print("News 5 Sentiment:", sentiment_pipeline(news_5))
print("News 6 Sentiment:", sentiment_pipeline(news_6))
print("News 7 Sentiment:", sentiment_pipeline(news_7))
print("News 8 Sentiment:", sentiment_pipeline(news_8))
print("News 9 Sentiment:", sentiment_pipeline(news_9))
print("News 10 Sentiment:", sentiment_pipeline(news_10))