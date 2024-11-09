import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from utilities.constants import POSITIVE_QUESTIONS, NEGATIVE_QUESTIONS, NEUTRAL_QUESTIONS

global grouped_reviews
global qa_df
device='cuda' if torch.cuda.is_available() else 'cpu'
tqdm.pandas()


def build_grouped_data(group):
    """
    Build grouped data for summarization
    :param group: grouped dataframe on product id
    :return: insert rows in global dataframe
    """

    global grouped_reviews
    total_rows = len(group)

    for i in range(0, total_rows, n_size):
        values = ["User: " + rev for rev in group.combined[i:i + n_size].values]
        text = " ".join(values)
        grouped_reviews.loc[len(grouped_reviews)] = [group.product_id.iloc[0], group.weak_label.iloc[0], text]


def get_summary(text):
    """
    Generate summary for the text using the BART samsum model
    :param text: text to be summarized. (grouped and chunked reviews
    :return: summary of the text
    """

    try:
        inputs = tokenizer([text], return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = model.generate(inputs['input_ids'].to(device), num_beams=5, max_length=256)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(e)
        return ""


def get_label_score(text):
    """
    Get sentiment label and score for the text
    :param text:
    :return:
    """

    sentiment = sentiment_classifier(text)
    return sentiment[0]['label'], sentiment[0]['score']


def get_question(sentiment):
    """
    Get a random question based on sentiment
    :param sentiment: sentiment of the text: positive, negative, neutral
    :return: random question
    """

    if sentiment == 'positive':
        return np.random.choice(POSITIVE_QUESTIONS)
    elif sentiment == 'negative':
        return np.random.choice(NEGATIVE_QUESTIONS)
    else:
        return np.random.choice(NEUTRAL_QUESTIONS)


def build_qa_data(row):
    """
    Generate rows where agent's answer is not available
    :param row: dataframe row with question
    :return: row added to the global general questions dataframe
    """

    global qa_df
    qa_df.loc[len(qa_df)] = [
        np.nan,
        np.nan,
        "I'm here to help with questions related to Amazon products. It seems like this query isn't about a product, so I may not be able to assist with it. Please feel free to ask anything about products, and I'll be happy to help!",
        np.nan,
        np.nan,
        row.instruction]


if __name__ == "__main__":
    grouped_reviews = pd.DataFrame(columns=['product_id', 'weak_label', 'review'])
    qa_df = pd.DataFrame(columns=['product_id', 'review', 'summary', 'sentiment', 'sentiment_score', 'question'])

    # Load the datasets
    # Databricks dolly general QA dataset
    ds = load_dataset("bergr7f/databricks-dolly-15k-subset-general_qa")
    # Amazon reviews dataset cleaned and filtered
    df = pd.read_csv(r"C:\Users\rajtu\OneDrive\Desktop\Raj\Datasets\amazon_reviews_us_Beauty_cleaned_filtered.csv")
    # Discarded reviews dataset
    ddf = pd.read_csv(r"C:\Users\rajtu\OneDrive\Desktop\Raj\Datasets\discarded_reviews.csv")

    # Assign weak labels to the reviews based on sentiments
    df['combined'] = df.review_headline + " " + df.review_body
    df.loc[df.star_rating > 3, 'weak_label'] = 1
    df.weak_label.fillna(0, inplace=True)

    # Group reviews on product id and combine 5 reviews for summarization based on weak sentiment
    n_size = 5
    df.groupby(by= ['product_id', 'weak_label']).progress_apply(lambda group: build_grouped_data(group))
    # Sample review (only 5000 entries) and remove combined reviews with more than 1024 words
    grouped_reviews['weights'] = grouped_reviews.weak_label.map({0: 0.8, 1: 0.2})
    grouped_reviews = grouped_reviews.sample(5000, weights=grouped_reviews.weights, random_state=92)
    grouped_reviews = grouped_reviews[grouped_reviews.review.apply(lambda x: len(x.split())) < 1024]

    # Load BART samsum model for summarization
    tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/bart-large-cnn-samsum", device_map="auto")
    grouped_reviews['summary'] = grouped_reviews.review.progress_apply(lambda row: get_summary(row))

    # Load sentiment classifier model
    sentiment_classifier = pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment-latest", tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest", max_length=512, truncation=True, device=device)
    # Get sentiment label and score for the summarized reviews
    sentiment_scores = grouped_reviews.progress_apply(lambda row: get_label_score(row['review']), axis=1, result_type='expand')
    sentiment_scores.columns = ['sentiment', 'sentiment_score']
    grouped_reviews = pd.concat([grouped_reviews, sentiment_scores], axis=1, ignore_index=False)

    # Filter reviews based on sentiment and sentiment score, keeping high confidence reviews
    grouped_reviews = grouped_reviews[(
        (grouped_reviews.sentiment == 'negative') & (grouped_reviews.sentiment_score > 0.7) |
        (grouped_reviews.sentiment == 'positive') & (grouped_reviews.sentiment_score > 0.85) |
        (grouped_reviews.sentiment == 'neutral') & (grouped_reviews.sentiment_score > 0.6)
    )]

    # Get random question based on sentiment
    grouped_reviews['question'] = grouped_reviews.progress_apply(lambda row: get_question(row['sentiment']), axis=1)
    grouped_reviews['summary']  = "I'm happy to help with your question about this product. Based on the reviews, here's what I found: " + grouped_reviews['summary']

    # Sample general QA dataset and general build QA data as negative data
    general_qa = ds['train'].to_pandas()
    qa_sample = general_qa.sample(round(0.1 * len(grouped_reviews))) # 10% of the grouped reviews size
    qa_sample.apply(lambda row: build_qa_data(row), axis=1)
    grouped_reviews = pd.concat([grouped_reviews, qa_df])

    # Negative samples for products with no reviews or fewer reviews
    questions = [POSITIVE_QUESTIONS, NEGATIVE_QUESTIONS, NEUTRAL_QUESTIONS]
    no_product_df = pd.DataFrame(columns=['product_id', 'review', 'summary', 'sentiment', 'sentiment_score',
           'question'])

    for i in range(300):
        no_product_df.loc[len(no_product_df)] = [
            np.nan,
            np.nan,
            "It seems that there aren't enough reviews available for this product at the moment to provide a detailed answer. I'm unable to assist with this specific query due to the limited feedback. Please check back later or try asking about a different product!",
            np.nan,
            np.nan,
            np.random.choice(questions[np.random.randint(0, 3)])]

    # Save final dataset for SFT
    grouped_reviews = pd.concat([grouped_reviews, no_product_df])
    grouped_reviews.to_csv(r"C:\Users\rajtu\OneDrive\Desktop\Raj\Datasets\llama_finetuning_reviews_qa_dataset.csv", index=False)