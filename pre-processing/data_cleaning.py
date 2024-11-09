import pandas as pd
import csv
import nltk
from nltk.stem import WordNetLemmatizer 
import emoji
import re
from utilities.constants import CONTRACTIONS
nltk.download('wordnet')


def preprocess(text):
    """
    Pre-process the text.
    Removal of special characters, lower the text, expand contractions, remove emojis, remove html tags
    replace & with and, replace - with space, replace % with percent, replace ... with ., remove symbols except '.!?,',
    remove multiple whitespace
    :param text: text to pre-process
    :return: cleaned text
    """

    try:
        #remove html tags
        text = re.sub(r'<[^>]*>', '', text)
        #lowercase
        text = text.lower()
        #expand contractions
        text = " ".join([CONTRACTIONS[word] if word in CONTRACTIONS.keys()  else word for word in text.split()])
        #remove stop words
        #remove emojis
        text = emoji.replace_emoji(text, '')
        #replace & with and
        text = text.replace('&', 'and')
        #replace - with space
        text = text.replace('-', ' ')
        #replace % with percent
        text = text.replace('%', 'percent')
        #replace ... with .
        text = re.sub(r'\.+', '.', text)
        #replace \ with ''
        text = re.sub(r'\\+', '', text)
        #replace \ with ''
        text = re.sub(r'!+', '!', text)
        #remove symbols except '.!?,'
        text = re.sub('[^\w\s.!,?\'\"]', ' ', text)
        #remove multiple whitespace
        text = re.sub(r'\s+',' ', text)

    except Exception as e:
        print("ERROR", e)
        return
        
    return text

# Load the data
df = pd.read_csv(r"C:\Users\rajtu\OneDrive\Desktop\Raj\Datasets\amazon_reviews_us_Beauty_v1_00.tsv", sep="\t", quoting=csv.QUOTE_NONE)
df.dropna(subset= ['review_headline', 'review_body', 'product_title'], inplace= True)

# Apply pre-processing
df.review_body = df.review_body.progress_apply(lambda text: preprocess(text))
df.review_headline = df.review_headline.progress_apply(lambda text: preprocess(text))
df.product_title = df.product_title.progress_apply(lambda text: preprocess(text))

df.to_csv(r"C:\Users\rajtu\OneDrive\Desktop\Raj\Datasets\amazon_reviews_us_Beauty_v1_00_cleaned.tsv", index= False)

