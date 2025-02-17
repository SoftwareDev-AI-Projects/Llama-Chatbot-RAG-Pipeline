POSITIVE_QUESTIONS = [
    "What do customers love most about this product?",
    "How is the user experience with this product?",
    "Why do people think this product is a great value for money?",
    "What features of this product make it stand out from competitors?",
    "How does the build quality of this product impress buyers?",
    "What positive feedback do customers give about the design?",
    "Why is this product highly recommended by users?",
    "What aspects of the product are customers most satisfied with?",
    "What do buyers appreciate about the product's performance?",
    "Why do users consider this product a must-have?",
    "How does this product exceed customer expectations?",
    "What do reviewers say about the durability of this product?",
    "Why do customers praise the ease of use of this product?",
    "What makes this product a popular choice among buyers?",
    "How does the product meet the needs of its users?",
    "What are the top benefits mentioned by customers?",
    "Why do buyers think this product is worth the price?",
    "How do customers describe their satisfaction with this product?",
    "What positive experiences have customers shared?",
    "Why do customers believe this product is a great investment?"
]


NEGATIVE_QUESTIONS = [
    "What issues do customers frequently mention about this product?",
    "Why are some buyers disappointed with their purchase?",
    "What are the most common complaints about this product?",
    "How does the product fail to meet customer expectations?",
    "Why do some users consider this product to be poor quality?",
    "What negative feedback do customers give about the design?",
    "What aspects of this product do customers find frustrating?",
    "How does the product under perform according to reviews?",
    "Why do customers think this product is not worth the money?",
    "What are the biggest drawbacks mentioned by buyers?",
    "How does this product fall short in terms of durability?",
    "What do customers dislike about the user interface?",
    "Why are some users unhappy with the product's performance?",
    "What problems do reviewers highlight about this product?",
    "What makes customers regret buying this product?",
    "Why do customers say this product doesn't work as advertised?",
    "What do customers criticize most about this product?",
    "How do buyers describe their dissatisfaction with this product?",
    "What are the major flaws mentioned in customer reviews?",
    "Why do customers think this product needs improvement?"
]


NEUTRAL_QUESTIONS = [
    "What features does this product offer?",
    "How does this product compare to similar options?",
    "What do reviewers say about the product's specifications?",
    "How do customers describe the overall design of this product?",
    "What are the key characteristics of this product?",
    "How does the price of this product compare to others in the market?",
    "What do buyers mention about the setup process?",
    "How does the product function in everyday use?",
    "What do reviewers think about the customer service for this product?",
    "How is the product packaging described by customers?",
    "What are some common themes in the reviews?",
    "How do customers view the value of this product?",
    "What are some use cases mentioned for this product?",
    "How does this product perform over time?",
    "What do buyers think about the brand behind this product?",
    "What kind of improvements do customers suggest?",
    "How do customers describe their overall experience with this product?",
    "What do users think of the product's compatibility with other devices?",
    "What are the pros and cons of this product according to reviews?",
    "How do reviewers describe the size and weight of this product?"
]

CONTRACTIONS = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how i",
"i'd": "I would",
"i'd've": "I would have",
"i'll": "I will",
"i'll've": "I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have",
"dont": "do not",
"wont": "will not",
"cant": "can not",
"aint": "is not"
}