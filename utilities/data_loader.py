from typing import Iterator, AsyncIterator

import pandas as pd
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

class CSVLoader(BaseLoader):
    def __init__(self, path):
        self.file_path = path
        self.cols = ['product_id', 'product_title', 'star_rating', 'helpful_votes',
       'total_votes', 'vine', 'verified_purchase', 'review_headline',
       'review_body']

    def lazy_load(self) -> Iterator[Document]:
        df = pd.read_csv(self.file_path)
        for i in range(len(df)):
            row = df.iloc[i]
            yield Document(
                page_content= row['review_headline'] + " " + row['review_body'],
                metadata= {key: row[key] for key in self.cols[:-2]}
            )



