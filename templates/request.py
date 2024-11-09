from pydantic import BaseModel

class QueryRequestModel(BaseModel):
    question: str = "What are the good products in the market?"
    product_name: str = ""
    uuid: str = ""
