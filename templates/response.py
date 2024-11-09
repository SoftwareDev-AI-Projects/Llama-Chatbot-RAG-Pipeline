from pydantic import BaseModel

class QueryResponseModel(BaseModel):
    answer: str = ""
    status: bool = False
    message: str = ""
