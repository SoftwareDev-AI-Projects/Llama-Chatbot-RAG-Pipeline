import logging
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from rag.rag_pipeline import RAGPipeline
from templates.response import QueryResponseModel
from templates.request import QueryRequestModel
from dotenv import load_dotenv

load_dotenv()
data_path = os.getenv("DATA_PATH")
app = FastAPI()
pipe = RAGPipeline(data_path)
logger = logging.getLogger(__name__)

@app.post("/query", response_model=QueryResponseModel)
async def query(request: QueryRequestModel):
    """
    This function is the entry point for the API. It receives a request packet, extracts the question and product name
    and invokes the RAG pipeline to get the answer. The answer is then returned in the response packet.
    :param request: request packet received from the client
    :return: response packet as json
    """
    logger.log(logging.INFO, "Received a request")
    question = request.question
    product_name = request.product_name

    if not question:
        logger.log(logging.ERROR, "Question is empty in the request packet")
        raise HTTPException(status_code=500, detail="Question cannot be empty")

    logger.log(logging.INFO, f"Invoking RAG pipeline for the question: {question}")
    answer = pipe.get_answer(question, product_name)
    response = {"answer": answer, "status": True, "message": "Success"}
    logger.log(logging.INFO, f"Returning response: {response}")

    return JSONResponse(content=response)


if __name__ == "__main__":
    logger.log(logging.INFO, "Starting the server")
    print("Starting the server")
    uvicorn.run(app, host="0.0.0.0", port=8080)