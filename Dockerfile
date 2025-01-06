FROM python:3.11.9

RUN apt-get update
RUN apt-get install -y \
    build-essential \
    curl
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

ENV PYTHONDONTWRITEBYCODE=1
ENV PYTHONBUFFERED=1

WORKDIR /app

COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

EXPOSE 8080

CMD python -m uvicorn main:app --host=0.0.0.0 --port=8080
