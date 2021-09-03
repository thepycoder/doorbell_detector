FROM python:3.7.4

# Copy requirements first. Docker caches every step so this way we don't have to reinstall every pip package every time our code changes
COPY requirements.txt /
RUN pip3 install -r requirements.txt

COPY . /app
WORKDIR /app

ENV CONTAINER=True

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]