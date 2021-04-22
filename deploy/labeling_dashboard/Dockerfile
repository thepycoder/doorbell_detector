FROM python:3.7

# Copy requirements first. Docker caches every step so this way we don't have to reinstall every pip package every time our code changes
COPY requirements.txt /
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

ENTRYPOINT [ "gunicorn", "bell_dash:SERVER" ]