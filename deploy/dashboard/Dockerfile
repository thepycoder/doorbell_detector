FROM balenalib/raspberry-pi-python:3.7.4

# Copy requirements first. Docker caches every step so this way we don't have to reinstall every pip package every time our code changes
COPY requirements.txt /
RUN pip3 install --extra-index-url=https://www.piwheels.org/simple --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

ENTRYPOINT [ "gunicorn", "-c", "gunicorn.config.py", "bell_dash:SERVER" ]