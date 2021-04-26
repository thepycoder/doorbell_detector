FROM python:3.7

RUN apt-get install -y llvm-9 llvm-9-dev make g++

# Copy requirements first. Docker caches every step so this way we don't have to reinstall every pip package every time our code changes
COPY requirements.txt /
RUN LLVM_CONFIG=/usr/bin/llvm-config-9 pip3 install --extra-index-url=https://www.piwheels.org/simple --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

ENTRYPOINT [ "gunicorn", "bell_dash:SERVER" ]