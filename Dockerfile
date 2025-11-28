FROM python
WORKDIR /app
COPY . /app

RUN pip install cowsay

# install poetry
RUN pip install poetry
RUN poetry install --no-root


CMD ["python3", "main.py"]
