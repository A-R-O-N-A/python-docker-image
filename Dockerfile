FROM python
WORKDIR /app
COPY . /app

RUN pip install cowsay
RUN pip install poetry

CMD ["python3", "main.py"]
