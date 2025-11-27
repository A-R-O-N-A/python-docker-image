FROM python
WORKDIR /app
COPY . /app


RUN pip install cowsay

CMD ["python3", "main.py"]
