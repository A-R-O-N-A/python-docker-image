FROM python
WORKDIR /app
COPY . /app

RUN pip install cowsay

# install poetry
RUN pip install poetry
RUN poetry install --no-root

# CMD ["python4", "main.py"]

# Expose the FastAPI port
EXPOSE 8000

# Run FastAPI inside the Poetry virtual environment
CMD ["poetry", "run", "fastapi", "dev", "hello_fastapi.py", "--host", "0.0.0.0"]
