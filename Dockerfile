FROM python
WORKDIR /app
COPY . /app

# RUN pip install cowsay

# install poetry
RUN pip install poetry
RUN poetry install --no-root

# CMD ["python4", "main.py"]

# Expose the FastAPI port
EXPOSE 8000

# Run FastAPI inside the Poetry virtual environment

# access via localhost:8000 or 127.0.0.1:8000
# docker run -p 8000:8000 mypythonapp

CMD ["poetry", "run", "fastapi", "dev", "main.py", "--host", "0.0.0.0"]
