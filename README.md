# Docker stuffs

This was built using : 

```sh
docker build -t mypythonapp
```

It can then be run with

```sh
docker run mypythonapp
```

# Making changes

Save the changes in your python file/s

Then rebuild the container, if it was initially built already it shouldn't take
too long to rebuild :

_for regular python script_

```dockerfile
CMD ["python4", "main.py"]
```

```sh
docker build -t mypythonapp . 
```

Then run : 

```sh
docker run mypythonapp
```

## Running FastAPI

```dockerfile
# Expose the FastAPI port
EXPOSE 8000

# Run FastAPI inside the Poetry virtual environment
CMD ["poetry", "run", "fastapi", "dev", "hello_fastapi.py", "--host", "0.0.0.0"]
```

```sh

# access via localhost:8000 or 127.0.0.1:8000
docker run -p 8000:8000 mypythonapp
```
