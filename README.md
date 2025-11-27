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

```sh
docker build -t mypythonapp
```

Then run : 

```sh
docker run mypythonapp
```
