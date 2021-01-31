# Docker Container application for product development
This folder has an example on how to build a Docker container with Flask for applications in software development.

## 1. Create the Docker image
You need two essential files:
- Dockerfile that specifies the image, see this [Dockerfile](Dockerfile.prod);
- Requirements that indicates all dependencies (like python libraries), see this [example.txt](requirements.txt).

## 2. Build the Docker image
After you have the files ready, you use the following command:
```
sudo docker build -f Dockerfile.prod -t <app_name> .
```
You need to use the command line.

## 3. Deploy the Docker container
First, you start by deploying the Docker container in the background:
```
sudo docker run -d --name <container_name> image_name
```
Second, check the docker contrainers created so far:

```
sudo docker ps --all
```
Finally, run the Docker you have created:

```
sudo docker start -a -i <container_name>
```
Use the command line.

In case you want to stop the container, it is the same command above with `stop` instead of `start`. Removing a container or image:
- Container: `sudo docker rm <container_name>`;
- Image: `sudo docker image rm <image_name>`.
