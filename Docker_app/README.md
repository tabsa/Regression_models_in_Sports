# Docker Container application for product development
This folder has an example on how to build a Docker container with Flask for applications in software development.

## 1. Create the Docker image
You need two essential files:
- Dockerfile that specifies the image, see the [example](Dockerfile.prod)
- Requirements that indicates all dependencies (like python libraries), see the [example](requirements.txt)

## 2. Build the Docker image
After you have the files ready, you use the following command:
```
sudo docker build -f Dockerfile.prod -t <app_name> .
```
You need to use the command line.

## 3. Deploy the Docker container
