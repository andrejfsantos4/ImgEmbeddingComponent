# A VGG16 gRPC service for image embedding
An image embedder gRPC service that uses a VGG16 network. Includes an example gRPC client that facilitates sending requests and parsing the responses. This service is also available as a **Docker image**.

Once launched, the server will load the network and wait for requests. The client is the one who reads the images, sends requests to the server and parses the responses.

## How to deploy the server
### With Docker (recommended):
In a terminal type `docker run -p 8061:8061 -it --rm andrejfsantos/img_embedding:latest`

### Without Docker:
In a terminal type `python img_embedding_server.py`

For this approach you'll need a Python3 environment with the packages listed in [requirements.txt](requirements.txt).

## How to deploy the client
In a terminal type `python test_client.py`

This client was made both as an example and also to facilitate the use of this service. Inside the [python client](test/test_client.py) you'll find an example of how you can use it in your own code.
