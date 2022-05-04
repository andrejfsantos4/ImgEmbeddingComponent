from os import path
import logging
from typing import Optional

import grpc
import numpy as np
from numpy import ndarray

import img_embedding_pb2 as pb
import img_embedding_pb2_grpc as pb_grpc


class ImageEmbedder:
    """
    Establishes a gRPC channel with a service that embeds images using a pretrained VGG16 network. The embedding
    is taken from the output of the last max-pooling layer, which has dimension 7x7x512, resulting in an embedding of
    size 25088.
    """
    def __init__(self):
        # Configure prints to screen
        logging.basicConfig(format='Client %(levelname)s: %(message)s', level=logging.INFO)
        # Create the channel to communicate with the server
        self.channel = grpc.insecure_channel('localhost:8061')
        self.estimator_stub = pb_grpc.ImgEmbeddingStub(self.channel)

    def send_request_client(self, image_path: str) -> Optional[ndarray]:
        """
        Sends a request to the server with an image and receives the corresponding embedding.
        :param image_path: path to image to be processed
        :return: Image embedding as a numpy array with size (25088, ), or None is unable to read image
        """
        logging.info("Reading image...")
        if not path.isfile(image_path):
            logging.error("Provided path is not a file. Exiting..")
            return None

        with open(image_path, 'rb') as fp:
            image_bytes = fp.read()
        request = pb.Image(data=image_bytes)

        # Send the request and save the response
        logging.info("Sending request...")
        try:
            response_msg = self.estimator_stub.GetImgEmbedding(request)
        except grpc.RpcError as rpc_error:
            print('An error has occurred:')
            print(f'  Error Code: {rpc_error.code()}')
            print(f'  Details: {rpc_error.details()}')
            return None

        # Decode the response message into a numpy array
        embedding = np.array(response_msg.elems)
        return embedding


# The following code exemplifies how you can obtain an embedding in your own code, after importing ImageEmbedder
if __name__ == '__main__':

    embedder = ImageEmbedder()
    embedding = embedder.send_request_client("test/test_img.jpg")
    print(embedding)
