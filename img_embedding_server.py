from concurrent import futures
import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection

import img_embedding_pb2 as pb
import img_embedding_pb2_grpc as pb_grpc
import logging
import numpy as np
import io
import PIL
import torch
from torchvision import models, transforms


class ImgEmbeddingServicer(pb_grpc.ImgEmbeddingServicer):
    """Provides methods that implement functionality of image embedding server."""

    def __init__(self) -> None:
        logging.info("Loading VGG16...")
        self.full_model = models.vgg16()
        # Load the pretrained weights from a file
        self.full_model.load_state_dict(torch.load("vgg16-397923af.pth"))
        self.model = self.full_model.features

    def GetImgEmbedding(self, request: pb.Image, context) -> pb.Float1DArray:
        """Returns an embedding for an image using a pretrained VGG16."""

        # Check that the request is non-empty
        logging.info("Received request. Getting image embedding...")
        if not request or not request.data:
            logging.error("Received empty request.")
            return pb.Float1DArray()

        # Decode the image from bytes
        img_bytes = request.data
        img = PIL.Image.open(io.BytesIO(img_bytes))

        # Load pretrained VGG16 model from file
        # full_model = models.vgg16(pretrained=True)

        # Prepare image and model for inference
        img_transforms = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

        img = img_transforms(img).unsqueeze(0)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            img = img.cuda()
        self.model.eval()

        # Inference
        with torch.no_grad():
            features = torch.flatten(self.model(img)).cpu().numpy()

        return array_to_msg(features)


def array_to_msg(line: np.ndarray) -> pb.Float1DArray:
    """Returns a message of type Float1DArray from a numpy array."""
    array_msg = pb.Float1DArray()
    array_msg.elems.extend(line.tolist())
    return array_msg


def serve():
    server = grpc.server(futures.ThreadPoolExecutor())
    pb_grpc.add_ImgEmbeddingServicer_to_server(
        ImgEmbeddingServicer(), server)

    # Add reflection
    service_names = (
        pb.DESCRIPTOR.services_by_name['ImgEmbedding'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    server.add_insecure_port('[::]:8061')
    server.start()
    logging.info("Successfully started and waiting for connections..")
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(format='Server %(levelname)s: %(message)s', level=logging.INFO)
    serve()
