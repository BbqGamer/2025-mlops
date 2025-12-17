import bentoml
from PIL import Image
from transformers import pipeline


# Create the service
@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 60},
)
class ResNetService:
    def __init__(self):
        self.classifier = pipeline("image-classification", model="microsoft/resnet-18")

    @bentoml.api
    def predict(self, img: Image.Image) -> dict:
        results = self.classifier(img)
        return {"predictions": results}
