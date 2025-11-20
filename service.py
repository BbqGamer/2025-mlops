from __future__ import annotations

import bentoml
import numpy as np
import torch
from PIL.Image import Image

best_model = "lightning/mlruns/1/af64107168ff4519aac6a627e956b2e0/artifacts/epoch=6-step=1372/epoch=6-step=1372.ckpt"


@bentoml.service
class MnistClassifierService:
    def __init__(self):
        from lightning.train import LitClassifier

        self.model = LitClassifier(hidden_dim=196)
        chpt = torch.load(best_model, map_location="cpu")
        self.model.load_state_dict(chpt["state_dict"])

    @bentoml.api
    def predict(self, images: list[Image]) -> list[int]:
        self.model.eval()
        input_tensor = torch.tensor(np.array(images)).float()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            preds = outputs.argmax(dim=1)
        return preds.cpu().numpy().tolist()
