import bentoml
import os
import torch
from typing import List, Dict

model_path = 'leaknet.pt'

def load_model():
    global model, device

    if not os.path.isfile(model_path):
        raise RuntimeError(f"Missing the model file: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.jit.load(model_path, map_location=device)
    model.to(device)
    model.eval()

@bentoml.service
class LeakDetection:
    def __init__(self) -> None:
        load_model()
        
    def preprocess(self, data: List[Dict]):
        data = [[sample["pressure"], sample["flow"]] for sample in data]
        data = torch.tensor(data).unsqueeze(0)
        return data 

    @bentoml.api()
    def detect(self, data: List[Dict]) -> float:
        data = self.preprocess(data)
        
        with torch.inference_mode():
            output = model.predict_step(data).squeeze(0)
        
        return output.cpu().numpy().tolist()[0]
