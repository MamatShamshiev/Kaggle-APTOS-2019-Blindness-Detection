from typing import Mapping, Any
from catalyst.dl import Runner
import torch
import numpy as np

class ModelRunner(Runner):
    def predict_batch(self, batch: Mapping[str, Any]):
        output = self.model(batch["image"])
        batch["targets"] = batch["targets"].type(torch.cuda.FloatTensor).view(-1, 1)
        #batch["targets"] = batch["targets"].view(-1, 1)
        return {"logits": output}