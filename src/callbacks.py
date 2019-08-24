from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score
from spacecutter.losses import CumulativeLinkLoss

from catalyst.dl.core import Callback, RunnerState
from catalyst.dl import CriterionCallback

class OrdinalCriterionCallback(CriterionCallback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "ordinal_loss",
        criterion_key: str = None,
        loss_key: str = None,
    ):
        super().__init__(
            input_key=input_key,
            output_key=output_key,
            prefix=prefix,
            criterion_key=criterion_key,
            loss_key=loss_key
        )
        self.criterion = CumulativeLinkLoss()

    def _compute_loss(self, state: RunnerState, criterion):
        logits = state.output[self.output_key]
        targets = state.input[self.input_key]

        loss = self.criterion(logits, targets)

        return loss

class SmoothCCECallback(CriterionCallback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "smooth_loss",
        criterion_key: str = None,
        loss_key: str = None,
        n_classes: int = 5,
        eps: float = 0.2,
    ):
        super().__init__(
            input_key=input_key,
            output_key=output_key,
            prefix=prefix,
            criterion_key=criterion_key,
            loss_key=loss_key
        )
        self.n_classes = n_classes
        self.eps = eps


    def smoothCCE(self, y_pred, y_true):
        #y_true = y_true.contiguous().view(-1)
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()

        if self.eps > 0:
            log_prb = F.log_softmax(y_pred, dim=1)

            soft_one_hot = F.one_hot(y_true, num_classes=self.n_classes).float()
            soft_one_hot = soft_one_hot * (1 - self.eps)
            for i in range(soft_one_hot.shape[0]):
                if y_true[i] == 0:
                    soft_one_hot[i][1] += self.eps
                elif y_true[i] == 4:
                    soft_one_hot[i][4] += self.eps
                else:
                    soft_one_hot[i][y_true[i]-1] += self.eps/2
                    soft_one_hot[i][y_true[i]+1] += self.eps/2
            loss = -(soft_one_hot * log_prb).mean()
        else:
            loss = F.cross_entropy(y_pred, y_true, reduction='mean')

        return loss

    def _compute_loss(self, state: RunnerState, criterion):
        logits = state.output[self.output_key]
        targets = state.input[self.input_key]

        loss = self.smoothCCE(logits, targets)

        return loss


class KappaCriterionCallback(CriterionCallback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "kappa_loss",
        criterion_key: str = None,
        loss_key: str = None,
        scale: int = 2,
        n_classes: int = 5,
    ):
        super().__init__(
            input_key=input_key,
            output_key=output_key,
            prefix=prefix,
            criterion_key=criterion_key,
            loss_key=loss_key
        )
        self.scale = scale
        self.n_classes = n_classes
        self.weights = torch.arange(start=0, end=self.n_classes, dtype=torch.float) / (self.n_classes - 1)
        self.weights = (self.weights - torch.unsqueeze(self.weights, -1)) ** 2
        self.weights = self.weights.to(torch.device('cuda'))


    def qwk(self, y_pred, y_true):
        y_pred = F.softmax(y_pred.squeeze(), dim=-1)
        y_true_ohe = F.one_hot(y_true.long().squeeze(), num_classes=self.n_classes).float()

        hist_true = torch.sum(y_true_ohe, dim=0)
        hist_pred = torch.sum(y_pred, dim=0)
        
        E = hist_true.unsqueeze(dim=-1) * hist_pred
        E /= E.sum()
        
        O = torch.mm(y_pred.transpose(1, 0), y_true_ohe)
        O /= O.sum()
        
        num = self.weights * O
        den = self.weights * E
        
        QWK = (1 - num.sum() / den.sum())
        return QWK

    def _compute_loss(self, state: RunnerState, criterion):
        logits = state.output[self.output_key]
        targets = state.input[self.input_key]

        QWK = self.qwk(logits, targets)
        loss = -F.logsigmoid(self.scale * QWK)

        return loss


class KappaCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "kappa",
        class_names: List[str] = None,
        num_classes: int = 1,
        regression: bool = True,
    ):
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key

        self.class_names = class_names
        self.num_classes = num_classes \
            if class_names is None \
            else len(class_names)
        
        self.regression = regression
        assert self.num_classes is not None

    def _reset_stats(self):
        self.y_true = np.empty(0)
        self.y_pred = np.empty(0)
    
    def _round_preds(self, preds):
        coef = [0.5, 1.5, 2.5, 3.5]
        rounded_preds = np.zeros(len(preds), dtype=np.int)

        for i, pred in enumerate(preds):
            if pred < coef[0]:
                rounded_preds[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                rounded_preds[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                rounded_preds[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                rounded_preds[i] = 3
            else:
                rounded_preds[i] = 4
        return rounded_preds

    def on_loader_start(self, state: RunnerState):
        self._reset_stats()

    def on_batch_end(self, state: RunnerState):
        
        if self.regression is True:
            output = state.output[self.output_key].detach().cpu().numpy().ravel()
            rounded_preds = self._round_preds(output)
            self.y_pred = np.concatenate((self.y_pred, rounded_preds))
        else:
            output = state.output[self.output_key].detach().cpu().numpy().squeeze()
            preds = output.argmax(axis=1)
            self.y_pred = np.concatenate((self.y_pred, preds))
        targets = state.input[self.input_key].detach().cpu().int().numpy().ravel()
        self.y_true = np.concatenate((self.y_true, targets))
        
    def on_loader_end(self, state: RunnerState):
        
        score = cohen_kappa_score(self.y_true, self.y_pred, labels=range(self.num_classes), weights='quadratic')
        metric_name = f"{self.prefix}_score"
        state.metrics.epoch_values[state.loader_name][metric_name] = score

        self._reset_stats()
