from catalyst.dl import registry


from .experiment import Experiment
from .runner import ModelRunner as Runner
from .callbacks import KappaCallback, KappaCriterionCallback, SmoothCCECallback, OrdinalCriterionCallback
from .model import resnet34, resnet34_pretrained, resnext50_pretrained, efficientnet_pretrained, cadene_model, \
                   ordinal_efficientnet

registry.Model(resnet34)
registry.Model(resnet34_pretrained)
registry.Model(resnext50_pretrained)
registry.Model(efficientnet_pretrained)
registry.Model(cadene_model)
registry.Model(ordinal_efficientnet)

registry.Callback(KappaCallback)
registry.Callback(KappaCriterionCallback)
registry.Callback(SmoothCCECallback)
registry.Callback(OrdinalCriterionCallback)