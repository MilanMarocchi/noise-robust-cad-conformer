"""
    model_factory.py
    Author: Milan Marocchi

    Purpose: To create models
"""

from typing import Optional
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)
from torch.optim import lr_scheduler
import os
import torch.optim as optim

from models.transformers.mfcconformer import MFCConformer



HERE = os.path.abspath(os.getcwd())

def get_optimizer_and_scheduler(
        params, 
        optimizer_type: str, 
        lr: Optional[float] = None,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 1e-5,
        momentum: float = 0.9,
        step_size: int = 7*1000,
        gamma: float = 0.1,
        **kwargs
    ):
    """
    Returns the required optimiser.
    """
    fun = lambda epoch: gamma ** (epoch // step_size) 

    if lr is None:
        if optimizer_type == 'sgd':
            lr = 0.001
        elif optimizer_type == 'adam':
            lr = 1e-4
        elif optimizer_type == 'adamw':
            lr = 1e-5
        elif optimizer_type == "rmsprop":
            lr = 1e-4
        else:
            lr = 1e-5
    
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        return optimizer, lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(params, lr=lr, betas=(adam_beta1, adam_beta2), eps=adam_epsilon, weight_decay=weight_decay)
        return optimizer, None #lr_scheduler.LambdaLR(optimizer, lr_lambda=fun)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        return optimizer, lr_scheduler.LambdaLR(optimizer, lr_lambda=fun)
    elif optimizer_type == "rmsprop":
        optimizer = optim.RMSprop(params, lr=lr)
        return optimizer, lr_scheduler.LambdaLR(optimizer, lr_lambda=fun)
    else:
        raise NotImplementedError(f"Invalid Optimiser: {optimizer_type}")

def get_standard_model(device, config: PretrainedConfig, model_class: PreTrainedModel):
    """Creates a standard model for training"""
    model = model_class(config)
    model = model.to(device)

    return model


def get_single_models():
    return get_audio_models() + get_vision_models()

def get_vision_models():
    return (
    )

def get_audio_models():
    return (
        "mfcconformer",
    )

def get_multi_models():
    return (
    )

def get_unsupervised_models():
    return (
    )

class ModelFactory():
    """
    Model factory to make it easier to change between models and stuff
    """

    def __init__(self, device, class_names, freeze=False, optimizer_type='sgd', optimizer_params=None):
        self.device = device
        self.class_names = class_names
        self.freeze = freeze
        self.optimizer_type = optimizer_type
        self.optimizer_params = optimizer_params

        self.model_classes = self._get_models()
        self.audio_model = get_audio_models()
        self.single_models = get_single_models()
        self.multi_models = get_multi_models()


    def _get_models(self):
        return {
            "mfcconformer": MFCConformer,
        }

    def get_config(self, 
            model_code: str, 
            config: dict, 
            models:Optional[list[PreTrainedModel]] = None,
            aux_model_code:Optional[str] = None
        ) -> PretrainedConfig:
        """
        Creates the models config based on the model code and the config dictionary provided
        """
        if model_code == "ensemble" and (models is None or aux_model_code is None):
            raise ValueError("Must provide models for the ensemble")
        try:
            config["aux_model_type"] = aux_model_code
            return self.model_classes[model_code].config_class(**config)
        except KeyError:
            raise ValueError(f"Invalid model code: {model_code}")

    def get_class(self,
            model_code: str, 
    ):
        try:
            return self.model_classes[model_code]
        except KeyError:
            raise ValueError(f"Invalid model code: {model_code}")

    def create_model(self, 
        model_code: str, 
        config: dict, 
        models: Optional[list[PreTrainedModel]] = None, 
        aux_model_code: Optional[str] = None,
        freeze_extractor: bool = False
    ) -> PreTrainedModel:
        """
        Creates the model specified by the model_code
        """
        model_config = self.get_config(model_code, config, models, aux_model_code)
        print(f"Model config: {model_config}")

        if model_code in self.single_models:
            model_class = self.get_class(model_code)
            model = get_standard_model(self.device, model_config, model_class)
            return model
        else:
            raise ValueError(f"Invalid model: {model_code=}")