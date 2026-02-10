"""
    svm.py
    To run an svm to fit based on features from a neural network
"""

import joblib
import numpy as np
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from transformers import PreTrainedModel
import torch
import torch.nn as nn
from torch.utils.data import (
    Dataset,
    DataLoader
)
from tqdm.auto import tqdm

def make_pad_collate(
    x_key="input",          # key in each sample dict for the tensor to batch
    y_key="label",          # optional key for labels (set to None if none)
    time_dim=-1,            # which axis is time in x: -1 for [C,T] or [n_mfcc,T]; 0 for [T,C] or [T]
    pad_value=0.0,          # value to pad with
    pad_to=None,            # optionally pad up to a multiple (e.g., product of conv strides)
    max_len=None,           # optionally cap/truncate to this many time steps
    return_lengths=True,
    return_attention_mask=True,
):
    def _collate(batch):
        xs = [b[x_key] for b in batch]
        ys = None if y_key is None else torch.as_tensor([b[y_key] for b in batch])

        # Move time to axis 0 => [T, ...]
        xs_Tfirst = []
        for x in xs:
            if x.dim() == 1:
                xT = x  # [T]
            else:
                if time_dim == -1:
                    xT = x.transpose(0, -1).contiguous()  # [..., T] -> [T, ...]
                elif time_dim == 0:
                    xT = x
                else:
                    raise ValueError("time_dim must be 0 or -1")
            xs_Tfirst.append(xT)

        lengths = torch.as_tensor([xT.shape[0] for xT in xs_Tfirst], dtype=torch.long)
        L = int(lengths.max().item())

        # Optional: round up to a multiple (useful for conv/pooling stacks)
        if pad_to:
            L = int(math.ceil(L / pad_to) * pad_to)
        if max_len is not None:
            L = min(L, max_len)

        padded, masks = [], []
        for xT, l in zip(xs_Tfirst, lengths):
            # Trim if needed
            if xT.shape[0] > L:
                xT = xT[:L]
                l = torch.tensor(L)

            # Pad to L
            pad_len = L - xT.shape[0]
            if pad_len > 0:
                pad_shape = (pad_len,) + xT.shape[1:]
                pad = xT.new_full(pad_shape, pad_value)
                xT = torch.cat([xT, pad], dim=0)

            padded.append(xT)
            if return_attention_mask:
                m = torch.zeros(L, dtype=torch.bool)
                m[:l.item()] = True
                masks.append(m)

        # Stack and move time back to original place
        X = torch.stack([
            p if (p.dim() == 1 or time_dim == 0) else p.transpose(0, -1).contiguous()
            for p in padded
        ], dim=0)  # -> [B, T] or [B, C, T] (if time_dim == -1) or [B, T, C] (if time_dim == 0)

        out = {x_key: X}
        if ys is not None:
            out[y_key] = ys
        if return_lengths:
            out["lengths"] = lengths
        if return_attention_mask:
            out["attention_mask"] = torch.stack(masks, dim=0)  # [B, L] (True = real, False = pad)
        return out
    return _collate

class NeuralSVM(nn.Module):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    config = None

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.svm = None
        self.activations = []
        self.last_activation = None
        self.model = model
        self.selector = None
        self.classifier_layer = 0

    def fit(self, dataset: Dataset):
        self.model.eval()

        self.svm = svm.SVC()
        self.activations = []

        try:
            hook_handle = self.model.classifier[self.classifier_layer].register_forward_hook(self._save_activations)
        except:
            hook_handle = self.model.model.classifier[self.classifier_layer].register_forward_hook(self._save_activations)

        labels = []
        for data in tqdm(DataLoader(dataset, batch_size=32), ncols=120, desc="Fitting SVM"):
            input_vals = data["input_vals"]
            input_vals = input_vals.to(self.DEVICE)

            label = data["label"]
            label = [int(1 if i.split('.')[1] == '1' else 0) for i in label]
            labels.extend(label)

            #if self.model.backbone.config.model_type in ["energy_transformer_mfcc"]:
            #    input_vals = self.model.backbone.preprocess_mfcc_multichannel_batch(input_vals)
            #    self.model(input_vals, torch.ones(input_vals.shape[0], input_vals.shape[-1], dtype=torch.int64, device=input_vals.device), label, run_criterion=False)
            #else:
            self.model(input_vals, label)

        self.selector = SelectKBest(k=80)
        selected_features = self.selector.fit_transform(np.asarray(self.activations), labels)
        self.svm.fit(selected_features, labels)

        hook_handle.remove()

        return self

    def forward(self, input_vals, labels, **kwargs):
        if self.svm is None:
            raise ValueError("The svm has not yet been fit!")

        self.model.eval()

        hook_handle = self.model.classifier[self.classifier_layer].register_forward_hook(self._save_activations)

        #if self.model.backbone.config.model_type in ["energy_transformer_mfcc"]:
            #input_vals = self.model.backbone.preprocess_mfcc_multichannel_batch(input_vals)
            #self.model(input_vals, torch.ones(input_vals.shape[0], input_vals.shape[-1], dtype=torch.int64, device=input_vals.device), labels, run_criterion=False)
        #else:
        nn_prediction = self.model(input_vals, labels)
        out = torch.tensor(self.svm.predict(self.selector.transform(self.last_activation)))

        hook_handle.remove()

        return out

    def _save_activations(self, module, input, output):
        activation = output
        self.activations.extend(activation.cpu().detach().numpy())
        self.last_activation = activation.cpu().detach().numpy()

    def save_model(self, path: str):
        if self.svm is None or self.selector is None:
            raise ValueError("Nothing to save: fit() must be called before save_model().")

        payload = {
            "svm": self.svm,
            "selector": self.selector,
            "classifier_layer": self.classifier_layer,
        }
        joblib.dump(payload, path)

    def load_model(self, path: str, model: PreTrainedModel):
        payload = joblib.load(path)

        self.svm = payload["svm"]
        self.selector = payload["selector"]
        self.classifier_layer = payload.get("classifier_layer", 0)

        self.model = model
        return self
