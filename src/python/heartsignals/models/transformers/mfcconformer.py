from datetime import datetime
import logging
import time
import librosa
import numpy as np
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
)
from learners.loss_functions import CenterLoss, ContrastiveFocalLoss
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torch
import matplotlib.pyplot as plt

import torch
import torchaudio
import torchaudio.functional as F_audio
from processing.filtering import kpeak_normalise_signal_torch


def preprocess_mfcc_multichannel_batch(
    waveforms,  # shape: [B, T, C]
    sample_rate=4000,
    n_mfcc=128,
    melkwargs=None,
    device='cuda'
):
    waveforms = waveforms.to(device)  # [B, T, C]
    waveforms = waveforms.transpose(1, 2)  # [B, C, T]

    if melkwargs is None:
        melkwargs = {
            "n_fft": 512,
            "hop_length": 160,
            "n_mels": 128,
            "f_min": 20,
            "f_max": 450,
            "win_length": 512,
            "window_fn": torch.hann_window
        }

    B, C, Ti = waveforms.shape
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs=melkwargs
    ).to(device)

    # Process each channel independently, then concatenate along feature axis
    mfcc_list = []
    for c in range(C):
        waveform = kpeak_normalise_signal_torch(waveforms[:, c], k=26*4)  # [B, T]
        mfcc_c = mfcc_transform(waveform)  # [B, n_mfcc, T']

    # Concatenate MFCCs along feature (mel) axis: [B, C * n_mfcc, T']
    mfccs = torch.cat(mfcc_list, dim=1)
    return mfccs

def preprocess_mfcc_batch(
    waveforms,  # shape: [B, T]
    sample_rate=4000,
    n_mfcc=128,
    melkwargs=None,
    device='cuda'
):
    """
    Compute MFCCs in batch using torchaudio.transforms.MFCC and apply z-normalization.

    Args:
        waveforms (Tensor): A batch of waveforms with shape [B, T]
        sample_rate (int): Sampling rate
        n_mfcc (int): Number of MFCC coefficients to return
        melkwargs (dict): kwargs passed to the internal MelSpectrogram
        target_time_steps (int): Desired number of time steps after resizing
        device (str): Device to compute on

    Returns:
        Tensor of shape [B, n_mfcc, target_time_steps]
    """
    waveforms = waveforms.to(device)

    if melkwargs is None:
        melkwargs = {
            "n_fft": 512,
            "hop_length": 160,
            "n_mels": 128,
            "f_min": 20,
            "f_max": 450,
            "win_length": 512,
            "window_fn": torch.hann_window
        }

    # Define the MFCC transform
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs=melkwargs
    ).to(device)

    # Apply the transform to get MFCCs: [B, n_mfcc, time_steps]
    waveforms = kpeak_normalise_signal_torch(waveforms, k=26*4)  # [B, T]
    mfccs = mfcc_transform(waveforms)

    return mfccs

class MFCConformerConfig(PretrainedConfig):
    model_type = "mfccconformer"

    def __init__(
        self,
        input_dim=128,              # MFCC dimension
        hidden_dim=512,            # Model dimension
        num_layers=4,             # Number of encoder blocks
        num_heads=4,               # Attention heads
        conv_expansion_factor=2,   # For conv module
        conv_kernel_size=31,       # Kernel size in conv
        ff_multiplier=4,           # Feedforward hidden multiplier
        mlp_hidden_dim=512,        # MLP classifier hidden layer size
        num_classes=2,             # For binary classification
        dropout=0.2,
        layer_norm_eps=1e-5,
        num_channels=1,
        lora=False,
        lambda_c=0.01,
        alpha=0.5,
        beta=0.7,
        center=0.01,
        temperature=0.7,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim * num_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.conv_expansion_factor = conv_expansion_factor
        self.conv_kernel_size = conv_kernel_size
        self.ff_multiplier = ff_multiplier
        self.mlp_hidden_dim = mlp_hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_inputs = num_channels 
        self.lambda_c = lambda_c
        self.alpha = alpha
        self.beta = beta
        self.center = center
        self.temperature = temperature

class GLUConvModule(nn.Module):
    def __init__(self, dim, expansion_factor=2, kernel_size=31):
        super().__init__()
        self.pw_conv1 = nn.Conv1d(dim, dim * expansion_factor * 2, 1)
        self.glu = nn.GLU(dim=1)
        self.dw_conv = nn.Conv1d(
            dim * expansion_factor, dim * expansion_factor, kernel_size,
            padding=kernel_size // 2, groups=dim * expansion_factor
        )
        self.bn = nn.BatchNorm1d(dim * expansion_factor)
        self.swish = nn.SiLU()
        self.pw_conv2 = nn.Conv1d(dim * expansion_factor, dim, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pw_conv1(x)
        x = self.glu(x)
        x = self.dw_conv(x)
        x = self.bn(x)
        x = self.swish(x)
        x = self.pw_conv2(x)
        return x.transpose(1, 2)

class EncoderBlock(nn.Module):
    def __init__(self, config: MFCConformerConfig):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim * config.ff_multiplier),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * config.ff_multiplier, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        self.attn = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.norm_attn = nn.LayerNorm(config.hidden_dim)
        self.conv = GLUConvModule(config.hidden_dim, config.conv_expansion_factor, config.conv_kernel_size)
        self.norm_conv = nn.LayerNorm(config.hidden_dim)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim * config.ff_multiplier),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * config.ff_multiplier, config.hidden_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attn(self.norm_attn(x), self.norm_attn(x), self.norm_attn(x))[0]
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.ffn2(x)
        return x

class MFCConformer(PreTrainedModel):
    config_class = MFCConformerConfig

    def __init__(self, config: MFCConformerConfig):
        super().__init__(config)
        self.preprocessor = preprocess_mfcc_batch if config.num_inputs == 1 else preprocess_mfcc_multichannel_batch
        self.proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.encoder = nn.ModuleList([
            EncoderBlock(config) for _ in range(config.num_layers)
        ])
        self.final_ln = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # MLP Classifier: Single hidden layer
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, config.num_classes)
        )

        self.criterion = ContrastiveFocalLoss(
            center_loss_fn=CenterLoss(2, config.hidden_dim, lambda_c = self.config.lambda_c),
            alpha=self.config.alpha,
            beta=self.config.beta,
            temperature=self.config.temperature,
            center=self.config.center,
        )

        self.post_init()

    def extract_features(self, input_values, attention_mask=None, output_hidden_states=False):
        # input_values: [B, T, F]
        mfcc = self.preprocessor(input_values)
        x = mfcc.transpose(1, 2)
        x = self.proj(x)

        hidden_states = []

        for layer in self.encoder:
            x = layer(x)
            if output_hidden_states:
                hidden_states.append(x)

        x = self.final_ln(x)  # [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.pooling(x).squeeze(-1)  # [B, D]

        if output_hidden_states:
            return x, hidden_states

        return x

    def forward(self, input_values, labels, attention_mask=None, output_hidden_states=False):
        # input_values: [B, T, F]
        x = self.extract_features(input_values, attention_mask, output_hidden_states)
        logits = self.classifier(x)  # [B, num_classes]

        return x, logits, labels
