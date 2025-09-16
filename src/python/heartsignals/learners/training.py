"""
    training.py
    Author : Milan Marocchi

    Contains the training code to train the ml algorithm.
    Allows for testing of different pre-processing and ml algorithms.
"""
import os
import time
import math
import logging
import traceback
from typing import Any, Callable, List, Optional

import numpy as np
from tqdm.auto import tqdm
from processing.filtering import kpeak_normalise_signal_torch
import torchaudio.transforms as T
from util.classify_stats import RunningBinaryConfusionMatrix

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import default_collate
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import PreTrainedModel
from util.reproducible import seed_worker
from peft import PeftModel

from models.model_factory import get_optimizer_and_scheduler

class TrainerArguments():
    def __init__(
        self,
        output_dir: str,
        batch_size: int = 64,
        mini_batch: int = 64,
        num_workers: int = 16,
        num_epochs: int = 20,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        optim: str = "sgd",
        momentum: float = 0.9,
        step_size: int = 3,
        gamma: float = 0.1,
        dataset_idx: Optional[int] = None,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.95,
        max_grad_norm: Optional[float] = None,
        **kwargs         
    ):
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.mini_batch = mini_batch
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optim
        self.momentum = momentum
        self.dataset_idx = dataset_idx
        self.step_size = step_size
        self.gamma = gamma
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.max_grad_norm = max_grad_norm

    def __str__(self):
        return (
            f"TrainerArguments("
            f"output_dir='{self.output_dir}', "
            f"batch_size={self.batch_size}, "
            f"mini_batch={self.mini_batch}, "
            f"num_workers={self.num_workers}, "
            f"num_epochs={self.num_epochs}, "
            f"learning_rate={self.learning_rate}, "
            f"weight_decay={self.weight_decay}, "
            f"optimizer_type='{self.optimizer_type}', "
            f"momentum={self.momentum}, "
            f"step_size={self.step_size}, "
            f"gamma={self.gamma}, "
            f"adam_beta1={self.adam_beta1}, "
            f"adam_beta2={self.adam_beta2}, "
            f"dataset_idx={self.dataset_idx}, "
            f"max_grad_norm={self.max_grad_norm}"
            f")"
        )

    def __repr__(self):
        return self.__str__()


class Trainer():

    def __init__(
            self, 
            model: PreTrainedModel,
            args: TrainerArguments,
            train_dataset: Dataset,
            eval_dataset: Dataset,
            objective_stat: str = 'mcc',
            end_step_callback: Optional[Callable] = None,
            collator_fn: Optional[Any] = None,
            **kwargs
    ):
        self.args = args
        self.model = model
        self.model_config = model.config
        try:
            self.criterion = self.model.criterion
        except AttributeError:
            self.criterion = nn.CrossEntropyLoss()  # Default criterion if not set in model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.end_step_callback = end_step_callback
        self.objective_stat = objective_stat
        self.writer = SummaryWriter()
        self.mini_batch = self.args.mini_batch if self.args.batch_size >= self.args.mini_batch else self.args.batch_size
        self.forward_passes = 0
        self.collator_fn = collator_fn

        self.create_dataloaders()
        self.create_optimizer_and_scheduler()

    def train_model(self):
        raise NotImplementedError("Needs to be implemented.")

    def save_model(self):
        if self.args.output_dir is None:
            self.args.output_dir = os.path.abspath(".")

        save_dir = os.path.join(self.args.output_dir, "trained_model")
        os.makedirs(save_dir, exist_ok=True)

        # Save everything
        self._save_model(save_dir)

    def _save_model(self, save_dir):
        state_dict = self.model.state_dict()
        if hasattr(self.model, "model"):
            if isinstance(self.model.model, PeftModel):  # Check if LoRA is applied
                self.model.model.save_pretrained(save_dir)
            self.model.save_pretrained(save_dir, state_dict=state_dict)
        else:
            self.model.save_pretrained(save_dir, state_dict=state_dict)

        torch.save(self.args, os.path.join(save_dir, "training_arguments"))

    def save_checkpoint(self):
        if self.args.output_dir is None:
            self.args.output_dir = self.args.output_dir

        save_dir = os.path.join(self.args.output_dir, "checkpoint")
        os.makedirs(save_dir, exist_ok=True)

        # Save everything
        self._save_model(save_dir)

    def create_train_sampler(self, dataset):
        labels_list = dataset.get_labels()
        labels = torch.zeros(len(dataset), dtype=torch.long)
        for idx, label in enumerate(labels_list):
            if isinstance(label, str) and '.' in label:
                label_stripped = int(label.split('.')[1])
                label = label_stripped
            labels[idx] = int(label)

        class_counts = torch.tensor([(labels == i).sum() for i in torch.unique(torch.tensor(labels))])
        class_weights = 1. / class_counts.float()
        logging.log(logging.INFO, f"{class_weights=}")
        sample_weights: torch.Tensor = class_weights[labels]
        self.sample_weights = sample_weights

        test_sampler = WeightedRandomSampler(
            weights=sample_weights, # type: ignore
            num_samples=len(sample_weights), 
            replacement=True
        )

        return test_sampler

    def _setup_labels(self, labels):
        # If the labels have not been put into ints
        try:
            labels = [int(label) for label in labels]
        except ValueError:
            labels = [1 if int(x.split('.')[1]) == 1 else 0 for x in labels]

        labels = torch.tensor(labels)

        return labels


    def test_dataloaders(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        idx = 0
        for inputs in self.dataloaders["valid"]:
            self.set_mode("valid")
            input_vals = inputs["input_vals"]
            labels = inputs["label"]

            input_vals = input_vals.to(device)
            labels = self._setup_labels(labels).to(device)
            self.model.to(device) # type: ignore

            self.model(input_vals, labels)

            idx += 1
            if idx == 4:
                break

    def setup_dataloader(self, train_dataset, eval_dataset, sampler, num_workers):
        return {
            "train": DataLoader(
                train_dataset,
                batch_size=self.mini_batch,
                sampler=sampler,
                worker_init_fn=seed_worker,
                num_workers=num_workers,
                collate_fn=self.collator_fn if self.collator_fn is not None else default_collate
            ),
            "valid": DataLoader(
                eval_dataset,
                batch_size=self.mini_batch,
                worker_init_fn=seed_worker,
                num_workers=num_workers,
                collate_fn=self.collator_fn if self.collator_fn is not None else default_collate
            )
        }

    def create_dataloaders(self):
        sampler = self.create_train_sampler(self.train_dataset)
        self.args.num_workers = 0

        # FIXME: Remove this temp fix
        if self.args.num_workers == 0:
            self.dataloaders = self.setup_dataloader(self.train_dataset, self.eval_dataset, sampler, 0)
            return

        for num_workers in range(self.args.num_workers, 0, -1):
            self.dataloaders = self.setup_dataloader(self.train_dataset, self.eval_dataset, sampler, num_workers)

            try:
                self.test_dataloaders()
                break
            except Exception as e:
                print(f"Failed with num_workers={num_workers}. Error: {e}")

        logging.debug(f"{self.args.num_workers=}")
                
        # Ensure that the mini_batch size is not still too large.
        if num_workers == 0:
            try:
                self.test_dataloaders()
            except:
                self.mini_batch /= 2

    def create_optimizer_and_scheduler(self, **kwargs):
        optimizer, scheduler = get_optimizer_and_scheduler(
            self.model.parameters(), 
            self.args.optimizer_type,  # type: ignore Using below class
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            momentum=self.args.momentum, # type: ignore Using below class
            step_size=self.args.step_size, # type: ignore Using below class
            gamma=self.args.gamma  # type: ignore Using below class
        )
        self.optimizer = optimizer
        self.lr_scheduler = scheduler

    def set_mode(self, phase: str):
        if phase == 'train':
            self.model.train()  # Set model to training mode
        else:
            self.model.eval()   # Set model to evaluate mode

    def _is_metric_better(self, phase, epoch_measures, best_epoch_measure):

        if self.objective_stat == "loss":
            comparison = lambda new, best: new < best
        else:
            comparison = lambda new, best: new > best

        epoch_measure = (2/3*(epoch_measures['valid']) + 1/3*(epoch_measures['train'])) if phase == 'valid' else None
        best_model = epoch_measure is not None and (best_epoch_measure is None or 
                        not math.isnan(epoch_measure) and comparison(epoch_measure, best_epoch_measure))
        #logging.info(f'{epoch_measure=}, {best_model=}, {best_epoch_measure=}')

        return epoch_measure, best_model

    def update_summary_writer(self, stats: dict, aliases: dict, phase: str, epoch: int):
        for key in stats.keys():
            self.writer.add_scalar(f"{phase} : {aliases[key]}", stats[key], epoch)

    def load_checkpoint(self):
        checkpoint_dir = os.path.join(self.args.output_dir, "checkpoint")

        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"No saved checkpoint in {self.args.output_dir}")

        model_class = type(self.model)
        self.model : PreTrainedModel = model_class.from_pretrained(
            checkpoint_dir, 
            config=self.model_config,
            ignore_mismatched_sizes=True
        ).to(self.model.device) # type: ignore

class SupervisedTrainer(Trainer):

    def __init__(
            self, 
            model: PreTrainedModel,
            args: TrainerArguments,
            train_dataset: Dataset,
            eval_dataset: Dataset,
            objective_stat: str = 'mcc',
            end_step_callback: Optional[Callable] = None,
            **kwargs
    ):
        super().__init__(
            model, 
            args,
            train_dataset,
            eval_dataset,
            objective_stat,
            end_step_callback
        )
        # Try to set the weights of the criterion
        if hasattr(self.criterion, 'weights'):
            self.criterion.weights = self.sample_weights

    def train_step(self, inputs, labels, labels_dict, phase, runningCM):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        inputs = inputs.to(device)
        labels = self._setup_labels(labels).to(device)
        self.model = self.model.to(device) # Double check correct device for model

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            self.forward_passes += 1
            # May not have a config 
            try:
                if self.model.config.model_type == "inception":
                    logits, aux_logits = self.model(inputs, labels)
                    loss1 = self.criterion(logits, labels)
                    loss2 = self.criterion(aux_logits, labels)
                    loss = loss1 + 0.4*loss2
                elif "ast" in self.model.config.model_type or "whisper" in self.model.config.model_type or "audio" or "multi_input_audio" in self.model.config.model_type or "wav2vec" in self.model.config.model_type:
                    features, logits, labels = self.model(inputs, labels)
                    loss = self.criterion(features, logits, labels)
                    #loss = self.criterion(logits, labels)
                elif "vae" in self.model.config.model_type:
                    outputs = self.model(inputs, labels)
                    loss = self.criterion(*outputs, M_N = self.model.config.M_N)
                    logits = torch.zeros(labels.shape)
                elif "multihead" in self.model.config.model_type:
                    domain_labels = labels_dict["domain"].to(device)
                    abnormality_labels = labels_dict["abnormality"].to(device)
                    reconstructed, out_ab, out_dom, original_features = self.model(inputs, None)
                    loss = self.criterion(
                        reconstructed, 
                        out_ab, 
                        out_dom, 
                        abnormality_labels, 
                        domain_labels, 
                        original_features
                    )
                    logits = out_ab
                else:
                    logits = self.model(inputs, labels)
                    loss = self.criterion(logits, labels)
            except Exception as e:
                logging.info(e)
                logits = self.model(inputs, labels)
                loss = self.criterion(logits, labels)

            _, preds = torch.max(logits, 1)

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward(retain_graph=True)
                if (self.forward_passes * self.mini_batch) % self.args.batch_size == 0:
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad /= (self.args.batch_size // self.mini_batch)

                    if self.args.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_grad_norm)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # statistics
            runningCM.update(y_true=labels.data.to("cpu"), y_pred=preds.to("cpu"), loss=float(loss.item()*inputs.size(0)))

    def train(self, letskip=False, best_epoch_measure=-1, best_stats=None):
        """
        Allows for training of either a pytoch or tf model.
        Expects the model input to be an MLModel
        """
        since = time.time()
        if self.objective_stat == "loss":
            best_epoch_measure = float('inf')

        # Save the model checkpoint to be loaded later
        self.save_checkpoint()
        epoch_measures = {}
        best_epoch = 0

        for epoch in tqdm(range(0 if letskip else 1, self.args.num_epochs+1), desc='Epoch', ncols=120, position=0, leave=False):
            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                # Leigh's special
                if phase in ['train', 'valid'] and epoch == 0:
                    continue

                self.set_mode(phase)

                total_inputs = 0
                runningCM = RunningBinaryConfusionMatrix()

                for inputs in tqdm(self.dataloaders[phase], desc=f'{phase}', ncols=120, position=1, leave=False):
                    input_vals = inputs['input_vals']
                    labels = inputs['label']
                    labels_dict = inputs['labels_dict'] if 'labels_dict' in inputs.keys() else {}
                    total_inputs += input_vals.size(0)
                    self.train_step(input_vals, labels, labels_dict, phase, runningCM)

                # Step lr scheduler
                if phase == 'train':
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                assert total_inputs == runningCM.total(), f'{total_inputs=}, {runningCM.total()=}'
                stats = runningCM.get_stats()

                # Collect results
                if phase in ["train", "valid"]:
                    epoch_measures[phase] = stats[self.objective_stat]

                self.update_summary_writer(stats, runningCM.aliases, phase, epoch)
                print(f'{phase}')
                print(runningCM.display_stats(aliases=False))
                print(runningCM.base_stats)

                logging.debug(f'{best_epoch_measure=}')
                epoch_measure, best_model = self._is_metric_better(phase, epoch_measures, best_epoch_measure)
                if best_model:
                    best_epoch_measure = epoch_measure
                    self.save_checkpoint()
                    best_epoch = epoch
                    best_stats = stats
                    print('Found new best')

                if self.end_step_callback is not None:
                    self.end_step_callback(self, stats, epoch, phase, self.args.dataset_idx)

            print()
            
        torch.cuda.empty_cache()
        assert best_stats is not None, f'{best_stats=}'
        time_elapsed = time.time() - since
        print('\t Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('\t Best val measure: {:.4f}'.format(best_epoch_measure))
        print('\t Best val epoch: {}'.format(best_epoch))
        print('\t Best val stats:\n\t', ', '.join([f'{s}={best_stats[s]:.3f}' for s in best_stats]))

        # load best model weights
        self.load_checkpoint()

        # FIXME: Don't need to return this probably
        return self.model, best_stats, best_epoch_measure

class UnsupervisedTrainer(Trainer):
    """Unsupervised trainer class"""
    def __init__(
            self, 
            model: PreTrainedModel,
            args: TrainerArguments,
            train_dataset: Dataset,
            eval_dataset: Dataset,
            objective_stat: str = 'loss',
            end_step_callback: Optional[Callable] = None,
            collator_fn: Optional[Any] = None,
            **kwargs
    ):
        super().__init__(
            model, 
            args,
            train_dataset,
            eval_dataset,
            objective_stat,
            end_step_callback,
            collator_fn=collator_fn
        )

        self.masked_objective = True

    def apply_mask(self, input_values, mask_prob=0.15):
        """
        Apply a mask to a percentage of the input audio frames.
        """
        mask = np.random.rand(input_values.shape[1]) < mask_prob
        masked_values = input_values.clone()
        masked_values[:, mask] = 0  # Masked tokens are set to 0
        return masked_values, mask

    def _check_shape(self, tag, x_neg, mask, x_ref):
        if x_neg.shape != x_ref:
            raise RuntimeError(f"[{tag}] x shape drift: {tuple(x_neg.shape)} vs {tuple(x_ref)}")
        if mask is not None and mask.shape != (x_ref[0], x_ref[2]):
            raise RuntimeError(f"[{tag}] mask shape drift: {tuple(mask.shape)} vs {(x_ref[0], x_ref[2])}")

    def _ensure_3d(self, x):
        if x.dim() != 3:
            raise RuntimeError(f"Expected (B,C,T), got {tuple(x.shape)}")
    @torch.no_grad()
    def make_negatives(
        self,
        x: torch.Tensor,                # (B, C, T)
        mask: torch.Tensor,             # (B, T) True = valid
        *,
        # --- core params (kept from your version) ---
        max_shift: int = 800,
        noise_std: float = 0.01,
        time_num_masks: int = 2,
        time_max_width: int = 1600,
        time_max_frac: float = 0.40,
        freq_num_masks: int = 2,
        freq_max_width_hz: float = 500.0,
        sample_rate: float = 4000.0,
        taper_frac: float = 0.15,
        circular: bool = True,
        respect_valid_mask: bool = True,
        # --- new options bucketed in a dict ---
        neg_cfg: dict | None = None,
        # curriculum: pass current epoch if you want staged hardness
        epoch: int | None = None,
    ):
        """
        Build harder negatives by combining:
        1) shift (+ optional mild resample/time-warp) + Gaussian noise
        2) vectorized time masking (coverage-capped, respects mask)
        3) tapered frequency band-stop (rFFT)
        4) optional PCG-friendly extras via neg_cfg (EQ tilt/peaks, hum notch, micro-dropouts,
            colored noise, RMS normalize, curriculum)
        """

        # --------- default neg_cfg if None ----------
        if neg_cfg is None:
            neg_cfg = dict(
                time_warp=True,
                warp_scale=0.06,
                warp_seg=320,
                eq_peaks=2,
                eq_gain_db=3.0,
                shelf_gain_db=4.0,
                hum_notch_prob=0.3,
                micro_drop_prob=0.5,
                micro_drop_rate=0.02,
                micro_drop_maxw=32,
                snr_db=(18.0, 35.0),
                loudness_normalize=False,
            )

        # --------- curriculum helper ----------
        def _curr(default):
            cur = neg_cfg.get('curriculum')
            if not cur or epoch is None:
                return default
            cut = cur.get('epochs', [])
            levels = cur.get('levels', [])
            idx = 0
            for i, e in enumerate(cut):
                if epoch >= e:
                    idx = i
            if isinstance(default, dict):
                key = levels[idx] if idx < len(levels) else list(default.keys())[-1]
                return default.get(key, list(default.values())[-1])
            return default

        B, C, T = x.shape
        device = x.device
        x_neg = x.clone()

        # ========= 0) optional time-warp =========
        if neg_cfg.get('time_warp', False):
            warp_scale = float(_curr(neg_cfg.get('warp_scale', 0.06)))
            warp_seg   = int(_curr(neg_cfg.get('warp_seg', 320)))

            tlin = torch.linspace(-1, 1, T, device=device)
            K = max(2, T // max(2, warp_seg))
            ctrl = torch.linspace(-1, 1, K, device=device)
            scales = 1.0 + (2*torch.rand(B, K, device=device)-1.0)*warp_scale

            idx = torch.searchsorted(ctrl, tlin).clamp(1, K-1)
            left, right = ctrl[idx-1], ctrl[idx]
            w = ((tlin - left) / (right - left)).clamp(0, 1)
            s_t = (1-w)[None,:]*scales[:, idx-1] + w[None,:]*scales[:, idx]  # (B,T)
            s_t = s_t / s_t.mean(dim=1, keepdim=True)
            tau = torch.cumsum(s_t, dim=1)
            tau = 2*(tau / tau[:, -1:].clamp_min(1e-8) - 0.5)                # [-1,1]
            grid = torch.stack([tau, torch.zeros_like(tau)], dim=-1)         # (B,T,2)

            # x: (B,C,T,1), grid: (B,T,1,2)
            x_neg = F.grid_sample(
                x_neg.unsqueeze(-1), grid.unsqueeze(2),
                mode="bilinear", padding_mode="reflection", align_corners=True
            ).squeeze(-1)                                                    # (B,C,T)

            if respect_valid_mask and mask is not None:
                m = mask.to(x_neg.dtype).unsqueeze(1).unsqueeze(-1)          # (B,1,T,1)
                m_w = F.grid_sample(
                    m, grid.unsqueeze(2),
                    mode="nearest", padding_mode="zeros", align_corners=True
                ).squeeze(-1).squeeze(1)                                     # (B,T)
                mask = (m_w > 0.5)

        self._check_shape("after_time_warp", x_neg, mask, x.shape)

        # ========= 1) shift (+ noise) =========
        # -- SHIFT (circular) --
        self._ensure_3d(x_neg)
        if max_shift > 0:
            shifts = torch.randint(1, min(max_shift, T), (B,), device=device)  # (B,)
            t = torch.arange(T, device=device)[None, :]                         # (1,T)
            idx = (t - shifts[:, None]) % T                                     # (B,T)
            x_neg = x_neg.gather(-1, idx[:, None, :].expand(-1, C, -1))         # (B,C,T)
            if respect_valid_mask and mask is not None:
                mask = mask.gather(-1, idx)                                     # (B,T)
        self._check_shape("after_shift", x_neg, mask, x.shape)           # (B, T)

        if noise_std > 0:
            x_neg = x_neg + noise_std * torch.randn_like(x_neg)
        
        self._check_shape("after_shift", x_neg, mask, x.shape)

        # ========= 1b) micro-dropouts =========
        if neg_cfg.get('micro_drop_prob', 0.0) > 0:
            p = float(_curr(neg_cfg.get('micro_drop_prob', 0.5)))
            rate = float(_curr(neg_cfg.get('micro_drop_rate', 0.02)))
            maxw = int(_curr(neg_cfg.get('micro_drop_maxw', 32)))
            do = torch.rand(B, device=device) < p
            if do.any():
                m = torch.ones(B, 1, T, device=device, dtype=x.dtype)
                n = int(max(1, round(rate * T / max(maxw, 2))))
                starts = torch.randint(0, max(1, T - maxw), (do.sum(), n), device=device)
                widths = torch.randint(2, maxw, (do.sum(), n), device=device)
                bidx = torch.nonzero(do, as_tuple=False).squeeze(-1)
                for i, bi in enumerate(bidx):
                    for s, w in zip(starts[i], widths[i]):
                        m[bi, 0, s:s+w] = 0
                x_neg = x_neg * m
        self._check_shape("after_dropout", x_neg, mask, x.shape)

        # --- shape guard ---
        if x_neg.shape != x.shape:
            raise RuntimeError(
                f"make_negatives shape drift: got {tuple(x_neg.shape)} vs input {tuple(x.shape)}"
            )

        return x_neg, mask

    def train_step(self, inputs, labels, phase, runningCM):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        inputs = inputs.to(device)
        labels = self._setup_labels(labels).to(device) if not None else None

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            self.forward_passes += 1

            reconstruction_models = [
                "wav2vec_reconstruction",
                "wav2mamba_reconstruction",
                "wav2vec_cnn_reconstruction",
                "ast_reconstruction"
            ]

            energy_models = [
                "energy_transformer_mfcc",
                "energy_conformer",
            ]

            preds = labels.data.to("cpu")
            if self.model.config.model_type in energy_models:

                if inputs.dim() == 2:
                    inputs = inputs.unsqueeze(1)  # Add channel dimension if missing

                inputs = self.model.preprocess_mfcc_multichannel_batch(waveforms=inputs)

                m = torch.ones(inputs.shape[0], inputs.shape[-1], dtype=torch.bool, device=inputs.device)
                E_pos, _ = self.model.forward_energy(inputs, attention_mask=m, return_token_energy=False)

                inputs_neg, neg_m = self.make_negatives(inputs, m)
                E_neg, _ = self.model.forward_energy(inputs_neg, attention_mask=neg_m, return_token_energy=False)

                loss = torch.nn.functional.relu(E_pos - E_neg + 1.0).mean()

            elif "vae" in self.model.config.model_type:
                outputs = self.model(inputs, labels)
                loss = self.criterion(*outputs, M_N = self.model.config.M_N)
                logits = torch.zeros(labels.shape)
            elif self.model.config.model_type in reconstruction_models:
                output, output_classify, labels = self.model(inputs, labels)
                loss = self.criterion(output, output_classify, labels)
                _, preds = torch.max(output_classify, 1)
            else:
                logits = self.model(inputs, labels)
                loss = self.criterion(logits, labels) 

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward(retain_graph=True)
                if (self.forward_passes * self.mini_batch) % self.args.batch_size == 0:
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad /= (self.args.batch_size // self.mini_batch)

                    if self.args.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_grad_norm)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # statistics
            runningCM.update(y_true=labels.data.to("cpu"), y_pred=preds.to("cpu"), loss=float(loss.item()*inputs.size(0)))

    def _is_metric_better(self, phase, epoch_measures, best_epoch_measure):

        if self.objective_stat == "loss":
            comparison = lambda new, best: new < best
        else:
            comparison = lambda new, best: new > best

        epoch_measure = (2/3*(epoch_measures['valid']) + 1/3*(epoch_measures['train'])) if phase == 'valid' else None
        best_model = epoch_measure is not None and (best_epoch_measure is None or 
                        not math.isnan(epoch_measure) and comparison(epoch_measure, best_epoch_measure))
        #logging.info(f'{epoch_measure=}, {best_model=}, {best_epoch_measure=}')

        return epoch_measure, best_model
        

    def train(self):
        """
        Train models
        """
        since = time.time()

        # Save the model checkpoint to be loaded later
        self.save_checkpoint()
        epoch_measures = {}
        best_epoch = 0
        best_stats = None
        best_epoch_measure = None

        for epoch in tqdm(range(1, self.args.num_epochs+1), desc='Epoch', ncols=120, position=0, leave=False):
            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:


                self.set_mode(phase)
                runningCM = RunningBinaryConfusionMatrix()

                for inputs in tqdm(self.dataloaders[phase], ncols=120, position=1, leave=False):
                    input_vals = inputs['input_vals']
                    labels = inputs['label']
                    self.train_step(input_vals, labels, phase, runningCM)

                # Step lr scheduler
                if phase == 'train':
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                stats = runningCM.get_loss()

                # Collect results
                if phase in ["train", "valid"]:
                    epoch_measures[phase] = stats[self.objective_stat]

                self.update_summary_writer(stats, runningCM.aliases, phase, epoch)
                print(f'{phase}')
                print(runningCM.display_stats(aliases=False))
                print(runningCM.base_stats)

                logging.debug(f'{best_epoch_measure=}')
                epoch_measure, best_model = self._is_metric_better(phase, epoch_measures, best_epoch_measure)
                if best_model:
                    best_epoch_measure = epoch_measure
                    self.save_checkpoint()
                    best_epoch = epoch
                    best_stats = stats
                    print('Found new best')

                if self.end_step_callback is not None:
                    self.end_step_callback(self, stats, epoch, phase, self.args.dataset_idx)

            print()

        assert best_stats is not None, f'{best_stats=}'
        time_elapsed = time.time() - since
        print('\t Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('\t Best val measure: {:.4f}'.format(best_epoch_measure))
        print('\t Best val epoch: {}'.format(best_epoch))
        print('\t Best val stats:\n\t', ', '.join([f'{s}={best_stats[s]:.3f}' for s in best_stats]))

        # load best model weights
        self.load_checkpoint()

        # FIXME: Don't need to return this probably
        return self.model, best_stats, best_epoch_measure