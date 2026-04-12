import time
import torch
import random
import warnings
from pathlib import Path
import copy
import numpy as np
import pandas as pd
from typing import Dict, Mapping, Optional, Tuple, Any, Union
from typing import List, Tuple

from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

from anndata import AnnData
import scanpy as sc
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from omegaconf import OmegaConf

# multiprocessing.set_start_method('spawn', force=True)
import wandb
from scipy.sparse import issparse


from ..utils.custom_tokenizer import tokenize_and_pad_batch, random_mask_value
from ..utils.logger import create_logger
import matplotlib.pyplot as plt

from ..utils.set_optimizer import create_optimizer_dict
from ..custom_loss import (
    cce_loss,
    criterion_neg_log_bernoulli,
    masked_mse_loss,
    masked_relative_error,
    GenerativeExpressionLoss,
)
from ..utils.plot import process_and_log_umaps
from ..utils.metrics import cell_eval_to_wandb
from ..utils.misc import init_plot_worker


def _config_get(config, key, default=None):
    return (
        config.get(key, default)
        if hasattr(config, "get")
        else getattr(config, key, default)
    )


def _config_as_dict(config) -> Dict[str, Any]:
    if hasattr(config, "as_dict"):
        return dict(config.as_dict())
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    if isinstance(config, Mapping):
        return dict(config)
    return dict(config)


def _config_set(config, key, value) -> None:
    if hasattr(config, "update"):
        try:
            config.update({key: value}, allow_val_change=True)
            return
        except TypeError:
            try:
                config.update({key: value})
                return
            except Exception:
                pass
        except Exception:
            pass

    try:
        config[key] = value
        return
    except Exception:
        pass

    try:
        OmegaConf.update(config, key, value, merge=False)
        return
    except Exception:
        pass

    setattr(config, key, value)


def _uses_flow_matching(model: nn.Module, config) -> bool:
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    runtime_flow_active = getattr(base_model, "flow_matching_active", None)
    if runtime_flow_active is None:
        runtime_flow_active = _config_get(config, "flow_matching_active", None)
    if runtime_flow_active is not None:
        return bool(
            runtime_flow_active
            and getattr(base_model, "flow_perturbation_generator", None) is not None
        )
    return bool(
        getattr(base_model, "flow_matching", False)
        or _config_get(config, "flow_matching", False)
    )


def _needs_next_perturbation_labels(
    config,
    flow_matching_enabled: bool,
    has_lochness_next_pred: bool,
) -> bool:
    return bool(
        flow_matching_enabled
        or _config_get(config, "next_weight", 0) > 0
        or _config_get(config, "CCE", False)
        or has_lochness_next_pred
    )


def _parameter_matches_prefix(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(f"{prefix}.")


def _summarize_key_roots(keys: List[str]) -> Dict[str, int]:
    summary = {}
    for key in keys:
        root = key.split(".")[0]
        summary[root] = summary.get(root, 0) + 1
    return dict(sorted(summary.items()))


def _resolve_init_checkpoint_path(config) -> Optional[Path]:
    checkpoint_dir = _config_get(config, "init_checkpoint_dir")
    if checkpoint_dir is None:
        checkpoint_dir = _config_get(config, "phase_init_checkpoint_dir")
    checkpoint_file = _config_get(config, "init_checkpoint_file")
    if checkpoint_file is None:
        checkpoint_file = _config_get(config, "phase_init_checkpoint_file")

    if checkpoint_file:
        checkpoint_file_path = Path(str(checkpoint_file)).expanduser()
        if checkpoint_file_path.is_absolute():
            return checkpoint_file_path.resolve()
        if checkpoint_dir:
            return (
                Path(str(checkpoint_dir)).expanduser() / checkpoint_file_path
            ).resolve()
        return checkpoint_file_path.resolve()

    if checkpoint_dir:
        return (Path(str(checkpoint_dir)).expanduser() / "final_model.pt").resolve()

    return None


def _load_initial_checkpoint_if_configured(model, config, device=None, logger=None):
    checkpoint_path = _resolve_init_checkpoint_path(config)
    if checkpoint_path is None:
        return None
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Initialization checkpoint not found: {checkpoint_path}"
        )

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state_dict, Mapping) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"Initialization checkpoint must contain a state_dict mapping, got {type(state_dict)}"
        )

    loaded_keys = set(state_dict.keys())
    model_keys = set(model.state_dict().keys())
    missing_keys = sorted(model_keys - loaded_keys)
    unexpected_keys = sorted(loaded_keys - model_keys)
    model.load_state_dict(state_dict, strict=False)

    flow_missing_only = bool(missing_keys) and all(
        key.startswith("flow_perturbation_generator.") for key in missing_keys
    )
    report = {
        "checkpoint_path": str(checkpoint_path),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "missing_key_roots": _summarize_key_roots(missing_keys),
        "unexpected_key_roots": _summarize_key_roots(unexpected_keys),
        "flow_missing_only": flow_missing_only,
    }

    if logger is not None:
        logger.info("Loaded initialization checkpoint: %s", checkpoint_path)
        if not missing_keys and not unexpected_keys:
            logger.info(
                "Initialization checkpoint load matched model exactly (0 missing / 0 unexpected keys)"
            )
        else:
            logger.warning(
                "Initialization checkpoint used partial load: %d missing / %d unexpected keys",
                len(missing_keys),
                len(unexpected_keys),
            )
            if missing_keys:
                logger.warning(
                    "Initialization checkpoint missing key roots: %s",
                    report["missing_key_roots"],
                )
            if unexpected_keys:
                logger.warning(
                    "Initialization checkpoint unexpected key roots: %s",
                    report["unexpected_key_roots"],
                )
            if flow_missing_only:
                logger.info(
                    "Partial load is expected for flow-stage initialization from a non-flow checkpoint: only flow_perturbation_generator.* keys are missing"
                )

    return report


def _apply_trainable_module_whitelist(model, config, logger=None):
    trainable_prefixes = _config_get(config, "trainable_module_prefixes")
    if trainable_prefixes is None:
        trainable_prefixes = _config_get(config, "trainable_modules")
    if not trainable_prefixes:
        return None
    if isinstance(trainable_prefixes, str):
        trainable_prefixes = [trainable_prefixes]

    named_parameters = list(model.named_parameters())
    if not named_parameters:
        raise ValueError("Model has no named parameters to apply a freeze policy")

    per_prefix = {}
    for prefix in trainable_prefixes:
        matched = [
            (name, param)
            for name, param in named_parameters
            if _parameter_matches_prefix(name, prefix)
        ]
        per_prefix[prefix] = {
            "tensor_count": len(matched),
            "parameter_count": int(sum(param.numel() for _, param in matched)),
            "names": [name for name, _ in matched],
        }

    unknown_prefixes = [
        prefix for prefix, info in per_prefix.items() if info["tensor_count"] == 0
    ]
    if unknown_prefixes:
        raise ValueError(
            f"Unknown trainable module prefixes: {unknown_prefixes}. "
            "Expected prefixes must match model.named_parameters()."
        )

    for name, param in named_parameters:
        param.requires_grad = any(
            _parameter_matches_prefix(name, prefix) for prefix in trainable_prefixes
        )

    trainable_names = [name for name, param in named_parameters if param.requires_grad]
    frozen_names = [name for name, param in named_parameters if not param.requires_grad]
    summary = {
        "requested_prefixes": list(trainable_prefixes),
        "matched_prefixes": per_prefix,
        "trainable_tensor_count": len(trainable_names),
        "trainable_parameter_count": int(
            sum(param.numel() for _, param in named_parameters if param.requires_grad)
        ),
        "frozen_tensor_count": len(frozen_names),
        "frozen_parameter_count": int(
            sum(
                param.numel()
                for _, param in named_parameters
                if not param.requires_grad
            )
        ),
        "trainable_names": trainable_names,
        "frozen_names": frozen_names,
    }

    if logger is not None:
        logger.info("Applied trainable module whitelist: %s", trainable_prefixes)
        logger.info(
            "Parameter freeze summary: trainable=%d tensors / %d params | frozen=%d tensors / %d params",
            summary["trainable_tensor_count"],
            summary["trainable_parameter_count"],
            summary["frozen_tensor_count"],
            summary["frozen_parameter_count"],
        )
        logger.info(
            "Trainable module matches: %s",
            {
                prefix: {
                    "tensor_count": info["tensor_count"],
                    "parameter_count": info["parameter_count"],
                }
                for prefix, info in per_prefix.items()
            },
        )

    return summary


def _resolve_staged_training_plan(config) -> Optional[Dict[str, Any]]:
    stage2_start_epoch = _config_get(config, "stage2_start_epoch")
    if stage2_start_epoch is None:
        return None

    stage2_start_epoch = int(stage2_start_epoch)
    if stage2_start_epoch <= 1:
        raise ValueError(
            "stage2_start_epoch must be greater than 1 for staged training"
        )

    overrides = {}
    for key, value in _config_as_dict(config).items():
        if key.startswith("stage2_") and key != "stage2_start_epoch":
            overrides[key[len("stage2_") :]] = copy.deepcopy(value)

    if not overrides:
        raise ValueError(
            "stage2_start_epoch is set but no stage2_* overrides were provided"
        )

    return {
        "start_epoch": stage2_start_epoch,
        "overrides": overrides,
    }


def _initialize_runtime_training_controls(model, config, logger=None) -> None:
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    flow_generator = getattr(base_model, "flow_perturbation_generator", None)
    if flow_generator is None:
        return

    flow_matching_active = _config_get(
        config,
        "flow_matching_active",
        getattr(base_model, "flow_matching", False),
    )
    base_model.flow_matching_active = bool(flow_matching_active)
    if (
        logger is not None
        and _config_get(config, "flow_matching_active", None) is not None
    ):
        logger.info(
            "Initialized runtime flow activation: flow_matching_active=%s",
            base_model.flow_matching_active,
        )


def _apply_staged_training_overrides(model, config, overrides, logger=None):
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    applied = {}
    for key, value in overrides.items():
        value_copy = copy.deepcopy(value)
        applied[key] = {
            "old": _config_get(config, key, None),
            "new": value_copy,
        }
        _config_set(config, key, value_copy)
        if hasattr(base_model, key):
            setattr(base_model, key, copy.deepcopy(value_copy))

    if hasattr(base_model, "flow_perturbation_generator"):
        if "flow_matching_active" in overrides:
            base_model.flow_matching_active = bool(overrides["flow_matching_active"])
        elif _config_get(config, "flow_matching_active", None) is not None:
            base_model.flow_matching_active = bool(
                _config_get(config, "flow_matching_active")
            )

        if (
            getattr(base_model, "flow_matching_active", False)
            and getattr(base_model, "flow_perturbation_generator", None) is None
        ):
            raise ValueError(
                "Staged training requested flow_matching_active=true, but the model was built without flow_perturbation_generator. "
                "Set top-level flow_matching=true in the config so the flow module exists before the stage switch."
            )

    if logger is not None:
        logger.info(
            "Applied staged-training overrides: %s",
            {key: report["new"] for key, report in applied.items()},
        )
    return applied


def _staged_training_requires_optimizer_reset(overrides: Mapping[str, Any]) -> bool:
    optimizer_keys = {
        "ADV",
        "dab_weight",
        "lr",
        "lr_ADV",
        "schedule_interval",
        "schedule_ratio",
        "trainable_module_prefixes",
        "trainable_modules",
    }
    return any(key in overrides for key in optimizer_keys)


def _compute_flow_losses(
    output_dict: Mapping[str, Any],
    config,
    criterion_mvc: nn.Module,
    target_values_next: Tensor,
    full_expr_next: Tensor,
    masked_positions: Tensor,
    sf_next: Tensor,
) -> Optional[Dict[str, Tensor]]:
    flow_dict = output_dict.get("flow_dict")
    if not isinstance(flow_dict, dict):
        return None
    training_dict = flow_dict.get("training")
    if not isinstance(training_dict, dict):
        return None

    loss_velocity = masked_mse_loss(
        training_dict["pred_velocity"], training_dict["target_velocity"]
    )
    loss_endpoint_latent = masked_mse_loss(
        training_dict["latent_1_hat"], training_dict["target_latent"]
    )

    loss_endpoint_nb = loss_velocity.new_zeros(())
    if "endpoint_mvc_output" in training_dict:
        endpoint_target = (
            target_values_next
            if _config_get(config, "mvc_masked_train", True)
            else full_expr_next
        )
        endpoint_mask = (
            masked_positions if _config_get(config, "mvc_masked_train", True) else None
        )
        loss_endpoint_nb = criterion_mvc(
            training_dict["endpoint_mvc_output"],
            endpoint_target,
            endpoint_mask,
            scale_factor=sf_next,
        )

    flow_loss_weight = _config_get(config, "flow_loss_weight", 1.0)
    flow_endpoint_latent_loss_weight = _config_get(
        config, "flow_endpoint_latent_loss_weight", 1.0
    )
    flow_endpoint_nb_loss_weight = _config_get(
        config, "flow_endpoint_nb_loss_weight", 1.0
    )
    loss_total = (
        flow_loss_weight * loss_velocity
        + flow_endpoint_latent_loss_weight * loss_endpoint_latent
        + flow_endpoint_nb_loss_weight * loss_endpoint_nb
    )
    return {
        "loss_velocity": loss_velocity,
        "loss_endpoint_latent": loss_endpoint_latent,
        "loss_endpoint_nb": loss_endpoint_nb,
        "loss_total": loss_total,
    }


def train(
    model: nn.Module,
    loader: DataLoader,
    config,
    vocab,
    optim_dict: Dict,
    epoch=0,
    logger=None,
    device=None,
) -> None:
    """
    Train the model for one epoch.
    """
    logger = create_logger() if logger is None else logger
    criterion = masked_mse_loss
    criterion_dab = nn.CrossEntropyLoss()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_pert = nn.CrossEntropyLoss()
    criterion_adv = nn.CrossEntropyLoss()  # consider using label smoothing
    criterion_ps = nn.MSELoss()  # this is the loss for predicting PS scores
    criterion_mvc = GenerativeExpressionLoss()
    # criterion_ps = nn.CrossEntropyLoss()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    total_loss, total_mse, total_gepc = 0.0, 0.0, 0.0
    total_mse_next, total_gepc_next = 0.0, 0.0
    total_error, total_error_next = 0.0, 0.0
    total_dab, total_adv_E, total_adv_D = 0.0, 0.0, 0.0
    total_cls, total_pert, total_ps, total_ps_next = 0.0, 0.0, 0.0, 0.0
    total_flow_velocity, total_flow_endpoint_latent = 0.0, 0.0
    total_flow_endpoint_nb, total_flow_total = 0.0, 0.0
    log_interval = config.log_interval
    start_time = time.time()
    flow_matching_enabled = _uses_flow_matching(model, config)

    scaler = optim_dict["scaler"]
    discriminator = optim_dict["discriminator"]
    optimizer = optim_dict["optimizer"]
    scheduler = optim_dict["scheduler"]
    optimizer_dab = optim_dict["optimizer_dab"]
    scheduler_dab = optim_dict["scheduler_dab"]
    optimizer_E = optim_dict["optimizer_E"]
    scheduler_E = optim_dict["scheduler_E"]
    optimizer_D = optim_dict["optimizer_D"]
    scheduler_D = optim_dict["scheduler_D"]

    # check ps_next_weight. The ps_next prediction is used for predicting the lochness score for a new gene from pert_next label
    if hasattr(config, "pred_lochness_next"):
        has_lochness_next_pred = True
        ps_next_training_weight = config.pred_lochness_next
    else:
        has_lochness_next_pred = False
        ps_next_training_weight = config.ps_weight * config.next_weight
    needs_next_perturbation_labels = _needs_next_perturbation_labels(
        config,
        flow_matching_enabled,
        has_lochness_next_pred,
    )

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        target_values_next = batch_data["target_values_next"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)  # added
        perturbation_labels = batch_data["perturbation_labels"].to(device)  # added
        sf = batch_data["sf"].to(device)
        sf_next_raw = batch_data["sf_next"].to(device)
        sf_next = sf_next_raw if flow_matching_enabled else sf
        celltype_labels_next = batch_data["celltype_labels_next"].to(device)  # added
        perturbation_labels_next = batch_data["perturbation_labels_next"].to(
            device
        )  # added
        next_gene_ids = (
            batch_data["next_gene_ids"].to(device) if flow_matching_enabled else None
        )
        next_src_key_padding_mask = (
            next_gene_ids.eq(vocab[config.pad_token])
            if next_gene_ids is not None
            else None
        )

        mvc_src = (
            None
            if config.get("mvc_masked_train", True)
            else batch_data["full_gene_ids"].to(device)
        )
        if config.ps_weight > 0:
            ps_score = batch_data["ps"].to(device)
            ps_score_next = batch_data["ps_next"].to(device)  #

        src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])
        with torch.cuda.amp.autocast(enabled=config.amp):
            # import pdb; pdb.set_trace()

            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels
                if config.use_batch_label
                else None,  # if config.DSBN else None,
                pert_labels=perturbation_labels if config.perturbation_input else None,
                pert_labels_next=perturbation_labels_next
                if needs_next_perturbation_labels
                else None,
                next_gene_ids=next_gene_ids,
                next_values=target_values_next if flow_matching_enabled else None,
                next_src_key_padding_mask=next_src_key_padding_mask,
                sf=sf,
                sf_next=sf_next,
                MVC=config.GEPC,
                ECS=config.ecs_thres > 0,
                CLS=config.get("cell_type_classifier", True),
                CCE=config.CCE,
                PERTPRED=config.get("genotype_classifier", True),
                PSPRED=config.ps_weight > 0,
                mvc_src=mvc_src,
            )

            masked_positions = input_values.eq(
                config.mask_value
            )  # the postions to predict
            loss_mse = criterion(
                output_dict["mlm_output"], target_values, masked_positions
            )
            loss = config.this_weight * loss_mse
            metrics_to_log = {"train/mse": loss_mse.item()}

            if config.CCE and len(output_dict["contrastive_dict"]) > 0:
                cce_mode = config.get("cce_mode", "cell+geno")
                logit_norm = config.get("logit_norm", False)
                if (
                    cce_mode == "cell_geno"
                ):  ## supervised contrastive loss plus a custom contrastive loss
                    cce_weight = max(
                        config.perturbation_classifier_weight
                        * config.cell_type_classifier_weight,
                        1,
                    )
                    input_labels = (
                        celltype_labels * 1000 + perturbation_labels
                    )  # x1000 make labels unique combination of celltype and genotype
                    pert_labels = celltype_labels_next * 1000 + perturbation_labels_next
                    loss_cce = cce_loss(
                        output_dict["contrastive_dict"],
                        input_labels,
                        pert_labels,
                        logit_norm=logit_norm,
                    )
                    metrics_to_log["train/cce"] = loss_cce.item()
                    loss += loss_cce * cce_weight

                if cce_mode == "celltype" or cce_mode == "cell+geno":
                    loss_cce_celltype = cce_loss(
                        output_dict["contrastive_dict"],
                        celltype_labels,
                        celltype_labels_next,
                        logit_norm=logit_norm,
                    )
                    metrics_to_log["train/cce_celltype"] = loss_cce_celltype.item()
                    loss += loss_cce_celltype * max(
                        config.cell_type_classifier_weight, 1
                    )

                if cce_mode == "genotype" or cce_mode == "cell+geno":
                    loss_cce_genotype = cce_loss(
                        output_dict["contrastive_dict"],
                        perturbation_labels,
                        perturbation_labels_next,
                        logit_norm=logit_norm,
                    )
                    metrics_to_log["train/cce_genotype"] = loss_cce_genotype.item()
                    loss += loss_cce_genotype * max(
                        config.perturbation_classifier_weight, 1
                    )

            # next value?
            loss_mse_next = criterion(
                output_dict["mlm_output"], target_values_next, masked_positions
            )
            # disable now
            # loss = loss + config.next_weight * loss_mse_next
            metrics_to_log.update({"train/mse_next": loss_mse_next.item()})

            if config.explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + config.this_weight * loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
                # added
                loss_zero_log_prob_next = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values_next, masked_positions
                )
                # loss = loss + config.next_weight *loss_zero_log_prob_next
                metrics_to_log.update(
                    {"train/nzlp_next": loss_zero_log_prob_next.item()}
                )
            if config.GEPC:
                mvc_target_values = (
                    target_values
                    if config.get("mvc_masked_train", True)
                    else batch_data["full_expr"].to(device)
                )
                mvc_target_values_next = (
                    target_values_next
                    if config.get("mvc_masked_train", True)
                    else batch_data["full_expr_next"].to(device)
                )
                mvc_masked_positions = (
                    masked_positions if config.get("mvc_masked_train", True) else None
                )
                loss_gepc = criterion_mvc(
                    output_dict["mvc_output"],
                    mvc_target_values,
                    mvc_masked_positions,
                    scale_factor=sf,
                )
                loss = loss + config.this_weight * loss_gepc
                metrics_to_log.update({"train/mvc": loss_gepc.item()})
                # added
                loss_gepc_next = criterion_mvc(
                    output_dict["mvc_output_next"],
                    mvc_target_values_next,
                    mvc_masked_positions,
                    scale_factor=sf_next,
                )
                loss = loss + config.next_weight * loss_gepc_next
                metrics_to_log.update({"train/mvc_next": loss_gepc_next.item()})

                if config.explicit_zero_prob and config.distribution is None:
                    loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mvc_output"]["zero_probs"],
                        mvc_target_values,
                        mvc_masked_positions,
                    )
                    loss = loss + config.this_weight * loss_gepc_zero_log_prob
                    metrics_to_log.update(
                        {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
                    )
                    # added
                    loss_gepc_zero_log_prob_next = criterion_neg_log_bernoulli(
                        output_dict["mvc_output_next"]["zero_probs"],
                        mvc_target_values_next,
                        mvc_masked_positions,
                    )
                    loss = loss + config.next_weight * loss_gepc_zero_log_prob_next
                    metrics_to_log.update(
                        {"train/mvc_nzlp_next": loss_gepc_zero_log_prob_next.item()}
                    )
            flow_losses = _compute_flow_losses(
                output_dict,
                config,
                criterion_mvc,
                target_values_next,
                batch_data["full_expr_next"].to(device),
                masked_positions,
                sf_next_raw,
            )
            if flow_losses is not None:
                loss = loss + flow_losses["loss_total"]
                metrics_to_log.update(
                    {
                        "train/flow_velocity": flow_losses["loss_velocity"].item(),
                        "train/flow_endpoint_latent": flow_losses[
                            "loss_endpoint_latent"
                        ].item(),
                        "train/flow_endpoint_nb": flow_losses[
                            "loss_endpoint_nb"
                        ].item(),
                        "train/flow_total": flow_losses["loss_total"].item(),
                    }
                )
            if config.get("cell_type_classifier", True):
                loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                loss = loss + config.cell_type_classifier_weight * loss_cls
                metrics_to_log.update({"train/cls": loss_cls.item()})
                # add for next cls prediction
                loss_cls_next = criterion_cls(
                    output_dict["cls_output_next"], celltype_labels_next
                )
                loss = (
                    loss
                    + config.cell_type_classifier_weight
                    * config.next_weight
                    * loss_cls_next
                )
                metrics_to_log.update({"train/cls_next": loss_cls_next.item()})

                error_rate = 1 - (
                    (output_dict["cls_output"].argmax(1) == celltype_labels)
                    .sum()
                    .item()
                ) / celltype_labels.size(0)

            if config.get("genotype_classifier", True):
                loss_pert = criterion_pert(
                    output_dict["pert_output"], perturbation_labels
                )
                loss = loss + config.perturbation_classifier_weight * loss_pert
                metrics_to_log.update({"train/pert": loss_pert.item()})
                # add for next pert prediction
                loss_pert_next = criterion_pert(
                    output_dict["pert_output_next"], perturbation_labels_next
                )
                loss = (
                    loss
                    + config.perturbation_classifier_weight
                    * config.next_weight
                    * loss_pert_next
                )
                metrics_to_log.update({"train/pert_next": loss_pert_next.item()})

            if config.ps_weight > 0:
                loss_ps = criterion_ps(output_dict["ps_output"], ps_score)
                # import pdb; pdb.set_trace()
                # print(f"loss_ps: {loss_ps}")
                loss = loss + config.ps_weight * loss_ps
                metrics_to_log.update({"train/ps": loss_ps.item()})
                loss_ps_next = criterion_ps(
                    output_dict["ps_output_next"], ps_score_next
                )
                loss = loss + ps_next_training_weight * loss_ps_next
                metrics_to_log.update({"train/ps_next": loss_ps_next.item()})

            if config.ecs_thres > 0:
                loss_ecs = config.ecs_weight * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})

            if config.dab_weight > 0:
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                loss = loss + config.dab_weight * loss_dab
                metrics_to_log.update({"train/dab": loss_dab.item()})

        model.zero_grad()
        # print(f"loss: {loss}")
        # import pdb; pdb.set_trace()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0 and logger is not None:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        if config.ADV:
            # rerun the model for adversarial training
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels
                if config.use_batch_label
                else None,  # if config.DSBN else None,
                pert_labels=perturbation_labels if config.perturbation_input else None,
                pert_labels_next=perturbation_labels_next
                if needs_next_perturbation_labels
                else None,
                next_gene_ids=next_gene_ids,
                next_values=target_values_next if flow_matching_enabled else None,
                next_src_key_padding_mask=next_src_key_padding_mask,
                sf=sf,
                sf_next=sf_next,
                MVC=config.GEPC,
                ECS=config.ecs_thres > 0,
                CLS=config.get("cell_type_classifier", True),
                # CCE=config.CCE,
                PERTPRED=config.get("genotype_classifier", True),
                PSPRED=config.ps_weight > 0,
                # do_sample=config.do_sample_in_train,
                # generative_training=False
            )

            # TRAINING DISCRIMINATOR
            loss_adv_D = config.adv_weight * criterion_adv(
                discriminator(output_dict["cell_emb"].detach()), batch_labels
            )
            if epoch > config.adv_D_delay_epochs:
                discriminator.zero_grad()
                loss_adv_D.backward()
                optimizer_D.step()

            # TRAINING ENCODER
            loss_adv_E = (
                -1
                * config.adv_weight
                * criterion_adv(discriminator(output_dict["cell_emb"]), batch_labels)
            )
            # NOTE: the loss is negative here because we want to maximize
            # the cross_entropy_loss, in other words, disguise against the discriminator
            if epoch > config.adv_E_delay_epochs:
                model.zero_grad()
                discriminator.zero_grad()
                loss_adv_E.backward()
                optimizer_E.step()

        wandb.log(metrics_to_log)

        with torch.no_grad():
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )
            mre_next = masked_relative_error(
                output_dict["mlm_output"], target_values_next, masked_positions
            )

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_mse_next += loss_mse_next.item()
        total_gepc += loss_gepc.item() if config.GEPC else 0.0
        total_gepc_next += loss_gepc_next.item() if config.GEPC else 0.0
        total_error += mre.item()
        total_error_next += mre_next.item()

        total_dab += loss_dab.item() if config.dab_weight > 0 else 0.0
        total_adv_E += loss_adv_E.item() if config.ADV else 0.0
        total_adv_D += loss_adv_D.item() if config.ADV else 0.0

        total_cls += (
            loss_cls.item() if config.get("cell_type_classifier", True) else 0.0
        )
        total_pert += (
            loss_pert.item() if config.get("genotype_classifier", True) else 0.0
        )
        total_ps += loss_ps.item() if config.ps_weight > 0 else 0.0
        total_ps_next += loss_ps_next.item() if ps_next_training_weight > 0 else 0.0
        if flow_losses is not None:
            total_flow_velocity += flow_losses["loss_velocity"].item()
            total_flow_endpoint_latent += flow_losses["loss_endpoint_latent"].item()
            total_flow_endpoint_nb += flow_losses["loss_endpoint_nb"].item()
            total_flow_total += flow_losses["loss_total"].item()

        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_mse_next = total_mse_next / log_interval
            cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
            cur_gepc_next = total_gepc_next / log_interval if config.GEPC else 0.0
            cur_error = total_error / log_interval
            cur_error_next = total_error_next / log_interval
            cur_dab = total_dab / log_interval if config.dab_weight > 0 else 0.0
            cur_adv_E = total_adv_E / log_interval if config.ADV else 0.0
            cur_adv_D = total_adv_D / log_interval if config.ADV else 0.0
            cur_cls = (
                total_cls / log_interval
                if config.get("cell_type_classifier", True)
                else 0.0
            )
            cur_pert = (
                total_pert / log_interval
                if config.get("genotype_classifier", True)
                else 0.0
            )
            cur_ps = total_ps / log_interval if config.ps_weight > 0 else 0.0
            cur_ps_next = (
                total_ps_next / log_interval if ps_next_training_weight > 0 else 0.0
            )
            cur_flow_velocity = (
                total_flow_velocity / log_interval if flow_matching_enabled else 0.0
            )
            cur_flow_endpoint_latent = (
                total_flow_endpoint_latent / log_interval
                if flow_matching_enabled
                else 0.0
            )
            cur_flow_endpoint_nb = (
                total_flow_endpoint_nb / log_interval if flow_matching_enabled else 0.0
            )
            cur_flow_total = (
                total_flow_total / log_interval if flow_matching_enabled else 0.0
            )
            # ppl = math.exp(cur_loss)
            if logger is not None:
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.8f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                    f"mse_next {cur_mse_next:5.2f} | mre_next {cur_error_next:5.2f} |"
                    f"cls {cur_cls:5.2f} | pert {cur_pert:5.2f} | ps {cur_ps:5.2f} | ps_next {cur_ps_next:5.2f} |"
                    + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
                    + (f"gepc_next {cur_gepc_next:5.2f} |" if config.GEPC else "")
                    + (
                        f"flow {cur_flow_total:5.2f} | flow_v {cur_flow_velocity:5.2f} | flow_z {cur_flow_endpoint_latent:5.2f} | flow_nb {cur_flow_endpoint_nb:5.2f} |"
                        if flow_matching_enabled
                        else ""
                    )
                    + (f"dab {cur_dab:5.2f} |" if config.dab_weight > 0 else "")
                    + (f"adv_E {cur_adv_E:5.2f} |" if config.ADV else "")
                    + (f"adv_D {cur_adv_D:5.2f} |" if config.ADV else "")
                )
            total_loss = 0
            total_mse = 0
            total_mse_next = 0
            total_gepc = 0
            total_gepc_next = 0
            total_error = 0
            total_error_next = 0
            total_dab = 0
            total_adv_E = 0
            total_adv_D = 0
            total_cls = 0
            total_pert = 0
            total_ps = 0
            total_ps_next = 0
            total_flow_velocity = 0
            total_flow_endpoint_latent = 0
            total_flow_endpoint_nb = 0
            total_flow_total = 0

            start_time = time.time()


def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mse_next", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre_next", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/cls", summary="min", step_metric="epoch")
    wandb.define_metric("valid/pert", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mvc", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mvc_next", summary="min", step_metric="epoch")
    wandb.define_metric("valid/ps", summary="min", step_metric="epoch")
    wandb.define_metric("valid/ps_next", summary="min", step_metric="epoch")
    wandb.define_metric("valid/flow_velocity", summary="min", step_metric="epoch")
    wandb.define_metric(
        "valid/flow_endpoint_latent", summary="min", step_metric="epoch"
    )
    wandb.define_metric("valid/flow_endpoint_nb", summary="min", step_metric="epoch")
    wandb.define_metric("valid/flow_total", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


def evaluate(
    model: nn.Module, loader: DataLoader, config, vocab, epoch=0, device=None
) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    criterion = masked_mse_loss
    criterion_dab = nn.CrossEntropyLoss()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_pert = nn.CrossEntropyLoss()
    criterion_adv = nn.CrossEntropyLoss()  # consider using label smoothing
    criterion_ps = nn.MSELoss()  # this is the loss for predicting PS scores
    criterion_mvc = GenerativeExpressionLoss()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    total_loss = 0.0
    total_loss_next = 0.0
    total_error = 0.0
    total_error_next = 0.0
    total_dab = 0.0
    total_cls = 0.0
    total_pert = 0.0
    total_ps = 0.0
    total_ps_next = 0.0
    total_num = 0
    total_mvc = 0
    total_mvc_next = 0
    total_flow_velocity, total_flow_endpoint_latent = 0.0, 0.0
    total_flow_endpoint_nb, total_flow_total = 0.0, 0.0
    flow_matching_enabled = _uses_flow_matching(model, config)

    if hasattr(config, "pred_lochness_next"):
        has_lochness_next_pred = True
        ps_next_training_weight = config.pred_lochness_next
    else:
        has_lochness_next_pred = False
        ps_next_training_weight = config.ps_weight * config.next_weight
    needs_next_perturbation_labels = _needs_next_perturbation_labels(
        config,
        flow_matching_enabled,
        has_lochness_next_pred,
    )

    with torch.no_grad():
        for batch, batch_data in enumerate(loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            target_values_next = batch_data["target_values_next"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)  # added
            perturbation_labels = batch_data["perturbation_labels"].to(device)  # added
            perturbation_labels_next = batch_data["perturbation_labels_next"].to(
                device
            )  # added
            ps_score = batch_data["ps"].to(device)  # added
            ps_score_next = batch_data["ps_next"].to(device)  # added
            sf = batch_data["sf"].to(device)
            sf_next_raw = batch_data["sf_next"].to(device)
            sf_next = sf_next_raw if flow_matching_enabled else sf
            next_gene_ids = (
                batch_data["next_gene_ids"].to(device)
                if flow_matching_enabled
                else None
            )
            next_src_key_padding_mask = (
                next_gene_ids.eq(vocab[config.pad_token])
                if next_gene_ids is not None
                else None
            )
            src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])
            mvc_src = (
                None
                if config.get("mvc_masked_train", True)
                else batch_data["full_gene_ids"].to(device)
            )
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels
                    if config.use_batch_label
                    else None,  # if config.DSBN else None,
                    pert_labels=perturbation_labels
                    if config.perturbation_input
                    else None,
                    pert_labels_next=perturbation_labels_next
                    if needs_next_perturbation_labels
                    else None,
                    next_gene_ids=next_gene_ids,
                    next_values=target_values_next if flow_matching_enabled else None,
                    next_src_key_padding_mask=next_src_key_padding_mask,
                    sf=sf,
                    sf_next=sf_next,
                    MVC=config.GEPC,
                    ECS=config.ecs_thres > 0,
                    CLS=config.get("cell_type_classifier", True),
                    PERTPRED=config.get("genotype_classifier", True),
                    PSPRED=config.ps_weight > 0,
                    mvc_src=mvc_src,
                )
                output_values = output_dict["mlm_output"]

                masked_positions = input_values.eq(config.mask_value)
                loss = criterion(output_values, target_values, masked_positions)
                # import pdb; pdb.set_trace()
                # print(f"total mask:{sum(masked_positions)}")
                # print(f"output_values_shape: {output_values.shape}")
                # print(output_values * masked_positions )
                # print(f"target_values_shape: {target_values.shape}")
                # print(target_values * masked_positions )

                loss_mse_next = criterion(
                    output_values, target_values_next, masked_positions
                )
                # print(f"target_values_next_shape: {target_values_next.shape}")
                # print(target_values_next * masked_positions)
                if config.GEPC:
                    mvc_target_values = (
                        target_values
                        if config.get("mvc_masked_train", True)
                        else batch_data["full_expr"].to(device)
                    )
                    mvc_target_values_next = (
                        target_values_next
                        if config.get("mvc_masked_train", True)
                        else batch_data["full_expr_next"].to(device)
                    )
                    # mvc_masked_positions = masked_positions if config.get('mvc_masked_train', True) else None

                    loss_gepc = loss_gepc = criterion_mvc(
                        output_dict["mvc_output"],
                        mvc_target_values,
                        scale_factor=sf,
                    )
                    loss_gepc_next = criterion_mvc(
                        output_dict["mvc_output_next"],
                        mvc_target_values_next,
                        scale_factor=sf_next,
                    )

                if config.dab_weight > 0:
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                if config.get("cell_type_classifier", True):  # added
                    loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                    # = loss + loss_cls
                if config.get("genotype_classifier", True):
                    loss_pert = criterion_pert(
                        output_dict["pert_output"], perturbation_labels
                    )
                    # = loss + loss_pert

                if config.ps_weight > 0:
                    loss_ps = criterion_ps(output_dict["ps_output"], ps_score)
                    # = loss + loss_pert
                if ps_next_training_weight > 0:
                    loss_ps_next = criterion_ps(
                        output_dict["ps_output_next"], ps_score_next
                    )
                flow_losses = _compute_flow_losses(
                    output_dict,
                    config,
                    criterion_mvc,
                    target_values_next,
                    batch_data["full_expr_next"].to(device),
                    masked_positions,
                    sf_next_raw,
                )

            total_loss += loss.item() * len(input_gene_ids)
            total_loss_next += loss_mse_next.item() * len(input_gene_ids)
            if config.GEPC:
                total_mvc += loss_gepc.item() * len(input_gene_ids)
                total_mvc_next += loss_gepc_next.item() * len(input_gene_ids)
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item() * len(input_gene_ids)
            total_error_next += masked_relative_error(
                output_values, target_values_next, masked_positions
            ).item() * len(input_gene_ids)
            if config.dab_weight > 0:
                total_dab += loss_dab.item() * len(input_gene_ids)
            if config.get("cell_type_classifier", True):  # added
                total_cls += loss_cls.item() * len(input_gene_ids)
            if config.get("genotype_classifier", True):
                total_pert += loss_pert.item() * len(input_gene_ids)
            if config.ps_weight > 0:
                total_ps += loss_ps.item() * len(input_gene_ids)
            if ps_next_training_weight > 0:
                total_ps_next += loss_ps_next.item() * len(input_gene_ids)
            if flow_losses is not None:
                total_flow_velocity += flow_losses["loss_velocity"].item() * len(
                    input_gene_ids
                )
                total_flow_endpoint_latent += flow_losses[
                    "loss_endpoint_latent"
                ].item() * len(input_gene_ids)
                total_flow_endpoint_nb += flow_losses["loss_endpoint_nb"].item() * len(
                    input_gene_ids
                )
                total_flow_total += flow_losses["loss_total"].item() * len(
                    input_gene_ids
                )
            total_num += len(input_gene_ids)

    wandb.log(
        {
            "valid/mse": total_loss / total_num,
            "valid/mse_next": total_loss_next / total_num,
            "valid/mvc": total_mvc / total_num,
            "valid/mvc_next": total_mvc_next / total_num,
            "valid/mre": total_error / total_num,
            "valid/mre_next": total_error_next / total_num,
            "valid/dab": total_dab / total_num,
            "valid/cls": total_cls / total_num,
            "valid/pert": total_pert / total_num,
            "valid/ps": total_ps / total_num,
            "valid/ps_next": total_ps_next / total_num,
            "valid/flow_velocity": total_flow_velocity / total_num,
            "valid/flow_endpoint_latent": total_flow_endpoint_latent / total_num,
            "valid/flow_endpoint_nb": total_flow_endpoint_nb / total_num,
            "valid/flow_total": total_flow_total / total_num,
            "valid/sum_mse_dab": (total_loss + config.dab_weight * total_dab)
            / total_num,
            "epoch": epoch,
        },
    )

    return (
        total_loss / total_num,
        total_loss_next / total_num,
        total_mvc / total_num,
        total_mvc_next / total_num,
        total_error / total_num,
        total_error_next / total_num,
        total_dab / total_num,
        total_cls / total_num,
        total_pert / total_num,
        total_ps / total_num,
        total_ps_next / total_num,
        total_flow_velocity / total_num,
        total_flow_endpoint_latent / total_num,
        total_flow_endpoint_nb / total_num,
        total_flow_total / total_num,
    )


def eval_testdata(
    model: nn.Module,
    adata_t: AnnData,
    gene_ids: List[str],
    train_data_dict: Dict,
    config,
    include_types: List[str] = ["cls", "pert"],
    input_layer_key="X_binned",
    next_layer_key="X_binned_next",
    logger=None,
    epoch=0,
    eval_key="",  # titles for evaluation
    make_plots=True,
    predict_expr=False,
    mvc_full_expr=False,
    sizefactor=False,
    sample=False,
) -> Optional[Dict]:  # Returns a dictionary containing the AnnData object
    """
    Evaluate the model on test data and return an AnnData object with embeddings.
    Plotting and UMAP are offloaded to a separate process.
    """
    logger = create_logger() if logger is None else logger
    model.eval()

    # copy adata_t to avoid reuse previously computed results stored in adata_t
    adata_t = adata_t.copy()  # make sure it is a independent copy for faster loading

    cell_type_to_index = train_data_dict["cell_type_to_index"]
    genotype_to_index = train_data_dict["genotype_to_index"]
    vocab = train_data_dict["vocab"]
    # make sure adata_t is ready for training
    shared_genes = adata_t.var.index.isin(list(vocab.stoi.keys()))
    logger.info(
        f"{sum(shared_genes)} genes shared between model vocab's {len(vocab)} and anndata's {adata_t.shape[1]} genes"
    )
    adata_t = adata_t[:, shared_genes]
    # adata_t = adata_t[adata_t.obs['celltype'].isin(cell_type_to_index)] if 'celltype' in adata_t.obs.columns else adata_t
    # adata_t = adata_t[adata_t.obs['genotype'].isin(genotype_to_index)] if 'genotype' in adata_t.obs.columns else adata_t
    gene_ids = vocab(adata_t.var.index.tolist())
    if "genotype_next" in adata_t.obs.keys():
        adata_t = adata_t[adata_t.obs["genotype_next"].isin(genotype_to_index)]
    all_counts = (
        adata_t.layers[input_layer_key].toarray()
        if issparse(adata_t.layers[input_layer_key])
        else adata_t.layers[input_layer_key]
    )
    from ..utils.pert_data_loader import _get_sf

    sf = _get_sf(all_counts) if sizefactor else None
    if next_layer_key in adata_t.layers:
        all_counts_next = (
            adata_t.layers[next_layer_key].toarray()
            if issparse(adata_t.layers[next_layer_key])
            else adata_t.layers[next_layer_key]
        )
    else:
        all_counts_next = None

    if (
        "celltype" in adata_t.obs.columns
        and config.cell_type_classifier
        and adata_t.obs["celltype"].isin(cell_type_to_index).all()
    ):
        celltypes_labels = adata_t.obs["celltype"].tolist()  # make sure count from 0
        celltypes_labels = np.array(celltypes_labels)
        celltypes_labels = np.array(
            [
                cell_type_to_index[ctype] if ctype in genotype_to_index else 0
                for ctype in celltypes_labels
            ]
        )
    else:
        # celltypes_labels = np.array(random.choices( [0,1], k=adata_t.shape[0]))
        celltypes_labels = None

    if "genotype" in adata_t.obs.columns and (
        config.perturbation_classifier_weight > 0 or config.perturbation_input
    ):
        perturbation_labels = adata_t.obs["genotype"].tolist()  # make sure count from 0
        perturbation_labels = np.array(perturbation_labels)
        perturbation_labels = np.array(
            [
                genotype_to_index[pert] if pert in genotype_to_index else 0
                for pert in perturbation_labels
            ]
        )
    else:
        # perturbation_labels = np.array(random.choices( [0,1], k=adata_t.shape[0]))
        perturbation_labels = None

    perturbation_indexes = perturbation_labels

    # evaluate the next prediction?
    next_cell_prediction = False
    perturbation_labels_next = None
    if config.next_cell_pred_type == "pert":
        if "genotype_next" in adata_t.obs.columns:
            # this is the pred_next
            # if config.perturbation_classifier_weight > 0:
            next_cell_prediction = True
        else:
            logger.warning(
                "next cell pred is set to pert but the provided adata does not have genotype_next column"
            )
            next_cell_prediction = False
        if next_cell_prediction:
            perturbation_labels_next = adata_t.obs[
                "genotype_next"
            ].tolist()  # make sure count from 0
    if config.next_cell_pred_type == "lochness":
        if hasattr(config, "pred_lochness_next") and config.pred_lochness_next > 0:
            next_cell_prediction = True
        else:
            next_cell_prediction = False
        if next_cell_prediction:
            perturbation_labels_next = adata_t.obs[
                "genotype_next"
            ].tolist()  # make sure count from 0

    if next_cell_prediction:
        perturbation_labels_next = np.array(perturbation_labels_next)
        perturbation_labels_next = np.array(
            [
                genotype_to_index[perturbation_type]
                for perturbation_type in perturbation_labels_next
            ]
        )
    else:
        # perturbation_labels_next = random.choices( [0,1], k=adata_t.shape[0])
        perturbation_labels_next = None
    perturbation_indexes_next = perturbation_labels_next

    if "batch_id" in adata_t.obs.columns:  # and config.DSBN:
        batch_ids = adata_t.obs["batch_id"].tolist()
    else:
        batch_ids = random.choices([0, 1], k=adata_t.shape[0])

    batch_ids = np.array(batch_ids)

    if mvc_full_expr:  # if we want to get full expression from mvc decoder
        cls_gene_ids = np.insert(
            gene_ids, 0, vocab[config.cls_token]
        )  # default should always be to insert a cls token at the front
        full_gene_ids = torch.stack(
            [torch.from_numpy(cls_gene_ids).long() for i in range(adata_t.shape[0])],
            dim=0,
        )
    else:
        full_gene_ids = None
    # Evaluate cls cell embeddings
    if "cls" in include_types:
        sampling_mode = config.get("sampling_mode", "simple")
        hvg_inds = None
        max_seq_len = config.max_seq_len
        if sampling_mode == "expressed":
            max_seq_len = (
                10000  # 10k genes should include all expressed genes for most 10X data
            )
        elif sampling_mode == "hvg":
            hvg_col = config.get("hvg_col", "highly_variable")
            assert hvg_col in adata_t.var.keys(), (
                "adata must have calculated HVGs or adata.var must have hvg_col"
            )
            hvg_inds = (
                np.where(adata_t.var[hvg_col])[0],
                np.where(~adata_t.var[hvg_col])[0],
            )
            max_seq_len = adata_t.var[hvg_col].sum() + config.get("non_hvg_size", 1000)

        tokenized_all, gene_idx_list = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=max_seq_len,
            vocab=vocab,
            pad_token=config.pad_token,
            pad_value=config.pad_value,
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=True,
            sampling_mode=config.get("sampling_mode", "simple"),
            nonzero_prop=config.get("nonzero_prop", 0.7),
            fix_nonzero_prop=config.get("fix_nonzero_prop", False),
            hvg_inds=hvg_inds,
            non_hvg_size=config.get("non_hvg_size", 1000),
        )

        all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
        if logger is not None:
            logger.info(
                f"Evaluating data using {all_gene_ids.shape[1]} tokens for each cell"
            )

        if next_layer_key in adata_t.layers:
            tokenized_all_next, _ = tokenize_and_pad_batch(
                all_counts_next,
                gene_ids,
                max_len=max_seq_len,
                vocab=vocab,
                pad_token=config.pad_token,
                pad_value=config.pad_value,
                append_cls=True,  # append <cls> token at the beginning
                include_zero_gene=True,
                sample_indices=gene_idx_list,
                sampling_mode=config.get("sampling_mode", "simple"),
                nonzero_prop=config.get("nonzero_prop", 0.7),
                fix_nonzero_prop=config.get("fix_nonzero_prop", False),
                hvg_inds=hvg_inds,
                non_hvg_size=config.get("non_hvg_size", 1000),
            )
            all_gene_ids_next, all_values_next = (
                tokenized_all_next["genes"],
                tokenized_all_next["values"],
            )

        src_key_padding_mask = all_gene_ids.eq(vocab[config.pad_token])
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=config.amp):
            # cell_embeddings = model.encode_batch(all_gene_ids,all_values.float(),
            #    src_key_padding_mask=src_key_padding_mask,
            #    batch_size=config.batch_size,
            #    batch_labels=torch.from_numpy(batch_ids).long() if config.use_batch_label else None, # if config.DSBN else None,
            #    time_step=0,
            #    return_np=True,
            # )
            (
                cell_embeddings,
                cell_embeddings_next,
                pert_preds,
                cls_preds,
                ps_preds,
                ps_preds_next,
                expr_dict,
            ) = model.encode_batch_with_perturb(
                all_gene_ids,
                all_values.float(),
                src_key_padding_mask=src_key_padding_mask,
                batch_size=config.batch_size,
                batch_labels=torch.from_numpy(batch_ids).long()
                if config.use_batch_label
                else None,  # if config.DSBN else None,
                pert_labels=torch.from_numpy(perturbation_indexes).long()
                if config.perturbation_input
                else None,
                pert_labels_next=torch.from_numpy(perturbation_indexes_next).long()
                if next_cell_prediction
                else None,
                sf=torch.Tensor(sf) if sizefactor else None,
                time_step=0,
                return_np=True,
                predict_expr=predict_expr,
                mvc_src=full_gene_ids,
                sample=sample,
            )

        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )
        cell_embeddings_next = cell_embeddings_next / np.linalg.norm(
            cell_embeddings_next, axis=1, keepdims=True
        )
        adata_t.obsm["X_scGPT"] = cell_embeddings

        adata_t.obsm["X_scGPT_next"] = cell_embeddings_next
        # adata_t.obsm["X_pert_pred"] = pert_preds
        if config.ps_weight > 0:
            adata_t.obsm["ps_pred"] = ps_preds
        if config.next_cell_pred_type == "lochness":
            adata_t.obsm["ps_pred_next"] = ps_preds_next
        for k in expr_dict:
            adata_t.obsm[k] = expr_dict[k]
        # require: genotype_to_index

        # Assuming ret_adata.obsm['X_pert_pred'] is a numpy array or can be converted to one

        # TODO: This is hardcoded to for genotype and celltype, change to a function in the future (will require major refactor of model and loader code)
        # Convert logits to probabilities using softmax
        X_genotype_cls_probs = np.exp(pert_preds) / np.sum(
            np.exp(pert_preds), axis=1, keepdims=True
        )
        # Assign the probabilities back to the AnnData object
        adata_t.obsm["X_pert_pred_probs"] = (
            X_genotype_cls_probs  # backward compatibility
        )
        adata_t.obsm["genotype_pred_probs"] = X_genotype_cls_probs
        # prompt: convert X_pert_pred_probs, which is the probabilities of each label, into label predictions, whose order is defined in genotype_to_index
        # Convert probabilities to predicted labels
        label_predictions = np.argmax(X_genotype_cls_probs, axis=1)
        # Map predicted indices back to genotypes using genotype_to_index
        # Assuming genotype_to_index is a dictionary where keys are indices and values are genotypes
        index_to_genotype = {v: k for k, v in genotype_to_index.items()}
        predicted_genotypes = [index_to_genotype[i] for i in label_predictions]
        # Add the predicted genotypes to the AnnData object
        adata_t.obs["predicted_genotype"] = predicted_genotypes
        if perturbation_labels is not None:
            adata_t.obs["genotype_id"] = (
                adata_t.obs["genotype"]
                .map(genotype_to_index)
                .astype(
                    pd.CategoricalDtype(categories=list(genotype_to_index.values()))
                )
            )

        X_celltype_cls_probs = np.exp(cls_preds) / np.sum(
            np.exp(cls_preds), axis=1, keepdims=True
        )
        adata_t.obsm["X_cls_pred_probs"] = (
            X_celltype_cls_probs  # backward compatibility
        )
        adata_t.obsm["celltype_pred_probs"] = X_celltype_cls_probs
        label_predictions_cls = np.argmax(X_celltype_cls_probs, axis=1)
        index_to_celltype = {v: k for k, v in cell_type_to_index.items()}
        predicted_celltypes = [index_to_celltype[i] for i in label_predictions_cls]
        adata_t.obs["predicted_celltype"] = predicted_celltypes
        if celltypes_labels is not None:
            adata_t.obs["celltype_id"] = (
                adata_t.obs["celltype"]
                .map(cell_type_to_index)
                .astype(
                    pd.CategoricalDtype(categories=list(cell_type_to_index.values()))
                )
            )
    return adata_t


def wrapper_train(
    model,
    config,
    data_gen,
    logger=None,
    save_dir=None,
    device=None,
    eval_adata_dict: Dict = {},
):
    logger = create_logger() if logger is None else logger
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_batch_types = data_gen["num_batch_types"]
    vocab = data_gen["vocab"]
    best_val_loss = float("inf")
    best_avg_bio = 0.0
    best_model = None
    define_wandb_metrcis()

    if save_dir is None:
        save_dir = Path(
            f"./save/dev_{config.dataset_name}-{time.strftime('%b%d-%H-%M')}/"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

    staged_training_plan = _resolve_staged_training_plan(config)
    if staged_training_plan is not None:
        if staged_training_plan["start_epoch"] > config.epochs:
            raise ValueError(
                f"stage2_start_epoch={staged_training_plan['start_epoch']} exceeds total epochs={config.epochs}"
            )
        logger.info(
            "Configured staged training: epochs 1-%d use base config; epoch %d starts stage 2 with overrides %s",
            staged_training_plan["start_epoch"] - 1,
            staged_training_plan["start_epoch"],
            sorted(staged_training_plan["overrides"].keys()),
        )

    _initialize_runtime_training_controls(model, config, logger=logger)
    _load_initial_checkpoint_if_configured(model, config, device=device, logger=logger)
    _apply_trainable_module_whitelist(model, config, logger=logger)
    optimizer_dict = create_optimizer_dict(model, device, config, num_batch_types)

    # save the current configurations before epoch starts
    torch.save(vocab, save_dir / "vocab.pt")
    running_parameters = {
        "cell_type_to_index": data_gen["cell_type_to_index"],
        "genotype_to_index": data_gen["genotype_to_index"],
        "genes": data_gen["genes"],  # genes,
        "gene_ids": data_gen["gene_ids"],  # gene_ids,
        "ps_names": data_gen["ps_names"],
        "n_ps": data_gen.get("n_ps", 0),
        "num_batch_labels": data_gen["num_batch_types"],
        "config": config.as_dict(),  # config as dictionary
    }
    torch.save(running_parameters, save_dir / "running_parameters.pt")
    import json

    json.dump(config.as_dict(), open(save_dir / "config.json", "w"))
    # later, use the following to load json file
    # config_data = json.load(open(save_dir / 'config.json', 'r'))
    train_loader, valid_loader = data_gen["train_loader"], data_gen["valid_loader"]
    executor = ProcessPoolExecutor(
        max_workers=4,
        initializer=init_plot_worker,
        mp_context=multiprocessing.get_context("spawn"),
    )
    evaltest_processes = []
    eval_type = "cell-eval" if config.next_cell_pred_type == "pert" else "umap"
    for epoch in range(1, config.epochs + 1):
        if (
            staged_training_plan is not None
            and epoch == staged_training_plan["start_epoch"]
        ):
            logger.info("Entering staged training phase 2 at epoch %d", epoch)
            _apply_staged_training_overrides(
                model,
                config,
                staged_training_plan["overrides"],
                logger=logger,
            )
            if any(
                key in staged_training_plan["overrides"]
                for key in ("trainable_module_prefixes", "trainable_modules")
            ):
                _apply_trainable_module_whitelist(model, config, logger=logger)
            if _staged_training_requires_optimizer_reset(
                staged_training_plan["overrides"]
            ):
                optimizer_dict = create_optimizer_dict(
                    model, device, config, num_batch_types
                )
                logger.info(
                    "Recreated optimizer state at the stage boundary to match the updated trainable-parameter policy"
                )
            best_val_loss = float("inf")
            best_model = None
            best_avg_bio = 0.0
            logger.info(
                "Reset best-model tracking at the stage boundary so final evaluation is selected from post-transition epochs only"
            )

        epoch_start_time = time.time()
        # Clean up background UMAP and metric calculations on past test-data eval
        remaining_processes = []
        for p in evaltest_processes:
            if p.done():
                try:
                    result = p.result()
                    metrics_to_log = result["metrics"]
                    formatted_results = json.dumps(metrics_to_log, indent=4)
                    logger.info(f"Metrics:\n{formatted_results}")
                    for key, img_path in result["images"].items():
                        metrics_to_log[key] = wandb.Image(img_path)
                    if metrics_to_log:
                        wandb.log(metrics_to_log)
                    logger.info(
                        f"Finished {result['eval_dict_key']} {eval_type} for epoch {result['epoch']}"
                    )
                except Exception as e:
                    logger.warning(f"{eval_type} process failed due to: {e}")
            else:
                remaining_processes.append(p)
        # Joins the process to release resources
        evaltest_processes = remaining_processes
        logger.info(f"Active {eval_type} processes: {len(evaltest_processes)}")

        if config.do_train:
            train(
                model,
                train_loader,
                config,
                vocab,
                optimizer_dict,
                epoch=epoch,
                logger=logger,
                device=device,
            )
        (
            val_loss,
            val_loss_next,
            val_mvc,
            val_mvc_next,
            val_mre,
            val_mre_next,
            val_dab,
            val_cls,
            val_pert,
            val_ps,
            val_ps_next,
            val_flow_velocity,
            val_flow_endpoint_latent,
            val_flow_endpoint_nb,
            val_flow_total,
        ) = evaluate(
            model,
            loader=valid_loader,
            config=config,
            vocab=vocab,
            epoch=epoch,
            device=device,
        )
        elapsed = time.time() - epoch_start_time
        if logger is not None:
            logger.info("-" * 89)
            logger.info(
                f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                f"valid loss/mse {val_loss:5.4f} | mvc {val_mvc:5.4f} | "
                f"valid loss/mse_next {val_loss_next:5.4f} | mvc_next {val_mvc_next:5.4f} | "
                f"valid dab {val_dab:5.4f} | valid cls {val_cls:5.4f} | valid pert {val_pert:5.4f} |"
                f"valid ps {val_ps:5.4f} | valid ps_next {val_ps_next:5.4f} |"
                + (
                    f" valid flow {val_flow_total:5.4f} | flow_v {val_flow_velocity:5.4f} | flow_z {val_flow_endpoint_latent:5.4f} | flow_nb {val_flow_endpoint_nb:5.4f} |"
                    if _uses_flow_matching(model, config)
                    else ""
                )
            )
            logger.info("-" * 89)
        loss = val_loss * 0.1
        if config.next_cell_pred_type == "identity":
            loss += val_cls * int(config.cell_type_classifier) + val_pert * int(
                config.genotype_classifier
            )
        elif config.next_cell_pred_type == "pert":
            loss += val_mvc_next
        else:
            loss += val_ps + val_ps_next
        if _uses_flow_matching(model, config):
            loss += val_flow_total
        best_model_epoch = 0
        if loss < best_val_loss:
            best_val_loss = loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            if logger is not None:
                logger.info(f"Best model with score {best_val_loss:5.4f}")

        # if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
        eval_expr_interval = max(
            round(2e5 / len(train_loader.dataset)),
            abs(config.get("eval_expr_interval", 2)) // 2 * 2,
        )  # this must be even to match the save interval below
        predict_expr_tmp = (
            True
            if eval_expr_interval
            and epoch % eval_expr_interval == 1
            and config.get("next_cell_pred_type") == "pert"
            else False
        )
        save_eval_interval = config.get("save_eval_interval", 2)
        if epoch % save_eval_interval == 1:
            logger.info(f"Saving model to {save_dir}")
            torch.save(best_model.state_dict(), save_dir / "best_model.pt")
            torch.save(model.state_dict(), save_dir / f"model_e{epoch}.pt")
        if epoch % eval_expr_interval == 1:
            # logger.info(f"Saving model to {save_dir}")
            save_dir2 = save_dir / f"e{epoch}_imgs"
            save_dir2.mkdir(parents=True, exist_ok=True)
            for key, item in eval_adata_dict.items():
                # Step 1: Get AnnData with embeddings from the main process
                if config.next_cell_pred_type == "pert":
                    eval_adata = item[0]
                    true = item[1]
                else:
                    eval_adata = item

                results = eval_testdata(
                    # best_model,
                    model,  # use current model
                    adata_t=eval_adata,  # adata_t=data_gen['adata_sorted'], # if config.per_seq_batch_sample else adata,
                    gene_ids=data_gen["gene_ids"],
                    train_data_dict=data_gen,
                    config=config,
                    include_types=["cls"],
                    logger=logger,
                    epoch=epoch,
                    eval_key=key,
                    predict_expr=predict_expr_tmp,
                    mvc_full_expr=predict_expr_tmp,
                    sizefactor=True,
                    sample=True,
                )
                adata_with_embeddings = results
                # Step 2: Save the data to a temporary file for the child process
                # temp_adata_path = save_dir2 / f"temp_adata_{eval_dict_key}_e{epoch}.h5ad"
                # adata_with_embeddings.write_h5ad(temp_adata_path)

                # Step 3: Create and start the background process
                logger.info(
                    f"Starting background process for UMAP on epoch {epoch} for '{key}'"
                )

                # Pass data_gen['ps_names'] if it exists, otherwise None
                ps_names = data_gen.get("ps_names", None)
                # p = multiprocessing.Process(
                #   target=process_and_log_umaps,
                #  args=(adata_with_embeddings, config, epoch, eval_dict_key, save_dir2, ps_names)
                # )
                # p.start()
                # t =cell_eval_to_wandb(true, results, save_dir, epoch, key)
                if config.next_cell_pred_type == "pert":
                    p = executor.submit(
                        cell_eval_to_wandb,
                        true,
                        results,
                        save_dir,
                        epoch,
                        key,
                        config.get("min_eval_cells", 30),
                        model.distribution,
                    )
                else:
                    p = executor.submit(
                        process_and_log_umaps,
                        adata_with_embeddings,
                        OmegaConf.structured(dict(config)),
                        epoch,
                        key,
                        save_dir2,
                        ps_names,
                    )
                evaltest_processes.append(p)

            # metrics_to_log["test/best_model_epoch"] = best_model_epoch
            wandb.log({"test/best_model_epoch": best_model_epoch})
            # wandb.log({"avg_bio": results.get("avg_bio", 0.0)})

        optimizer_dict["scheduler"].step()

        if optimizer_dict["DAB_separate_optim"]:
            optimizer_dict["scheduler_dab"].step()
        if config.ADV:
            optimizer_dict["scheduler_D"].step()
            optimizer_dict["scheduler_E"].step()

    # One final gather for the background processes
    for p in evaltest_processes:
        if p.done():
            try:
                result = p.result()
                metrics_to_log = result["metrics"]
                formatted_results = json.dumps(metrics_to_log, indent=4)
                logger.info(f"Metrics:\n{formatted_results}")
                for key, img_path in result["images"].items():
                    metrics_to_log[key] = wandb.Image(img_path)
                if metrics_to_log:
                    wandb.log(metrics_to_log)
                logger.info(
                    f"Finished {result['eval_dict_key']} {eval_type} for epoch {result['epoch']}"
                )
            except Exception as e:
                logger.warning(f"{eval_type} process failed due to: {e}")
        else:
            remaining_processes.append(p)
    # save the best model
    if best_model is None:
        best_model = copy.deepcopy(model)
    torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    torch.save(model.state_dict(), save_dir / "final_model.pt")
    logger.info("Saved terminal checkpoint to %s", save_dir / "final_model.pt")

    return best_model
