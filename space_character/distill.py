"""Module « distillé » lightx2v fusionné dans Wan-Animate : 4 étapes au lieu de 20, pour une qualité proche.

Wan-Animate est construit sur Wan2.1-I2V-14B : les 731 couches ciblées par le module lightx2v de ce modèle existent
toutes dans Wan-Animate (vérifié sur leurs index de poids). Distillé, le modèle donne en 4 étapes, sans guidage, une
image proche des 20 étapes officielles : le même temps de GPU qu'avant (6 étapes non distillées) pour un résultat
bien plus net. La fusion se fait dans les poids (W += up·down, biais += diff_b, autres paramètres += diff) : aucun
coût pendant le calcul. Couches enveloppées par l'adaptateur de rééclairage (peft) : fusion dans leur couche de base.
"""
from __future__ import annotations

import re

import torch

REPO = "lightx2v/Wan2.1-Distill-Loras"
FILE = "wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors"
STEPS = 4                      # nombre d'étapes pour lequel le module a été distillé
_PART = re.compile(r"^(?:diffusion_model\.)?(.+)\.(lora_down\.weight|lora_up\.weight|diff_b|diff|alpha)$")


def targets(lora: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Tenseurs du module regroupés par couche visée : {« blocks.0.self_attn.q » : {« lora_down.weight » : …}}."""
    groups: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in lora.items():
        m = _PART.match(name)
        if m:
            groups.setdefault(m[1], {})[m[2]] = tensor
    return groups


def _param(params: dict[str, torch.nn.Parameter], target: str, kind: str) -> torch.nn.Parameter | None:
    """Poids ou biais de la couche visée ; dans sa couche de base si peft l'a enveloppée ; ou le paramètre lui-même
    (ex. « blocks.0.modulation »)."""
    names = [f"{target}.base_layer.{kind}", f"{target}.{kind}"] + ([target] if kind == "weight" else [])
    return next((params[n] for n in names if n in params), None)


def _add(param: torch.nn.Parameter, delta: torch.Tensor) -> None:
    """Ajout calculé en float32 puis arrondi une seule fois au format du modèle (bf16)."""
    param.copy_((param.float() + delta.reshape(param.shape)).to(param.dtype))


@torch.no_grad()
def merge(model: torch.nn.Module, lora: dict[str, torch.Tensor], scale: float = 1.0) -> int:
    """Fusionne le module dans les poids du modèle (là où ils sont : GPU pendant un calcul). Renvoie le nombre de
    couches modifiées ; lève une erreur si une couche visée manque (mauvais modèle : rien n'est à moitié fusionné)."""
    params = dict(model.named_parameters())
    groups = targets(lora)
    missing = [t for t, parts in groups.items()
               if _param(params, t, "weight") is None or ("diff_b" in parts and _param(params, t, "bias") is None)]
    if missing:
        raise RuntimeError(f"Module distillé incompatible : {len(missing)} couches introuvables ({', '.join(missing[:3])}…)")
    for target, parts in groups.items():
        weight = _param(params, target, "weight")
        dev = weight.device
        if "lora_down.weight" in parts and "lora_up.weight" in parts:
            down = parts["lora_down.weight"].to(dev, torch.float32)
            up = parts["lora_up.weight"].to(dev, torch.float32)
            alpha = float(parts["alpha"]) if "alpha" in parts else down.shape[0]   # sans alpha : échelle 1
            _add(weight, (up.flatten(1) @ down.flatten(1)) * (scale * alpha / down.shape[0]))
        if "diff" in parts:
            _add(weight, parts["diff"].to(dev, torch.float32) * scale)
        if "diff_b" in parts:
            _add(_param(params, target, "bias"), parts["diff_b"].to(dev, torch.float32) * scale)
    return len(groups)
