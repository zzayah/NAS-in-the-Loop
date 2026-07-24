"""
Shared DNN architectures and serialisation helpers for F1TENTH.

Architecture overview (all produce a single scalar output):
    1 – tiny single-channel CNN (~134 params)
    2 – 3-stage single-channel CNN (~small)
    3 – [Conv(1→1), Conv(1→8)] → Linear(1072, 32) (~small)
    4 – [Conv(1→1), Conv(1→16)] → Linear(2144, 32) (~small-medium)
    5 – [Conv(1→8), Conv(8→16)] → Linear(2144, 64) (~medium)
    6 – [Conv(1→16), Conv(16→32)] → Linear(4288, 128) (~large)
"""

import io
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

def _build_dynamic_arch(model_cfg: dict[str, Any] | None = None) -> nn.Sequential:
    """
    Build a Conv1d→Linear network from a ``model.dynamic`` configuration.

    Expected fields under ``dynamic``::

        in_channels: int (default 1)
        input_length: int (default 1080)
        activation: "elu" or "relu" (default "elu")
        conv_layers: list of {out_channels, kernel_size?, stride?, padding?, pool_size?}
        fc_layers: list/int of hidden widths before the final scalar output
    """
    dyn = model_cfg.get("dynamic")
    conv_layers = dyn.get("conv_layers", [])
    in_channels = int(dyn.get("in_channels", 1))
    feature_length = int(dyn.get("input_length", 1080))
    activation = dyn.get("activation", "elu").lower()
    act_cls = nn.ReLU if activation == "relu" else nn.ELU
    layers: list[nn.Module] = []
    curr_channels = in_channels

    for idx, cfg in enumerate(conv_layers):
        out_channels = int(cfg["out_channels"])
        kernel_size = int(cfg.get("kernel_size", 3))
        stride = int(cfg.get("stride", 1))
        padding = int(cfg.get("padding", 0))
        pool_size = int(cfg.get("pool_size", 0))

        layers.append(
            nn.Conv1d(
                curr_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )
        layers.append(act_cls())

        feature_length = max(
            1, (feature_length + 2 * padding - kernel_size) // stride + 1
        )
        if pool_size and pool_size > 1:
            layers.append(nn.MaxPool1d(kernel_size=pool_size))
            feature_length = max(1, feature_length // pool_size)

        curr_channels = out_channels

    layers.append(nn.Flatten())
    flattened = feature_length * curr_channels

    fc_layers = dyn.get("fc_layers", [])
    if isinstance(fc_layers, int):
        fc_layers = [fc_layers]
    fc_layers = list(fc_layers)

    in_features = flattened
    for hidden in fc_layers:
        hidden = int(hidden)
        layers.append(nn.Linear(in_features, hidden))
        layers.append(act_cls())
        in_features = hidden

    layers.append(nn.Linear(in_features, 1)) # the last layer always has a nn.Linear with a single out feature
    return nn.Sequential(*layers)

def get_architecture(arch_id: int, model_cfg: dict[str, Any] | None = None) -> nn.Module:
    """
    Factory function for F1TENTH single-output LiDAR neural network architectures.

    All architectures accept a ``(batch, 1, 1080)`` tensor and return a
    ``(batch, 1)`` scalar — suitable for heading error, left wall distance,
    or track width prediction.

    Args:
        arch_id: An integer identifying the
            specific layer configuration (ordered roughly by parameter count).

    Returns:
        An uninitialised ``nn.Sequential`` module.

    Raises:
        ValueError: If ``arch_id`` is not in the supported set.
    """
    factories = {
        1: lambda: nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=8),
            nn.Flatten(),
            nn.Linear(134, 1),
        ),
        2: lambda: nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(1, 1, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(1, 1, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(32, 8),
            nn.ELU(),
            nn.Linear(8, 1),
        ),
        3: lambda: nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(1, 8, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(1072, 32),
            nn.ELU(),
            nn.Linear(32, 1),
        ),
        4: lambda: nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(1, 16, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(2144, 32),
            nn.ELU(),
            nn.Linear(32, 1),
        ),
        5: lambda: nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(8, 16, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(2144, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        ),
        6: lambda: nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(4288, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        ),
        7: lambda: _build_dynamic_arch(model_cfg), # add dropout layer after maxPooling (0-20% dropout)
    }

    if arch_id not in factories:
        raise ValueError(
            f"Architecture ID {arch_id} is not supported. "
            f"Valid IDs are: {sorted(factories.keys())}"
        )

    return factories[arch_id]()


# ---------------------------------------------------------------------------
# Self-sufficient model serialisation helpers
# ---------------------------------------------------------------------------


def save_as_torchscript(
    model: nn.Module,
    path: str | Path,
    metadata: dict[str, Any],
) -> None:
    """Save an ``nn.Module`` as a self-sufficient TorchScript ``.pt`` file.

    The saved file embeds both the full computation graph (via
    ``torch.jit.script``) and all metadata (``arch_id``, ``target_col``,
    training config, quantization flag, …) as a JSON sidecar inside the same
    archive.  No separate architecture class or external config file is
    required to load a non-quantized model.

    For quantized models (``metadata["quantized"] is True``) the *FP32*
    computation graph is scripted and the FP32 ``state_dict`` is additionally
    embedded as a binary extra file.  This is necessary because
    ``torchao.AffineQuantizedTensor`` weights are not TorchScript-serialisable.
    At load time the caller rebuilds the module, loads the FP32 weights, and
    re-applies ``quantize_()``—which is semantically equivalent since torchao
    INT8 weight quantisation is deterministic given the FP32 weights.

    Args:
        model: The FP32 ``nn.Module`` to script and save.  Must be in eval
            mode and **not** yet quantized.
        path: Destination path for the ``.pt`` file.
        metadata: Dict of metadata to embed (must be JSON-serialisable).
            Should contain at minimum ``"quantized"`` (bool).  For quantized
            models ``"arch_id"`` is also required so the model can be
            faithfully reconstructed at load time.
    """
    path = Path(path)
    extra_files: dict[str, bytes] = {
        "metadata.json": json.dumps(metadata).encode(),
    }

    # For quantized models embed the FP32 state_dict as well.
    # This lets _load_model reconstruct the quantized module without the
    # ScriptModule state_dict key-name complications.
    if metadata.get("quantized", False):
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        extra_files["state_dict.pt"] = buf.getvalue()

    if metadata.get("quantized", False):
        # torchao AffineQuantizedTensor / LinearActivationQuantizedTensor
        # weights are not TorchScript-serialisable, so we script a freshly
        # instantiated FP32 architecture for the computation graph but store
        # the quantized state_dict so it can be loaded directly after
        # re-applying quantize_() at inference time.
        arch_id: int = metadata["arch_id"]
        fp32_for_graph = get_architecture(arch_id)
        fp32_for_graph.eval()
        scripted = torch.jit.script(fp32_for_graph)
    else:
        scripted = torch.jit.script(model)
    torch.jit.save(scripted, str(path), _extra_files=extra_files)


def load_torchscript(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[torch.ScriptModule, dict[str, Any]]:
    """Load a self-sufficient TorchScript model saved by :func:`save_as_torchscript`.

    For *non-quantized* models the returned :class:`torch.ScriptModule` is
    ready for inference with **no** additional imports required — no
    ``nn_models.py``, no ``arch_id``, no config file.

    For *quantized* models (``metadata["quantized"] is True``), the returned
    module is the FP32 TorchScript graph.  Use the ``"arch_id"`` and the
    ``"_state_dict_bytes"`` private key in *metadata* to reconstruct the
    quantized module::

        scripted, meta = load_torchscript(path)
        if meta["quantized"]:
            model = get_architecture(meta["arch_id"])
            buf = io.BytesIO(meta["_state_dict_bytes"])
            model.load_state_dict(torch.load(buf, map_location=device))
            model.eval()
            quantize_(model, Int8DynamicActivationInt8WeightConfig())

    Args:
        path: Path to the ``.pt`` file produced by :func:`save_as_torchscript`.
        map_location: Device to map tensors to on load.

    Returns:
        A ``(scripted_model, metadata)`` tuple where *metadata* mirrors the
        dict passed to :func:`save_as_torchscript`.  For quantized files an
        extra ``"_state_dict_bytes"`` key (raw ``bytes``) is injected by this
        function for downstream reconstruction.
    """
    extra: dict[str, bytes] = {"metadata.json": b"", "state_dict.pt": b""}
    scripted = torch.jit.load(str(path), _extra_files=extra, map_location=map_location)
    metadata: dict[str, Any] = json.loads(extra["metadata.json"].decode())
    if extra["state_dict.pt"]:
        # Attach raw bytes under a private key; caller decides whether to use them.
        metadata["_state_dict_bytes"] = extra["state_dict.pt"]
    return scripted, metadata
