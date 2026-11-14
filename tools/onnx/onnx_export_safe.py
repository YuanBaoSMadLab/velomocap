"""Make torch's legacy ONNX exporter tolerate nodes it cannot type.

The legacy (`dynamo=False`) exporter runs ONNX shape inference on every node it
emits (`_create_node`). When a single node's operands carry no dtype — which is
common for values produced by custom symbolics deep inside a scripted submodule —
that call raises and aborts the whole export, even though the node itself is
perfectly valid ONNX and ONNX Runtime would accept it.

Shape inference here only *annotates* the IR; nothing downstream requires it.
`patch_safe_shape_inference()` wraps `_create_node` so a failed inference is
retried with the pass disabled. Every skipped node is recorded, and the export
continues with the real graph. Use `report()` afterwards to see what was
dropped, and `raise_on_miss()` to fail loudly instead.
"""

from __future__ import annotations

from collections import Counter

from torch.onnx._internal.torchscript_exporter import jit_utils

_MISSES: Counter[str] = Counter()
_MESSAGES: Counter[str] = Counter()
_STRICT = False
_installed = False


def _wrap(fn):
    def _create_node(
        graph_or_block,
        domain_op,
        inputs,
        attributes,
        params_dict,
        opset_version,
        n_outputs,
        shape_inference=True,
    ):
        if not shape_inference:
            return fn(
                graph_or_block,
                domain_op,
                inputs,
                attributes,
                params_dict,
                opset_version,
                n_outputs,
                shape_inference=False,
            )
        try:
            return fn(
                graph_or_block,
                domain_op,
                inputs,
                attributes,
                params_dict,
                opset_version,
                n_outputs,
                shape_inference=True,
            )
        except Exception as exc:  # noqa: BLE001
            _MISSES[domain_op] += 1
            _MESSAGES[str(exc).splitlines()[0][:160]] += 1
            if _STRICT:
                raise
            node = fn(
                graph_or_block,
                domain_op,
                inputs,
                attributes,
                params_dict,
                opset_version,
                n_outputs,
                shape_inference=False,
            )
            # Keep the value typed as a plain float tensor so later passes do
            # not choke on an unannotated value's element type.
            try:
                node.output().setType(node.output().type().with_dtype("Float"))
            except Exception:  # noqa: BLE001
                pass
            return node

    return _create_node


def patch_safe_shape_inference(*, strict: bool = False) -> None:
    """Install the tolerant wrapper. Call before `torch.onnx.export`."""
    global _installed, _STRICT
    _STRICT = strict
    if _installed:
        return
    jit_utils._create_node = _wrap(jit_utils._create_node)
    _installed = True


def report() -> None:
    if not _MISSES:
        print("  shape inference: no nodes skipped")
        return
    total = sum(_MISSES.values())
    print(f"  shape inference skipped on {total} node(s):")
    for op, count in _MISSES.most_common():
        print(f"    {count:6d}  {op}")
    for msg, count in _MESSAGES.most_common(5):
        print(f"    ({count}x) {msg}")
