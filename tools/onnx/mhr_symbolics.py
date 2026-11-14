"""ONNX symbolics for the scripted MHR submodule inside SAM 3D Body.

`mhr_model.pt` is TorchScript, so the exporter walks its frozen graph directly
and every operator in it has to have a lowering. torch's own table is missing
(or too strict about) a handful of them; this module supplies the rest and is
imported before `torch.onnx.export`.

Two things matter for correctness here:

* every scalar/vector constant is emitted as a **tensor** (`value_t`). An int
  attribute named `value` on `onnx::Constant` breaks ONNX shape inference as
  soon as a consumer op (a comparison, a `Slice`) reads it, which is exactly the
  "required keyword attribute 'value' has the wrong type" abort.
* `ReduceSum` in opset >= 13 takes `axes` as an *input*, not an attribute, so
  the reductions below build the axis tensor explicitly.
"""

from __future__ import annotations

import torch

OPSET = 17


# --------------------------------------------------------------------------
# small helpers over torch's symbolic helper
# --------------------------------------------------------------------------
def _sh():
    import torch.onnx.symbolic_helper as sh

    return sh


def _rank(value):
    sh = _sh()
    try:
        sizes = sh._get_tensor_sizes(value)
    except Exception:  # noqa: BLE001
        sizes = None
    if sizes is not None:
        return len(sizes)
    try:
        return sh._get_tensor_rank(value)
    except Exception:  # noqa: BLE001
        return None


def _const_int(g, value):
    """An int64 scalar Constant, as a tensor so consumers can infer a dtype."""
    return g.op("Constant", value_t=torch.tensor(int(value), dtype=torch.long))


def _const_ints(g, values):
    return g.op(
        "Constant", value_t=torch.tensor([int(v) for v in values], dtype=torch.long)
    )


def _const_axis(g, dim):
    sh = _sh()
    raw = None
    for kind in ("is", "i"):
        try:
            raw = sh._get_const(dim, kind, "dim")
            if raw is not None:
                break
        except Exception:  # noqa: BLE001
            pass
    return raw


# --------------------------------------------------------------------------
# the lowerings
# --------------------------------------------------------------------------
def _dim(g, self, **kwargs):
    """`aten::dim` → `Size(Shape(x))`, evaluated at run time.

    The static rank torch records for a traced value is unreliable inside the
    `while` loop that MHR's LBS uses to rank-match `inverse_bind_pose`: a stale
    value makes the loop run the wrong number of times and every downstream
    `Expand` then gets a target of the wrong rank. `Shape` is exact by
    construction, so it is used here instead.
    """
    return g.op("Size", g.op("Shape", self))


def _size(g, self, dim=None, **kwargs):
    """`aten::size` → ONNX `Shape`.

    Static sizes are not available for the intermediate quaternion / skeleton
    values MHR builds, but `Shape` needs no rank information at all. The
    two-argument form dispatches to the same registration, so both shapes are
    handled here.
    """
    if dim is not None:
        return _size_dim(g, self, dim)
    return g.op("Shape", self)


def _size_dim(g, self, dim=None, **kwargs):
    axis = _const_axis(g, dim)
    if axis is None:
        raise RuntimeError("aten::size: dim is not constant")
    shape = g.op("Shape", self)
    return g.op("Gather", shape, _const_int(g, axis), axis_i=0)


def _int_list(value):
    """A constant int list out of a symbolic argument, or None."""
    sh = _sh()
    if isinstance(value, (list, tuple)):
        try:
            return [int(v) for v in value]
        except Exception:  # noqa: BLE001
            return None
    for kind in ("is", "i"):
        try:
            raw = sh._get_const(value, kind, "size")
        except Exception:  # noqa: BLE001
            raw = None
        if raw is not None:
            items = raw if isinstance(raw, (list, tuple)) else [raw]
            try:
                return [int(v) for v in items]
            except Exception:  # noqa: BLE001
                return None
    try:
        raw = sh._parse_arg(value, "is")
    except Exception:  # noqa: BLE001
        return None
    if raw is None:
        return None
    try:
        return [int(v) for v in (raw if isinstance(raw, (list, tuple)) else [raw])]
    except Exception:  # noqa: BLE001
        return None


def _split_with_sizes(g, self, split_sizes, dim, **kwargs):
    """`aten::split_with_sizes` → one `Slice` per chunk.

    torch's symbolic needs `self.dim()`, which the MHR graph does not record for
    intermediate values. Slicing keeps every chunk a separate graph output with
    no shape information required at all.
    """
    sh = _sh()
    sizes = _int_list(split_sizes)
    if sizes is None:
        raise RuntimeError("aten::split: non-constant sizes")
    rank = _rank(self)
    if rank is None:
        raise RuntimeError("aten::split: rank unknown")
    axis = sh._get_const(dim, "i", "dim")
    if axis < 0:
        axis += rank

    outs = []
    start = 0
    for size in sizes:
        starts = [0] * rank
        ends = [2**31 - 1] * rank
        axes = list(range(rank))
        starts[axis] = start
        ends[axis] = start + size
        outs.append(
            g.op(
                "Slice",
                self,
                _const_ints(g, starts),
                _const_ints(g, ends),
                _const_ints(g, axes),
            )
        )
        start += size
    return tuple(outs)


def _cross(g, a, b, dim=None, **kwargs):
    """`aten::cross` on the last axis, expanded into arithmetic.

    torch's symbolic asserts on a known rank, which MHR's rank-agnostic values
    do not carry.
    """

    def comp(t, i):
        return g.op("Gather", t, _const_ints(g, [i]), axis_i=-1)

    a0, a1, a2 = (comp(a, i) for i in range(3))
    b0, b1, b2 = (comp(b, i) for i in range(3))

    mul = lambda x, y: g.op("Mul", x, y)  # noqa: E731
    sub = lambda x, y: g.op("Sub", x, y)  # noqa: E731
    return g.op(
        "Concat",
        sub(mul(a1, b2), mul(a2, b1)),
        sub(mul(a2, b0), mul(a0, b2)),
        sub(mul(a0, b1), mul(a1, b0)),
        axis_i=-1,
    )


def _hypot(g, a, b, **kwargs):
    return g.op("Sqrt", g.op("Add", g.op("Mul", a, a), g.op("Mul", b, b)))


def _reduce_sum(g, value, axes, keepdims):
    return g.op("ReduceSum", value, _const_ints(g, axes), keepdims_i=keepdims)


def _try_float(value):
    if value is None:
        return None
    for kind in ("f", "i"):
        try:
            v = _sh()._get_const(value, kind, "p")
            if v is not None:
                return float(v)
        except Exception:  # noqa: BLE001
            pass
    return None


def _norm(g, self, p=None, dim=None, keepdim=False, **kwargs):
    """`aten::norm` for `F.normalize`.

    MHR's scripted graph hands `p` over as a tensor constant that torch's own
    symbolic cannot parse; `normalize` only ever means p=2.
    """
    p_val = _try_float(p)
    if p_val is None:
        p_val = 2.0
    keep = 1 if keepdim else 0
    rank = _rank(self)

    if dim is None:
        if rank is None:
            flat = g.op("Flatten", self, axis_i=0)
            return g.op("Sqrt", _reduce_sum(g, g.op("Mul", flat, flat), [0], 0))
        axes = list(range(rank))
    else:
        raw = _const_axis(g, dim)
        if raw is None:
            raise RuntimeError("aten::norm: cannot resolve dim")
        # Keep the axis exactly as given (negative included): ONNX reductions
        # accept negative axes, and rewriting them needs a rank that the
        # values inside a scripted `Loop` body do not reliably carry.
        axes = list(raw) if isinstance(raw, (list, tuple)) else [raw]

    if abs(p_val - 2.0) < 1e-6:
        return g.op("Sqrt", _reduce_sum(g, g.op("Mul", self, self), axes, keep))
    if abs(p_val - 1.0) < 1e-6:
        return _reduce_sum(g, g.op("Abs", self), axes, keep)
    raise RuntimeError(f"aten::norm p={p_val} unsupported")


def _mT(g, self, **kwargs):
    """`aten::mT` — swap the last two axes.

    `Transpose.perm` is an attribute, so the rank is required; MHR only applies
    it to values whose shape the trace records.
    """
    rank = _rank(self)
    if rank is None or rank < 2:
        raise RuntimeError(f"aten::mT: rank unknown ({_sh()._get_tensor_sizes(self)})")
    perm = list(range(rank))
    perm[-1], perm[-2] = perm[-2], perm[-1]
    return g.op("Transpose", self, perm_i=perm)


def _bool_op(op_name):
    def fn(g, a, b, **kwargs):
        return g.op(op_name, _retensor(g, a), _retensor(g, b))

    return fn


def _aten_int(g, self, **kwargs):
    """`aten::Int` — a tensor-valued int, folded when the value is constant."""
    try:
        value = _sh()._get_const(self, "i", "self")
    except Exception:  # noqa: BLE001
        value = None
    if value is not None:
        return _const_int(g, int(value))
    return g.op("Cast", self, to_i=7)


def _cmp(op_name, negate=False):
    """Comparison ops whose operands carry no scalar type in MHR's graph.

    torch's `lt`/`gt`/... helpers call `_is_bool(other)`, which needs the
    operand's dtype. ONNX comparisons do not, so the node is emitted directly;
    constant operands are re-materialised as typed tensors because a constant
    whose `value` attribute is not a TensorProto breaks shape inference
    downstream.
    """

    def fn(g, self, other, **kwargs):
        node = g.op(op_name, _retensor(g, self), _retensor(g, other))
        return g.op("Not", node) if negate else node

    return fn


def _retensor(g, value):
    """Re-emit a constant `Value` as a tensor Constant, so it has a dtype."""
    if not isinstance(value, torch._C.Value):
        return value  # plain python scalar: `g.op` wraps it correctly
    node = value.node()
    if node.kind() != "onnx::Constant" or not node.hasAttribute("value"):
        return value
    if node.attributeNames().count("value") == 0:
        return value
    try:
        t = node.t_("value")
    except Exception:  # noqa: BLE001
        return value
    if t.dim() == 0 and t.scalar_type() == torch.int64:
        return _const_int(g, int(t))
    return g.op("Constant", value_t=t)


def _flatten(g, input, start_dim=0, end_dim=-1, **kwargs):
    """`aten::flatten` without needing the input rank.

    torch's lowering asks `_get_tensor_rank(input)`, which is unknown for MHR's
    intermediate values. Copying the untouched prefix and suffix of `Shape(x)`
    around a single `-1` has the same effect and needs no rank at all, except to
    resolve negative `start_dim` / `end_dim` — and that is done with `Size` on
    the shape tensor, at run time.
    """
    shape = g.op("Shape", input)
    rank = g.op("Size", shape)

    def resolve(value):
        # `start_dim` / `end_dim` arrive either as python ints or as traced
        # constant Values; both have to be turned into an int64 scalar.
        raw = value
        if not isinstance(value, int):
            try:
                raw = _sh()._get_const(value, "i", "dim")
            except Exception:  # noqa: BLE001
                raw = None
            if raw is None:
                raise RuntimeError(f"aten::flatten: non-constant dim {value}")
        scalar = _const_int(g, int(raw))
        negated = g.op("Less", scalar, _const_int(g, 0))
        return g.op("Where", negated, g.op("Add", scalar, rank), scalar)

    start = resolve(start_dim)
    end = resolve(end_dim)

    def one(value):
        return g.op("Unsqueeze", value, _const_ints(g, [0]))

    head = g.op("Slice", shape, one(_const_int(g, 0)), one(start), _const_ints(g, [0]))
    tail = g.op(
        "Slice",
        shape,
        one(g.op("Add", end, _const_int(g, 1))),
        one(_const_int(g, 2**62)),
        _const_ints(g, [0]),
    )
    pattern = g.op("Concat", head, _const_ints(g, [-1]), tail, axis_i=0)
    return g.op("Reshape", input, pattern)


def _sparse_coo_tensor(g, indices, values, size=None, *rest, **kwargs):
    """`torch.sparse_coo_tensor` → a dense zero matrix with the values scattered in.

    MHR's pose-corrective predictor stores its basis as a sparse COO matrix and
    immediately multiplies by a dense token, so the sparse form never has to
    survive into the graph: scattering the values with additive reduction gives
    the same product exactly. The zero matrix is built by `ConstantOfShape` so
    the dense basis is not embedded as a literal in the model file.
    """
    shape = _parse_ints(size)
    if shape is None:
        raise RuntimeError("aten::sparse_coo_tensor: non-constant size")
    if len(shape) != 2:
        raise RuntimeError(f"aten::sparse_coo_tensor: rank {len(shape)} unsupported")

    idx = g.op("Transpose", indices, perm_i=[1, 0])
    zero = g.op(
        "ConstantOfShape",
        _const_ints(g, shape),
        value_t=torch.tensor([0.0], dtype=torch.float32),
    )
    # The index depth equals the matrix rank, so each entry replaces one scalar
    # and `updates` stays 1-D: (nnz,) == indices.shape[:-1].
    return g.op("ScatterND", zero, idx, values, reduction_s="add")


def _rank_of(value):
    sh = _sh()
    sizes = sh._get_tensor_sizes(value)
    if sizes is not None:
        return len(sizes)
    try:
        return sh._get_tensor_rank(value)
    except Exception:  # noqa: BLE001
        return None


def _index_add(g, self, dim, index, other, alpha=None, **kwargs):
    """`aten::index_add` → `ScatterElements` with additive reduction.

    This is the same operation, not an approximation: every element of `other`
    is added into `self` at the position given by the matching `index` entry,
    accumulating on repeats.

    Everything is derived from `Shape()` at run time. torch's own lowering needs
    `self.dim()` and `other.dim()`, which are not carried by MHR's scripted
    values — its fallback expands `other` (length 51337 along the scatter axis)
    into `self` (18439) and produces a shape the runtime rejects outright.
    """
    sh = _sh()
    if alpha is not None:
        a = sh._maybe_get_scalar(alpha)
        if a is not None and float(sh._scalar(a)) != 1.0:
            raise RuntimeError("aten::index_add: alpha != 1 unsupported")

    axis = int(sh._get_const(dim, "i", "dim"))
    if axis < 0:
        rank = _rank_of(self)
        if rank is None:
            raise RuntimeError("aten::index_add: negative dim without known rank")
        axis += rank

    zero = _const_int(g, 0)
    one = _const_int(g, 1)

    index = g.op("Cast", index, to_i=7)  # int64
    other_shape = g.op("Shape", other)
    rank = g.op("Size", other_shape)
    positions = g.op("Range", zero, rank, one)
    # [1, …, K, …] with K on the scatter axis: reshape `index` into the shape a
    # single "row" of `other` would have, then broadcast it across the rest.
    is_axis = g.op("Equal", positions, _const_int(g, axis))
    pattern = g.op("Where", is_axis, g.op("Size", index), one)
    idx = g.op("Reshape", index, pattern)
    idx = g.op("Expand", idx, other_shape)

    return g.op("ScatterElements", self, idx, other, axis_i=axis, reduction_s="add")


def _parse_ints(value):
    """A constant int list from a symbolic argument, or None."""
    return _int_list(value)


_SYMBOLICS = {
    "aten::dim": _dim,
    "aten::size": _size,
    "aten::split_with_sizes": _split_with_sizes,
    "aten::cross": _cross,
    "aten::hypot": _hypot,
    "aten::norm": _norm,
    "aten::lt": _cmp("Less"),
    "aten::gt": _cmp("Greater"),
    "aten::le": _cmp("LessOrEqual"),
    "aten::ge": _cmp("GreaterOrEqual"),
    "aten::eq": _cmp("Equal"),
    "aten::ne": _cmp("Equal", negate=True),
    "aten::flatten": _flatten,
    "aten::mT": _mT,
    "aten::Int": _aten_int,
    "aten::__and__": _bool_op("And"),
    "aten::__or__": _bool_op("Or"),
    "aten::sparse_coo_tensor": _sparse_coo_tensor,
    "aten::sparse_coo_tensor.size": _sparse_coo_tensor,
    "aten::sparse_coo_tensor.indices_size": _sparse_coo_tensor,
    "aten::index_add": _index_add,
    "aten::index_add_": _index_add,
}

_OVERLOADS = {
    "aten::size.int": _size_dim,
    "aten::norm.ScalarOpt_dim": _norm,
    "aten::norm.Scalar": _norm,
}


def register(opset: int = OPSET) -> None:
    """Install the symbolics. Safe to call more than once.

    Two tables exist in this torch: the legacy `register_custom_op_symbolic`
    one and the newer `torchscript_exporter.registration` one. Which of them the
    exporter consults depends on the path it takes, so both are filled; a name
    only counts as failed when neither accepted it.
    """
    from torch.onnx._internal.torchscript_exporter import registration

    def install(name, fn):
        ok = False
        try:
            torch.onnx.register_custom_op_symbolic(name, fn, opset)
            ok = True
        except Exception:  # noqa: BLE001
            pass
        try:
            registration.custom_onnx_symbolic(name, opset)(fn)
            ok = True
        except Exception:  # noqa: BLE001
            pass
        if not ok:
            print(f"  (symbolic {name}: not registered)", flush=True)

    for name, fn in _SYMBOLICS.items():
        install(name, fn)
    for name, fn in _OVERLOADS.items():
        install(name, fn)
