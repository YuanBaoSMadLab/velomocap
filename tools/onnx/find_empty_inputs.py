"""Report nodes whose inputs are empty strings (i.e. unloadable graphs).

`torch.onnx.export` can emit a node with a missing input, and ONNX Runtime only
says which node it was. This walks the graph instead and prints where the empty
input should have come from, plus the nodes around it, so the failing lowering
can be found from the operator names alone.

    python tools/onnx/find_empty_inputs.py export/sam3d/sam3d_post.onnx
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--context", type=int, default=3)
    args = parser.parse_args()

    import onnx

    failed = 0
    for path in args.models:
        # `load_external_data=False` keeps the 693 MB sidecars off the heap; the
        # topology is entirely in the graph protobuf.
        model = onnx.load(path, load_external_data=False)
        graph = model.graph

        producer = {}
        for node in graph.node:
            for index, output in enumerate(node.output):
                if output:
                    producer[output] = (node, index)

        empties = [
            (node, index)
            for node in graph.node
            for index, name in enumerate(node.input)
            if name == ""
        ]
        if not empties:
            print(f"{os.path.basename(path):24s} clean")
            continue
        failed += 1
        print(f"{os.path.basename(path):24s} {len(empties)} empty input(s)")
        for node, index in empties:
            print(f"  node {node.name!r} {node.op_type} input {index} of {len(node.input)}")
            print(f"    inputs  {list(node.input)}")
            print(f"    outputs {list(node.output)}")
            if node.op_type in {"Add", "Mul", "Sub", "Div", "Concat", "Expand"}:
                for other in node.input:
                    if not other:
                        continue
                    source = producer.get(other)
                    if source is not None:
                        print(
                            f"    sibling {other!r} <- {source[0].op_type} "
                            f"{source[0].name!r} {list(source[0].input)}"
                        )
            # Show what consumes the broken node, to name the construct.
            for consumer in graph.node:
                if node.output and node.output[0] in consumer.input:
                    print(f"    consumed by {consumer.op_type} {consumer.name!r}")
        print("  --- named nodes nearest the break ---")
        names = [n.name for n in graph.node if not n.name.startswith(("Constant", "/Constant"))]
        for name in names[: args.context]:
            print(f"    {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
