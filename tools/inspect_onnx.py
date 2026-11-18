\
#!/usr/bin/env python3
"""Inspect ONNX model structure and emit JSON + Markdown reports.

This tool does not execute inference and does not prove Hailo-8 compatibility.
It deliberately reports facts from the ONNX graph without guessing model semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import onnx
    from onnx import TensorProto
except ImportError as exc:  # pragma: no cover - exercised in container
    raise SystemExit(
        "Missing dependency 'onnx'. Run this tool inside the project Docker image "
        "or install requirements.txt."
    ) from exc


MODEL_SUFFIXES = {".onnx"}
NMS_OPS = {"NonMaxSuppression"}
KNOWN_DYNAMIC_MARKERS = {"", "unk__0", "None"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_shape(value_info: Any) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    dims: list[Any] = []
    dynamic = False
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append({"symbol": dim.dim_param})
            dynamic = True
        else:
            dims.append(None)
            dynamic = True
    return {
        "name": value_info.name,
        "element_type": TensorProto.DataType.Name(tensor_type.elem_type)
        if tensor_type.elem_type else "UNDEFINED",
        "shape": dims,
        "dynamic": dynamic,
    }


def inspect_model(path: Path) -> dict[str, Any]:
    stat = path.stat()
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(str(path))

    graph = model.graph
    initializer_names = {item.name for item in graph.initializer}
    inputs = [
        tensor_shape(item)
        for item in graph.input
        if item.name not in initializer_names
    ]
    outputs = [tensor_shape(item) for item in graph.output]

    op_counter = Counter(node.op_type for node in graph.node)
    domains = sorted({node.domain or "ai.onnx" for node in graph.node})
    custom_domains = [domain for domain in domains if domain not in {"ai.onnx", "ai.onnx.ml"}]
    external_tensors = []
    for tensor in graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
            external_tensors.append({
                "name": tensor.name,
                "external_data": [
                    {"key": entry.key, "value": entry.value}
                    for entry in tensor.external_data
                ],
            })

    dynamic_inputs = [item["name"] for item in inputs if item["dynamic"]]
    dynamic_outputs = [item["name"] for item in outputs if item["dynamic"]]
    nms_nodes = [
        {
            "name": node.name,
            "domain": node.domain or "ai.onnx",
            "inputs": list(node.input),
            "outputs": list(node.output),
        }
        for node in graph.node
        if node.op_type in NMS_OPS
    ]

    opset_imports = [
        {"domain": item.domain or "ai.onnx", "version": int(item.version)}
        for item in model.opset_import
    ]

    return {
        "file": path.name,
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
        "onnx_valid": True,
        "ir_version": int(model.ir_version),
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "model_domain": model.domain,
        "model_version": int(model.model_version),
        "opset_imports": opset_imports,
        "graph_name": graph.name,
        "node_count": len(graph.node),
        "initializer_count": len(graph.initializer),
        "input_count": len(inputs),
        "output_count": len(outputs),
        "inputs": inputs,
        "outputs": outputs,
        "operator_counts": dict(sorted(op_counter.items())),
        "domains": domains,
        "custom_domains": custom_domains,
        "contains_nms": bool(nms_nodes),
        "nms_nodes": nms_nodes,
        "dynamic_input_names": dynamic_inputs,
        "dynamic_output_names": dynamic_outputs,
        "external_tensor_count": len(external_tensors),
        "external_tensors": external_tensors,
        "warnings": build_warnings(
            inputs, outputs, nms_nodes, custom_domains, external_tensors
        ),
    }


def build_warnings(
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    nms_nodes: list[dict[str, Any]],
    custom_domains: list[str],
    external_tensors: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    if not inputs:
        warnings.append("No runtime graph inputs were found.")
    if not outputs:
        warnings.append("No graph outputs were found.")
    if any(item["dynamic"] for item in inputs + outputs):
        warnings.append(
            "Dynamic input/output dimensions exist; confirm whether the target "
            "Hailo parser supports the graph and choose a fixed deployment shape."
        )
    if nms_nodes:
        warnings.append(
            "NonMaxSuppression operator exists in the graph. Check DFC support; "
            "consider exporting raw predictions and performing NMS on the host."
        )
    if custom_domains:
        warnings.append(
            "Custom operator domains exist. The Hailo parser may require custom "
            "conversion or an alternative export."
        )
    if external_tensors:
        warnings.append(
            "External tensor data is referenced. Keep the companion data files "
            "together with the ONNX file."
        )
    if len(inputs) > 1:
        warnings.append(
            "Multiple graph inputs exist; verify whether all are expected for inference."
        )
    return warnings


def markdown_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# ONNX Model Inspection Report",
        "",
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        "- Tool: `tools/inspect_onnx.py`",
        "- Scope: static ONNX graph inspection; no inference or Hailo compilation performed.",
        "",
    ]
    for result in results:
        lines.extend([
            f"## {result['file']}",
            "",
            f"- Valid ONNX: **{result['onnx_valid']}**",
            f"- Size: {result['size_bytes']:,} bytes",
            f"- SHA-256: `{result['sha256']}`",
            f"- Producer: `{result['producer_name']}` `{result['producer_version']}`",
            f"- IR version: `{result['ir_version']}`",
            f"- Opsets: `{json.dumps(result['opset_imports'], ensure_ascii=False)}`",
            f"- Graph nodes: {result['node_count']}",
            f"- Initializers: {result['initializer_count']}",
            f"- NMS operator present: **{result['contains_nms']}**",
            f"- Custom domains: `{result['custom_domains']}`",
            "",
            "### Inputs",
            "",
            "| Name | Type | Shape | Dynamic |",
            "|---|---|---|---|",
        ])
        for item in result["inputs"]:
            lines.append(
                f"| `{item['name']}` | {item['element_type']} | "
                f"`{json.dumps(item['shape'])}` | {item['dynamic']} |"
            )
        lines.extend([
            "",
            "### Outputs",
            "",
            "| Name | Type | Shape | Dynamic |",
            "|---|---|---|---|",
        ])
        for item in result["outputs"]:
            lines.append(
                f"| `{item['name']}` | {item['element_type']} | "
                f"`{json.dumps(item['shape'])}` | {item['dynamic']} |"
            )
        lines.extend(["", "### Operator counts", "", "```json",
                      json.dumps(result["operator_counts"], indent=2),
                      "```", "", "### Warnings", ""])
        if result["warnings"]:
            lines.extend(f"- {warning}" for warning in result["warnings"])
        else:
            lines.append("- No structural warning detected by this tool.")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("models/source"))
    parser.add_argument("--report-dir", type=Path, default=Path("models/reports"))
    parser.add_argument("--model", type=Path, action="append", default=[],
                        help="Inspect a specific ONNX file; can be supplied multiple times.")
    args = parser.parse_args()

    candidates = args.model or sorted(
        path for path in args.models_dir.glob("*") if path.suffix.lower() in MODEL_SUFFIXES
    )
    if not candidates:
        print(f"No .onnx files found in {args.models_dir}", file=sys.stderr)
        return 2

    args.report_dir.mkdir(parents=True, exist_ok=True)
    results = []
    failed = False
    for path in candidates:
        try:
            result = inspect_model(path)
            results.append(result)
            print(f"[OK] {path}: ONNX checker passed")
        except Exception as exc:
            failed = True
            results.append({
                "file": path.name,
                "path": str(path),
                "onnx_valid": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[FAIL] {path}: {type(exc).__name__}: {exc}", file=sys.stderr)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = args.report_dir / f"onnx-inspection-{stamp}.json"
    md_path = args.report_dir / f"onnx-inspection-{stamp}.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(markdown_report([r for r in results if r.get("onnx_valid")]), encoding="utf-8")

    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
