"""Program model + .vsr JSON serialization and validation.

A chain is a directed graph: vertex id -> list of outgoing edges. Entry
vertex of every chain is vertex 0. Branch vertices (2+ outgoing edges) must
carry a guard on every edge.

Load-time lint: every segment is decoded under the identity frame and a
warning (not a fault) is emitted for dead-zone or zero-length segments.
This cannot catch frame-dependent ambiguity; the runtime check remains.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .decode import Cubic26
from .errors import LoadError, VersorFault

VERSION = "0.1"


@dataclass
class Edge:
    seg: np.ndarray          # raw R^3 segment vector
    to: int                  # target vertex id
    guard: np.ndarray | None = None  # frame-local unit guard normal


@dataclass
class Chain:
    id: int
    vertices: dict[int, list[Edge]]  # vertex id -> outgoing edges
    comment: str = ""
    entry: int = 0


@dataclass
class Program:
    chains: list[Chain]  # index == chain id
    name: str = ""
    version: str = VERSION
    warnings: list[str] = field(default_factory=list)


def _vec3(x, what: str) -> np.ndarray:
    if not (isinstance(x, (list, tuple)) and len(x) == 3):
        raise LoadError(f"{what}: expected [x, y, z], got {x!r}")
    try:
        return np.array([float(c) for c in x])
    except (TypeError, ValueError):
        raise LoadError(f"{what}: non-numeric component in {x!r}")


def from_dict(data: dict) -> Program:
    if not isinstance(data, dict):
        raise LoadError("program must be a JSON object")
    version = str(data.get("version", VERSION))
    name = str(data.get("name", ""))
    raw_chains = data.get("chains")
    if not isinstance(raw_chains, list) or not raw_chains:
        raise LoadError("program needs a non-empty 'chains' list")

    warnings: list[str] = []
    by_id: dict[int, Chain] = {}
    for rc in raw_chains:
        cid = rc.get("id")
        if not isinstance(cid, int) or cid < 0:
            raise LoadError(f"chain id must be a non-negative int, got {cid!r}")
        if cid in by_id:
            raise LoadError(f"duplicate chain id {cid}")
        vertices: dict[int, list[Edge]] = {}
        for rv in rc.get("vertices", []):
            vid = rv.get("id")
            if not isinstance(vid, int):
                raise LoadError(f"chain {cid}: vertex id must be int, got {vid!r}")
            if vid in vertices:
                raise LoadError(f"chain {cid}: duplicate vertex id {vid}")
            edges = []
            for re_ in rv.get("out", []):
                seg = _vec3(re_.get("seg"), f"chain {cid} vertex {vid} seg")
                to = re_.get("to")
                if not isinstance(to, int):
                    raise LoadError(f"chain {cid} vertex {vid}: edge 'to' must be int")
                guard = None
                if "guard" in re_:
                    guard = _vec3(re_["guard"], f"chain {cid} vertex {vid} guard")
                    gn = float(np.linalg.norm(guard))
                    if gn < 1e-9:
                        raise LoadError(f"chain {cid} vertex {vid}: zero guard vector")
                    if abs(gn - 1.0) > 1e-6:
                        warnings.append(
                            f"chain {cid} vertex {vid}: guard not unit norm "
                            f"(|g| = {gn:.6g}), normalizing")
                        guard = guard / gn
                edges.append(Edge(seg=seg, to=to, guard=guard))
            vertices[vid] = edges
        if 0 not in vertices:
            raise LoadError(f"chain {cid}: missing entry vertex 0")
        by_id[cid] = Chain(id=cid, vertices=vertices, comment=str(rc.get("comment", "")))

    ids = sorted(by_id)
    if ids != list(range(len(ids))):
        raise LoadError(f"chain ids must be contiguous from 0, got {ids}")
    chains = [by_id[i] for i in ids]

    # structural validation
    for ch in chains:
        for vid, edges in ch.vertices.items():
            for e in edges:
                if e.to not in ch.vertices:
                    raise LoadError(
                        f"chain {ch.id} vertex {vid}: edge target {e.to} does not exist")
            if len(edges) >= 2:
                missing = [i for i, e in enumerate(edges) if e.guard is None]
                if missing:
                    raise LoadError(
                        f"chain {ch.id} vertex {vid}: branch vertex has "
                        f"{len(edges)} edges but edges {missing} lack guards")
            elif len(edges) == 1 and edges[0].guard is not None:
                warnings.append(
                    f"chain {ch.id} vertex {vid}: guard on single-edge vertex "
                    f"(ignored at runtime)")

    prog = Program(chains=chains, name=name, version=version, warnings=warnings)
    warnings.extend(lint(prog))
    return prog


def lint(prog: Program) -> list[str]:
    """Decode every segment under the identity frame; warn on dead zones."""
    dec = Cubic26()
    warnings = []
    for ch in prog.chains:
        for vid, edges in ch.vertices.items():
            for i, e in enumerate(edges):
                n = float(np.linalg.norm(e.seg))
                where = f"chain {ch.id} vertex {vid} edge {i}"
                if n < 1e-6:
                    warnings.append(f"{where}: zero-length segment (will fault)")
                    continue
                try:
                    dec.decode(e.seg / n)
                except VersorFault as f:
                    warnings.append(
                        f"{where}: dead zone under identity frame ({f.args[0]})")
    return warnings


def to_dict(prog: Program) -> dict:
    return {
        "version": prog.version,
        "name": prog.name,
        "chains": [
            {
                "id": ch.id,
                **({"comment": ch.comment} if ch.comment else {}),
                "vertices": [
                    {
                        "id": vid,
                        "out": [
                            {
                                "seg": [float(c) for c in e.seg],
                                "to": e.to,
                                **({"guard": [float(c) for c in e.guard]}
                                   if e.guard is not None else {}),
                            }
                            for e in edges
                        ],
                    }
                    for vid, edges in sorted(ch.vertices.items())
                ],
            }
            for ch in prog.chains
        ],
    }


def load(path: str) -> Program:
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise LoadError(f"{path}: invalid JSON: {e}")
    return from_dict(data)


def save(prog: Program, path: str) -> None:
    with open(path, "w") as f:
        json.dump(to_dict(prog), f, indent=2)
        f.write("\n")
