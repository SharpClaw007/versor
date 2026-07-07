"""Orientation-specialized chain cloning — the compiler transform that makes
frame-covariant calls possible (whitepaper §12.1).

A callee's raw vectors decode under the caller's *live* frame, so a chain
only behaves as authored when entered with the frame it was authored for
(identity, for builder/assembler output). Called from a rotated site, its
segments silently reinterpret into different opcodes. `specialize` fixes
this the way a compiler would: for every reachable (chain, entry-frame)
pair it emits a clone whose segments are conjugated into that entry frame
(v ↦ F·v·F⁻¹), and rewrites its CALLs to point at the right clones.

The algebra: a clone built for entry frame F, executing under live frame
F·G (G = the relative frame accumulated since entry), decodes each rewritten
segment as (F·G)⁻¹(F·v·F⁻¹)(F·G) = G⁻¹·v·G — exactly the original decode
under identity entry. Entry frames compose down the call tree: a call site
with relative frame G inside a clone for F requires the callee's clone for
F·G. The clone set is finite exactly when the program's rotation angles
generate a finite group; a budget guards the irrational case.

What is preserved and what changes: **guards are not rewritten** — they are
stored frame-local and rotated by the live frame at runtime, so
orientation-as-argument (branching on the caller's frame, as in add_two)
survives specialization untouched. Only the *accidental* frame-dependence —
opcode reinterpretation of raw segments — is removed. Programs whose
behavior relies solely on guards are behavior-preserved; naively-authored
recursive geometry (Koch, Lévy) becomes runnable as intended.

Restrictions (checked, with clear errors): the frame at every vertex must
be path-independent under identity entry — no POPF, no frame disagreement
at merge points, no JMPZ/JMPP immediately before a rotation (a skippable
rotation makes the frame runtime-dependent).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from .decode import get_decoder
from .errors import LoadError
from .isa import OPCODES
from .loader import Chain, Edge, Program
from .quat import Quat

ROT_AXES = {"ROTF": (1.0, 0.0, 0.0), "ROTG": (0.0, 1.0, 0.0),
            "ROTH": (0.0, 0.0, 1.0)}


class SpecializeError(LoadError):
    pass


def _decode_op(decoder, F: Quat, seg: np.ndarray):
    n = float(np.linalg.norm(seg))
    if n < 1e-9:
        raise SpecializeError("zero-length segment during frame analysis")
    v_local = F.conj().rotate(seg)
    key = decoder.decode(v_local / n)
    return OPCODES[key].mnemonic, n


def _vertex_frames(chain: Chain, decoder) -> dict[int, Quat]:
    """Relative frame at each reachable vertex under identity entry."""
    frames: dict[int, Quat] = {chain.entry: Quat.identity()}
    skippers: set[int] = set()  # vertices entered via a JMPZ/JMPP edge
    queue = deque([chain.entry])
    while queue:
        vid = queue.popleft()
        F = frames[vid]
        for e in chain.vertices[vid]:
            mnemonic, n = _decode_op(decoder, F, e.seg)
            F_next = F
            if mnemonic in ROT_AXES:
                if vid in skippers:
                    raise SpecializeError(
                        f"chain {chain.id} vertex {vid}: a rotation may be "
                        "skipped by JMPZ/JMPP — frame is runtime-dependent")
                F_next = (F * Quat.axis_angle(ROT_AXES[mnemonic], n)).normalized()
            elif mnemonic == "POPF":
                raise SpecializeError(
                    f"chain {chain.id} vertex {vid}: POPF makes the frame "
                    "path-dependent; specialize cannot analyze it")
            if mnemonic in ("JMPZ", "JMPP"):
                skippers.add(e.to)
            if e.to in frames:
                if not frames[e.to].approx(F_next, tol=1e-9):
                    raise SpecializeError(
                        f"chain {chain.id} vertex {e.to}: converging paths "
                        "carry different frames — cannot specialize")
            else:
                frames[e.to] = F_next
                queue.append(e.to)
    return frames


def _call_sites(chain: Chain, decoder, frames):
    """{(vertex, edge-index): (raw chain id, scale frac)} for every CALL."""
    sites = {}
    for vid, edges in chain.vertices.items():
        if vid not in frames:
            continue  # unreachable vertex: nothing to analyze
        for i, e in enumerate(edges):
            mnemonic, n = _decode_op(decoder, frames[vid], e.seg)
            if mnemonic == "CALL":
                sites[(vid, i)] = (int(math.floor(n)), n - math.floor(n))
    return sites


class _FrameRegistry:
    """Interns quaternions up to sign so frames can be dict keys."""

    def __init__(self):
        self.frames: list[Quat] = []

    def intern(self, F: Quat) -> int:
        for i, G in enumerate(self.frames):
            if F.approx(G, tol=1e-7):
                return i
        self.frames.append(F)
        return len(self.frames) - 1


def specialize(prog: Program, max_clones: int = 512) -> Program:
    """Clone chains per reachable entry frame so every chain behaves as
    authored, whatever frame it is called under. The root keeps identity."""
    decoder = get_decoder(prog.decoder)
    n_chains = len(prog.chains)

    analysis = {ch.id: None for ch in prog.chains}
    for ch in prog.chains:
        frames = _vertex_frames(ch, decoder)
        analysis[ch.id] = (frames, _call_sites(ch, decoder, frames))

    registry = _FrameRegistry()
    clone_ids: dict[tuple[int, int], int] = {}   # (src chain, frame key) -> new id
    clone_src: list[tuple[int, int]] = []        # new id -> (src chain, frame key)

    def clone_for(cid: int, F: Quat) -> int:
        key = (cid, registry.intern(F))
        if key not in clone_ids:
            if len(clone_src) >= max_clones:
                raise SpecializeError(
                    f"clone budget of {max_clones} exceeded — the program's "
                    "rotation angles likely generate an infinite group")
            clone_ids[key] = len(clone_src)
            clone_src.append(key)
            work.append(key)
        return clone_ids[key]

    work: deque = deque()
    clone_for(0, Quat.identity())
    while work:
        cid, fkey = work.popleft()
        F = registry.frames[fkey]
        frames, sites = analysis[cid]
        for (vid, _i), (raw_cid, _frac) in sites.items():
            F_site = (F * frames[vid]).normalized()
            clone_for(raw_cid % n_chains, F_site)

    new_chains: list[Chain] = []
    for new_id, (cid, fkey) in enumerate(clone_src):
        F = registry.frames[fkey]
        src = prog.chains[cid]
        frames, sites = analysis[cid]
        vertices: dict[int, list[Edge]] = {}
        for vid, edges in src.vertices.items():
            out = []
            for i, e in enumerate(edges):
                seg = F.rotate(e.seg)
                if (vid, i) in sites:
                    raw_cid, frac = sites[(vid, i)]
                    F_site = (F * frames[vid]).normalized()
                    new_cid = clone_ids[(raw_cid % n_chains,
                                         registry.intern(F_site))]
                    n_new = new_cid + frac
                    if n_new < 1e-9:
                        raise SpecializeError(
                            "cannot encode a CALL to clone 0 with zero "
                            "fractional magnitude")
                    seg = seg / np.linalg.norm(seg) * n_new
                out.append(Edge(seg=seg, to=e.to,
                                guard=None if e.guard is None
                                else e.guard.copy()))
            vertices[vid] = out
        w, x, y, z = (round(c, 3) for c in F.as_tuple())
        new_chains.append(Chain(
            id=new_id, vertices=vertices,
            comment=f"{src.comment or f'chain {cid}'} [entry F=({w},{x},{y},{z})]"))

    return Program(chains=new_chains, name=(prog.name or "program") + "-specialized",
                   version=prog.version, decoder=prog.decoder)
