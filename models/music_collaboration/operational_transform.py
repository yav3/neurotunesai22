"""Operational transformation for real-time collaborative music editing.

Implements OT for concurrent note-level edits:
- Insert, delete, and update operations on ordered note sequences
- Transform pairs so concurrent edits converge to the same document state
- Operation composition and inversion for undo
- Server-side transform pipeline for multi-client convergence
- Convergence verification (TP1 property)
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class OpType(Enum):
    INSERT = "insert"
    DELETE = "delete"
    UPDATE = "update"
    NOOP = "noop"


@dataclass
class NotePayload:
    """Payload describing a note event."""
    pitch: int = 60        # MIDI
    velocity: int = 80
    start_beat: float = 0.0
    duration: float = 1.0
    voice: int = 0

    def to_tuple(self) -> Tuple:
        return (self.pitch, self.velocity, self.start_beat, self.duration, self.voice)


@dataclass
class Operation:
    """A single OT operation on a note sequence.

    - INSERT: insert `payload` at `position`
    - DELETE: remove note at `position`
    - UPDATE: replace note at `position` with `payload`
    - NOOP: identity (produced by transform when ops cancel)
    """
    op_type: OpType
    position: int = 0
    payload: Optional[NotePayload] = None
    client_id: str = ""
    revision: int = 0

    def is_noop(self) -> bool:
        return self.op_type == OpType.NOOP


def transform(op_a: Operation, op_b: Operation) -> Tuple[Operation, Operation]:
    """Transform pair (op_a, op_b) → (op_a', op_b') satisfying TP1.

    After transform, applying op_a then op_b' yields the same state as
    applying op_b then op_a'.
    """
    a = copy.deepcopy(op_a)
    b = copy.deepcopy(op_b)

    if a.is_noop() or b.is_noop():
        return a, b

    # INSERT vs INSERT
    if a.op_type == OpType.INSERT and b.op_type == OpType.INSERT:
        if a.position < b.position:
            b.position += 1
        elif a.position > b.position:
            a.position += 1
        else:
            # tie-break: lower client_id wins position
            if a.client_id <= b.client_id:
                b.position += 1
            else:
                a.position += 1
        return a, b

    # INSERT vs DELETE
    if a.op_type == OpType.INSERT and b.op_type == OpType.DELETE:
        if a.position <= b.position:
            b.position += 1
        else:
            a.position -= 1
        return a, b
    if a.op_type == OpType.DELETE and b.op_type == OpType.INSERT:
        if b.position <= a.position:
            a.position += 1
        else:
            b.position -= 1
        return a, b

    # DELETE vs DELETE
    if a.op_type == OpType.DELETE and b.op_type == OpType.DELETE:
        if a.position < b.position:
            b.position -= 1
        elif a.position > b.position:
            a.position -= 1
        else:
            # both delete same position → both become noop
            a = Operation(OpType.NOOP, client_id=a.client_id, revision=a.revision)
            b = Operation(OpType.NOOP, client_id=b.client_id, revision=b.revision)
        return a, b

    # INSERT vs UPDATE
    if a.op_type == OpType.INSERT and b.op_type == OpType.UPDATE:
        if a.position <= b.position:
            b.position += 1
        return a, b
    if a.op_type == OpType.UPDATE and b.op_type == OpType.INSERT:
        if b.position <= a.position:
            a.position += 1
        return a, b

    # DELETE vs UPDATE
    if a.op_type == OpType.DELETE and b.op_type == OpType.UPDATE:
        if a.position < b.position:
            b.position -= 1
        elif a.position == b.position:
            # delete wins over update
            b = Operation(OpType.NOOP, client_id=b.client_id, revision=b.revision)
        return a, b
    if a.op_type == OpType.UPDATE and b.op_type == OpType.DELETE:
        if b.position < a.position:
            a.position -= 1
        elif b.position == a.position:
            a = Operation(OpType.NOOP, client_id=a.client_id, revision=a.revision)
        return a, b

    # UPDATE vs UPDATE
    if a.op_type == OpType.UPDATE and b.op_type == OpType.UPDATE:
        if a.position == b.position:
            # tie-break: lower client_id wins
            if a.client_id <= b.client_id:
                b = Operation(OpType.NOOP, client_id=b.client_id, revision=b.revision)
            else:
                a = Operation(OpType.NOOP, client_id=a.client_id, revision=a.revision)
        return a, b

    return a, b


def apply_op(doc: List[NotePayload], op: Operation) -> List[NotePayload]:
    """Apply a single operation to a note-sequence document."""
    if op.is_noop():
        return doc
    result = list(doc)
    if op.op_type == OpType.INSERT:
        pos = max(0, min(op.position, len(result)))
        if op.payload is not None:
            result.insert(pos, copy.deepcopy(op.payload))
    elif op.op_type == OpType.DELETE:
        if 0 <= op.position < len(result):
            result.pop(op.position)
    elif op.op_type == OpType.UPDATE:
        if 0 <= op.position < len(result) and op.payload is not None:
            result[op.position] = copy.deepcopy(op.payload)
    return result


def apply_ops(doc: List[NotePayload], ops: List[Operation]) -> List[NotePayload]:
    """Apply a sequence of operations in order."""
    for op in ops:
        doc = apply_op(doc, op)
    return doc


def compose(op_a: Operation, op_b: Operation) -> List[Operation]:
    """Compose two sequential operations into a minimal equivalent list.

    Not always reducible to a single op, so returns a list.
    """
    if op_a.is_noop():
        return [op_b]
    if op_b.is_noop():
        return [op_a]
    # INSERT then DELETE at same position → noop
    if (op_a.op_type == OpType.INSERT and op_b.op_type == OpType.DELETE
            and op_a.position == op_b.position):
        return [Operation(OpType.NOOP)]
    # INSERT then UPDATE at same position → INSERT with updated payload
    if (op_a.op_type == OpType.INSERT and op_b.op_type == OpType.UPDATE
            and op_a.position == op_b.position):
        merged = copy.deepcopy(op_a)
        merged.payload = copy.deepcopy(op_b.payload)
        return [merged]
    return [op_a, op_b]


def invert(op: Operation, doc: List[NotePayload]) -> Operation:
    """Compute the inverse operation (for undo).

    Requires the document state *before* op was applied.
    """
    if op.is_noop():
        return Operation(OpType.NOOP, client_id=op.client_id, revision=op.revision)
    if op.op_type == OpType.INSERT:
        return Operation(OpType.DELETE, position=op.position,
                         client_id=op.client_id, revision=op.revision)
    if op.op_type == OpType.DELETE:
        if 0 <= op.position < len(doc):
            old_note = copy.deepcopy(doc[op.position])
        else:
            old_note = NotePayload()
        return Operation(OpType.INSERT, position=op.position, payload=old_note,
                         client_id=op.client_id, revision=op.revision)
    if op.op_type == OpType.UPDATE:
        if 0 <= op.position < len(doc):
            old_note = copy.deepcopy(doc[op.position])
        else:
            old_note = NotePayload()
        return Operation(OpType.UPDATE, position=op.position, payload=old_note,
                         client_id=op.client_id, revision=op.revision)
    return Operation(OpType.NOOP)


@dataclass
class OTServer:
    """Server that maintains a canonical document and transform log.

    Clients send operations against their known revision; the server
    transforms against any operations that happened since that revision,
    applies the result, and broadcasts the transformed op.
    """
    document: List[NotePayload] = field(default_factory=list)
    history: List[Operation] = field(default_factory=list)
    revision: int = 0

    def receive(self, op: Operation) -> Operation:
        """Transform op against all operations since op.revision, apply, and return."""
        transformed = copy.deepcopy(op)
        for past_op in self.history[transformed.revision:]:
            transformed, _ = transform(transformed, past_op)
        self.document = apply_op(self.document, transformed)
        transformed.revision = self.revision
        self.history.append(transformed)
        self.revision += 1
        return transformed

    def snapshot_hash(self) -> str:
        blob = str([n.to_tuple() for n in self.document])
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def verify_tp1(op_a: Operation, op_b: Operation,
               doc: List[NotePayload]) -> bool:
    """Verify TP1 convergence property for a pair of concurrent operations.

    TP1: apply(apply(doc, a), b') == apply(apply(doc, b), a')
    where a', b' = transform(a, b).
    """
    a_prime, b_prime = transform(op_a, op_b)
    path_ab = apply_op(apply_op(doc, op_a), b_prime)
    path_ba = apply_op(apply_op(doc, op_b), a_prime)
    if len(path_ab) != len(path_ba):
        return False
    for na, nb in zip(path_ab, path_ba):
        if na.to_tuple() != nb.to_tuple():
            return False
    return True


def random_ops(n: int, doc_len: int, client_id: str = "test",
               rng: Optional[np.random.RandomState] = None) -> List[Operation]:
    """Generate n random operations for testing/fuzzing OT correctness."""
    if rng is None:
        rng = np.random.RandomState(42)
    ops: List[Operation] = []
    current_len = doc_len
    for _ in range(n):
        kind = rng.choice(["insert", "delete", "update"])
        if kind == "insert" or current_len == 0:
            pos = rng.randint(0, max(1, current_len + 1))
            payload = NotePayload(
                pitch=int(rng.randint(48, 84)),
                velocity=int(rng.randint(40, 120)),
                start_beat=float(rng.uniform(0, 32)),
                duration=float(rng.choice([0.25, 0.5, 1.0, 2.0])),
                voice=int(rng.randint(0, 4)),
            )
            ops.append(Operation(OpType.INSERT, position=pos, payload=payload,
                                 client_id=client_id))
            current_len += 1
        elif kind == "delete" and current_len > 0:
            pos = rng.randint(0, current_len)
            ops.append(Operation(OpType.DELETE, position=pos, client_id=client_id))
            current_len -= 1
        elif kind == "update" and current_len > 0:
            pos = rng.randint(0, current_len)
            payload = NotePayload(
                pitch=int(rng.randint(48, 84)),
                velocity=int(rng.randint(40, 120)),
            )
            ops.append(Operation(OpType.UPDATE, position=pos, payload=payload,
                                 client_id=client_id))
    return ops


def convergence_test(doc: List[NotePayload], ops_a: List[Operation],
                     ops_b: List[Operation]) -> Tuple[bool, str]:
    """Run two sequences of operations through two independent OT servers
    and check that both arrive at the same final document state."""
    server1 = OTServer(document=copy.deepcopy(doc))
    server2 = OTServer(document=copy.deepcopy(doc))
    for op in ops_a:
        op.revision = server1.revision
        server1.receive(op)
    for op in ops_b:
        op.revision = server2.revision
        server2.receive(op)
    # Now apply B's ops to server1 and A's ops to server2
    for op in ops_b:
        op.revision = 0
        server1.receive(op)
    for op in ops_a:
        op.revision = 0
        server2.receive(op)
    h1 = server1.snapshot_hash()
    h2 = server2.snapshot_hash()
    if h1 == h2:
        return True, h1
    return False, f"diverged: {h1} vs {h2}"
