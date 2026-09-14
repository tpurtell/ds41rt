"""Immutable pre-route dSpark anchor selection adapted from ds4rt's policy.

Coordinates use V4.1 token IDs: visible history ends at position p >= 1 and
the known teacher-forced token is p+1. Every selected anchor keeps five drafts.
All anchors belonging to a source record must be issued jointly by the adapter.
"""
import hashlib
import struct


def splitmix64(value):
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def select_anchors(records, *, count=327680, seed=20260809):
    if type(count) is not int or type(seed) is not int or not records:
        raise ValueError("anchor selection requires records and integer count/seed")
    lengths = [len(record["input_ids"]) for record in records]
    eligible = [max(length - 2, 0) for length in lengths]
    total = sum(eligible)
    if not 0 < count <= total:
        raise ValueError("anchor count must fit the eligible V4.1 corpus; no automatic resizing")
    groups = [[] for _ in records]
    record_index, start = 0, 0
    digest = hashlib.sha256()
    for stratum in range(count):
        lower, upper = stratum * total // count, (stratum + 1) * total // count
        ordinal = lower + splitmix64((seed & ((1 << 64) - 1)) ^ stratum) % (upper - lower)
        while ordinal >= start + eligible[record_index]:
            start += eligible[record_index]
            record_index += 1
        position = ordinal - start + 1
        groups[record_index].append(position)
        digest.update(struct.pack("<QQ", record_index, position))
    return dict(schema="ds41rt-dspark-stratified-anchors-v1", seed=seed, count=count,
                eligible=total, positions=tuple(tuple(group) for group in groups),
                coordinate_sha256=digest.hexdigest(), proposal_rows=count * 5,
                batching="all-selected-anchors-per-original-record", known_token="corpus-position-plus-one")
