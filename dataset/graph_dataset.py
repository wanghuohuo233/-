"""Graph tensors for PyTorch GNN diffusion training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from dataset.material_dataset import MaterialRecord
from utils.geo_utils import build_adjacency, element_feature, graph_descriptor, evaluate_material


@dataclass
class GraphBatchArrays:
    descriptors: np.ndarray
    conditions: np.ndarray
    node_features: np.ndarray
    adjacency: np.ndarray
    mask: np.ndarray


def record_to_graph_arrays(record: MaterialRecord, max_atoms: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_features = np.zeros((max_atoms, 8), dtype=np.float32)
    mask = np.zeros(max_atoms, dtype=np.float32)
    positions = np.asarray(record.positions, dtype=float)
    elements = list(record.elements)

    for idx, element in enumerate(elements[:max_atoms]):
        xy = positions[idx, :2] / 6.0
        z = np.array([positions[idx, 2] / 12.0])
        node_features[idx] = np.concatenate([element_feature(element), xy, z]).astype(np.float32)
        mask[idx] = 1.0

    adjacency = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    small_adj = build_adjacency(record.positions)
    size = min(max_atoms, small_adj.shape[0])
    adjacency[:size, :size] = small_adj[:size, :size].astype(np.float32)
    adjacency += np.eye(max_atoms, dtype=np.float32)
    adjacency *= mask[:, None] * mask[None, :]
    return node_features, adjacency, mask


def build_graph_batch(records: Iterable[MaterialRecord], max_atoms: int = 8) -> GraphBatchArrays:
    records = list(records)
    descriptors = np.vstack([graph_descriptor(record) for record in records]).astype(np.float32)
    conditions = []
    nodes: List[np.ndarray] = []
    adjs: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    for record in records:
        metrics = record.properties or evaluate_material(record)
        conditions.append([metrics["delta_g_h"], metrics["stability_score"], metrics["synthesis_score"]])
        node_features, adjacency, mask = record_to_graph_arrays(record, max_atoms=max_atoms)
        nodes.append(node_features)
        adjs.append(adjacency)
        masks.append(mask)
    return GraphBatchArrays(
        descriptors=descriptors,
        conditions=np.asarray(conditions, dtype=np.float32),
        node_features=np.stack(nodes),
        adjacency=np.stack(adjs),
        mask=np.stack(masks),
    )
