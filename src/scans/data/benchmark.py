from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkDataset:
    key: str
    display_name: str
    remote_name: str | None
    small: bool
    expected_nodes: int
    expected_edges: int
    edge_path: str | None = None

    @property
    def evaluation_negative_sets(self) -> tuple[str, ...]:
        return ("sns", "mns", "cns", "mix")


SEHP_DATASETS: tuple[BenchmarkDataset, ...] = (
    # The DHG coauthorship files correspond to the AHP/HyGEN authorship
    # datasets: nodes are papers and hyperedges group papers by author.
    BenchmarkDataset("cora", "Cora-A", "coauthorship_cora", True, 2388, 970),
    BenchmarkDataset("dblp", "DBLP-A", "coauthorship_dblp", True, 41302, 20865),
    BenchmarkDataset("citeseer", "Citeseer", "cocitation_citeseer", True, 1458, 1004),
    BenchmarkDataset("pubmed", "Pubmed", "cocitation_pubmed", True, 3840, 7531),
    BenchmarkDataset(
        "cora_cocitation",
        "Cora",
        None,
        True,
        1434,
        1483,
        "data/sehp/processed/cora_cocitation_edges.txt",
    ),
    BenchmarkDataset(
        "dblp_collaboration",
        "DBLP",
        None,
        True,
        15639,
        18390,
        "data/sehp/processed/dblp_collaboration_edges.txt",
    ),
    BenchmarkDataset("ndc_class", "NDC_class", None, True, 1149, 1047),
    BenchmarkDataset("email_enron", "Email-Enron", None, True, 143, 1459, "data/processed/email_enron_edges.txt"),
    BenchmarkDataset("email_eu", "Email-Eu", None, True, 998, 0, "data/processed/email_eu_edges.txt"),
    BenchmarkDataset("contact_high_school", "Contact-High", None, True, 327, 7818, "data/processed/contact_high_school_edges.txt"),
    BenchmarkDataset("contact_primary_school", "Contact-Primary", None, True, 242, 12704, "data/processed/contact_primary_school_edges.txt"),
    BenchmarkDataset("tags_ask_ubuntu", "Tags-AskUbuntu", None, True, 0, 0, "data/processed/tags_ask_ubuntu_edges.txt"),
    BenchmarkDataset("tags_math_sx", "Tags-MathSX", None, True, 0, 0, "data/processed/tags_math_sx_edges.txt"),
    BenchmarkDataset("tags_stack_overflow", "Tags-StackOverflow", None, True, 0, 0, "data/processed/tags_stack_overflow_edges.txt"),
    BenchmarkDataset("coauth_mag_geology", "Coauth-MAG-Geology", None, True, 0, 0, "data/processed/coauth_mag_geology_edges.txt"),
    BenchmarkDataset("coauth_mag_history", "Coauth-MAG-History", None, True, 0, 0, "data/processed/coauth_mag_history_edges.txt"),
    BenchmarkDataset("dblp_10k", "DBLP-10K", None, True, 31968, 10000, "data/processed/coauth_dblp_10k_edges.txt"),
    BenchmarkDataset("recipe100k", "Recipe100k", "recipe-100k-v2", False, 100896, 11822),
    BenchmarkDataset("recipe200k", "Recipe200k", "recipe-200k-v2", False, 240094, 18049),
)

# Mechanism-based extended benchmark set. These datasets have enough repeated
# local higher-order structure for anchored retrieval and remain feasible under
# the mandatory full-data 60/20/20 protocol.
SCANS_EXTENDED_DATASETS: tuple[str, ...] = (
    "cora_cocitation",
    "dblp_collaboration",
    "pubmed",
    "email_eu",
    "contact_high_school",
)

SEHP_BASELINES: tuple[str, ...] = (
    "HyperGCN",
    "UniGCNII",
    "HDS",
    "EDGNN",
    "HyperSAGNN",
    "NHP",
    "AHP",
    "SEHP",
)

SEHP_SPLIT = {"train": 0.6, "val": 0.2, "test": 0.2}
SEHP_METRICS = ("auc", "precision")
DHG_REMOTE_ROOT = "https://download.moon-lab.tech:28501/datasets"


def dataset_by_key(key: str) -> BenchmarkDataset:
    for dataset in SEHP_DATASETS:
        if dataset.key == key:
            return dataset
    raise KeyError(key)


def processed_edge_path(root: Path, key: str) -> Path:
    dataset = dataset_by_key(key)
    if dataset.edge_path is not None:
        return root / dataset.edge_path
    return root / "data" / "sehp" / "processed" / f"{key}_edges.txt"


def processed_feature_path(root: Path, key: str) -> Path:
    return root / "data" / "sehp" / "processed" / f"{key}_features.pkl"
