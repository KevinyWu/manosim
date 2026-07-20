"""Shared paths and filenames for manosim."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "myhand.yaml"

MANO_MODELS_DIR = PROJECT_ROOT / "assets" / "mano_models"
OUTPUT_ROOT = PROJECT_ROOT / "assets"

RHAND_SUBDIR = "mano_rhand"
LHAND_SUBDIR = "mano_lhand"
BONE_MESHES_SUBDIR = "meshes"
MESH_HAND_XML = "mesh_hand.xml"
CAPSULE_HAND_XML = "capsule_hand.xml"


def resolve_path(p: str) -> Path:
    """Resolve a yaml path against PROJECT_ROOT; pass absolutes through."""
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path
