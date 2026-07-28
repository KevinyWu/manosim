"""MANO geometry pipeline."""

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import trimesh
from manotorch.manolayer import ManoLayer
from rich.console import Console
from scipy.spatial import ConvexHull

console = Console()

# MANO joint order, wrist (bone 0) at the palm, thumb on top:
#        15--14--13-----\
#                        \
#        3-- 2-- 1-------0
#        6-- 5-- 4------/
#       12--11--10-----/
#        9-- 8-- 7----/
PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14)
CHILDREN = {p: [c for c in range(16) if PARENTS[c] == p] for p in range(16)}

# ManoLayer emits 21 native landmarks in the order shown below.
# SNAP_TO_NATIVE maps the bone order above onto these native indices.
#
#        4-- 3-- 2-- 1-----\
#                           \
#        8-- 7-- 6-- 5------0
#       12--11--10-- 9-----/
#       16--15--14--13----/
#       20--19--18--17---/
TIPS_NATIVE_IDX = (4, 8, 12, 16, 20)  # thumb, index, middle, ring, pinky tips
TIP_LANDMARK_OF_BONE: dict[int, int] = {3: 1, 6: 2, 9: 4, 12: 3, 15: 0}

# Bone order above -> native landmark index
SNAP_TO_NATIVE = (0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3)


def resolve_shape(
    yaml_shape: Optional[list[float]], shape_file: Optional[Path]
) -> np.ndarray:
    """yaml shape > shape_file > mean hand (zeros). Returns (1, 10) float32."""
    if yaml_shape is not None:
        assert len(yaml_shape) == 10, f"shape needs 10 floats, got {len(yaml_shape)}"
        console.print("[cyan]Using yaml shape[/cyan]")
        return np.asarray(yaml_shape, dtype=np.float32).reshape(1, 10)
    if shape_file is not None and shape_file.exists():
        shape = np.load(shape_file).astype(np.float32)
        console.print(f"[cyan]Loaded shape from {shape_file}[/cyan]")
        return shape[None] if shape.ndim == 1 else shape
    console.print("[yellow]No shape provided; using mean hand (zeros)[/yellow]")
    return np.zeros((1, 10), dtype=np.float32)


def segment_by_bone(
    verts: np.ndarray, faces: np.ndarray, weights: np.ndarray
) -> list[dict]:
    """Partition mesh per bone via LBS argmax. Drops faces straddling bones."""
    bone_of_vert = np.argmax(weights, axis=1)
    submeshes = []
    for bone_idx in range(16):
        mask = bone_of_vert == bone_idx
        if not mask.any():
            continue
        old_indices = np.where(mask)[0]
        remap = {old: new for new, old in enumerate(old_indices)}
        sub_verts = verts[old_indices]
        kept = [
            [remap[v] for v in face] for face in faces if all(v in remap for v in face)
        ]
        if not kept:
            continue
        submeshes.append(
            {"vertices": sub_verts, "faces": np.asarray(kept), "joint_idx": bone_idx}
        )
    return submeshes


def orient_outward(verts_centered: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Flip winding so normals point away from origin (convex, centered mesh)."""
    out = []
    for f in faces:
        v0, v1, v2 = verts_centered[f]
        n = np.cross(v1 - v0, v2 - v0)
        out.append([f[0], f[2], f[1]] if np.dot(n, v0) < 0 else list(f))
    return np.asarray(out)


def fit_capsule_radius(
    v_local: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, radius_pct: float
) -> float:
    """Percentile of perpendicular distance from the bone cloud to axis p_a -> p_b."""
    axis = p_b - p_a
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    rel = v_local - p_a
    s = rel @ axis
    perp = rel - np.outer(s, axis)
    return float(np.percentile(np.linalg.norm(perp, axis=1), radius_pct))


def save_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    """Save a mesh to an OBJ file."""
    lines = [f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in verts]
    lines += [f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}" for f in faces]
    path.write_text("\n".join(lines) + "\n")


def prepare_bodies(
    submeshes: list[dict],
    total_mass_kg: float,
    mesh_dir: Path,
) -> tuple[dict[int, dict], np.ndarray]:
    """Compute hull, center, mass, inertia, cm per bone; write OBJ files."""
    mesh_dir.mkdir(parents=True, exist_ok=True)
    centers = np.zeros((16, 3))
    hulls: dict[int, tuple[np.ndarray, np.ndarray, trimesh.Trimesh]] = {}
    for sm in submeshes:
        i = sm["joint_idx"]
        v, f = sm["vertices"], sm["faces"]
        hull_faces = ConvexHull(v).simplices if v.shape[0] >= 4 else f
        center = v.mean(axis=0)
        centers[i] = center
        v_local = v - center
        hull_faces = orient_outward(v_local, hull_faces)
        hulls[i] = (
            v_local,
            hull_faces,
            trimesh.Trimesh(vertices=v_local, faces=hull_faces, process=False),
        )

    vols = {i: max(h[2].volume, 1e-9) for i, h in hulls.items()}
    vol_sum = sum(vols.values())

    bodies: dict[int, dict] = {}
    for i, (v_local, hull_faces, mesh) in hulls.items():
        mass = total_mass_kg * vols[i] / vol_sum
        density = mass / max(mesh.volume, 1e-9)
        inertia = mesh.moment_inertia * density
        cm = mesh.center_mass
        obj_path = mesh_dir / f"mano_bone_{i}.obj"
        save_obj(obj_path, v_local, hull_faces)
        bodies[i] = {
            "v_local": v_local,
            "hull_faces": hull_faces,
            "mass": mass,
            "inertia": inertia,
            "cm": np.asarray(cm),
            "obj_path": obj_path,
        }
    return bodies, centers


def forward_mano(
    side: Literal["right", "left"],
    shape: np.ndarray,
    mano_root: Path,
    total_mass_kg: float,
    mesh_dir: Path,
) -> tuple[dict[int, dict], np.ndarray, np.ndarray, np.ndarray]:
    """Forward ManoLayer at zero pose. Returns bodies, centers, joints, tips."""
    console.print(f"[cyan]Forwarding MANO ({side}) at zero pose[/cyan]")
    layer = ManoLayer(
        rot_mode="axisang",
        side=side,
        center_idx=0,
        mano_assets_root=str(mano_root),
        use_pca=False,
        flat_hand_mean=True,
    )
    # Right shape file used for both sides; mirror x of the shape basis for left
    if side == "left":
        layer.th_shapedirs[:, 0, :] *= -1

    pose_coeffs = torch.zeros((1, 48), dtype=torch.float32)
    shape_t = torch.tensor(shape, dtype=torch.float32)
    out = layer(pose_coeffs, shape_t)

    verts = out.verts.detach().cpu().numpy()[0]
    joints_all = out.joints.detach().cpu().numpy()[0]
    joints = joints_all[list(SNAP_TO_NATIVE)]
    tips = joints_all[list(TIPS_NATIVE_IDX)]
    weights = layer.th_weights.detach().cpu().numpy()
    faces = layer.th_faces.detach().cpu().numpy()

    submeshes = segment_by_bone(verts, faces, weights)
    bones_found = sorted(sm["joint_idx"] for sm in submeshes)
    console.print(
        f"[cyan]Segmented {len(submeshes)}/16 bones[/cyan] indices={bones_found}"
    )

    bodies, centers = prepare_bodies(submeshes, total_mass_kg, mesh_dir)
    return bodies, centers, joints, tips


def capsule_fit(
    i: int,
    body: dict,
    centers: np.ndarray,
    joints: np.ndarray,
    tips: np.ndarray,
    build: dict,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit bone i's capsule. Returns p_a, p_b, radius relative to its centroid."""
    radius_overrides = {
        int(k): float(v) for k, v in build["capsule_radius_pct_overrides"].items()
    }
    # Capsule spans landmark -> landmark: proximal joint of this bone to
    # either its kinematic child joint or the corresponding fingertip
    # landmark (for tip phalanges). Length = anatomical bone length, not
    # the vertex extent — sidesteps LBS-argmax truncation near joints.
    kids = CHILDREN[i]
    if kids:
        distal_landmark = joints[kids[0]]
    else:
        distal_landmark = tips[TIP_LANDMARK_OF_BONE[i]]
    p_a = joints[i] - centers[i]
    p_b = distal_landmark - centers[i]
    pct = radius_overrides.get(i, build["capsule_radius_pct"])
    radius = fit_capsule_radius(body["v_local"], p_a, p_b, radius_pct=pct)
    # Tip landmark sits on the mesh surface (outermost vertex), so the
    # hemispherical cap would poke past the fingertip. Pull p_b back
    # along the axis by `radius` so the cap apex lands on the landmark.
    if not kids:
        seg = p_b - p_a
        seg_len = float(np.linalg.norm(seg))
        if seg_len > 1e-9:
            p_b = p_b - (radius / seg_len) * seg
    return p_a, p_b, radius
