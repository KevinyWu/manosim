"""Build mjcf files for a MANO hand.

Emits two MJCFs per side under `<output_root>/<name>/mano_{r,l}hand/`:
  - mesh_hand.xml - one mesh geometry per bone
  - capsule_hand.xml - simplified capsule per phalanx with a mesh palm
"""

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import trimesh
import tyro
import yaml
from manotorch.manolayer import ManoLayer
from rich.console import Console
from scipy.spatial import ConvexHull

from .paths import (
    BONE_MESHES_SUBDIR,
    CAPSULE_HAND_XML,
    DEFAULT_CONFIG_PATH,
    LHAND_SUBDIR,
    MANO_MODELS_DIR,
    MESH_HAND_XML,
    OUTPUT_ROOT,
    RHAND_SUBDIR,
    resolve_path,
)

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


def _resolve_shape(
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


def _segment_by_bone(
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


def _orient_outward(verts_centered: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Flip winding so normals point away from origin (convex, centered mesh)."""
    out = []
    for f in faces:
        v0, v1, v2 = verts_centered[f]
        n = np.cross(v1 - v0, v2 - v0)
        out.append([f[0], f[2], f[1]] if np.dot(n, v0) < 0 else list(f))
    return np.asarray(out)


def _fit_capsule_radius(
    v_local: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, radius_pct: float
) -> float:
    """Percentile of perpendicular distance from the bone cloud to axis p_a -> p_b."""
    axis = p_b - p_a
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    rel = v_local - p_a
    s = rel @ axis
    perp = rel - np.outer(s, axis)
    return float(np.percentile(np.linalg.norm(perp, axis=1), radius_pct))


def _save_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    """Save a mesh to an OBJ file."""
    lines = [f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in verts]
    lines += [f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}" for f in faces]
    path.write_text("\n".join(lines) + "\n")


def _prepare_bodies(
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
        hull_faces = _orient_outward(v_local, hull_faces)
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
        _save_obj(obj_path, v_local, hull_faces)
        bodies[i] = {
            "v_local": v_local,
            "hull_faces": hull_faces,
            "mass": mass,
            "inertia": inertia,
            "cm": np.asarray(cm),
            "obj_path": obj_path,
        }
    return bodies, centers


def _inertia_str_mjcf(inertia: np.ndarray) -> str:
    """Convert inertia array to a string in MJCF format."""
    ixx, iyy, izz = inertia[0, 0], inertia[1, 1], inertia[2, 2]
    ixy, ixz, iyz = inertia[0, 1], inertia[0, 2], inertia[1, 2]
    return f"{ixx:.6e} {iyy:.6e} {izz:.6e} {ixy:.6e} {ixz:.6e} {iyz:.6e}"


def _emit_mjcf(
    bodies: dict[int, dict],
    centers: np.ndarray,
    joints: np.ndarray,
    tips: np.ndarray,
    mjcf_path: Path,
    collision_shape: Literal["mesh", "capsule"],
    build: dict,
) -> None:
    """Emit a minimal actuated-wrist MJCF file."""
    mjcf_dir = mjcf_path.parent
    color = (
        build["mesh_hand_color"]
        if collision_shape == "mesh"
        else build["capsule_hand_color"]
    )
    r, g, blu = color
    rgba = f"{r / 255:.3f} {g / 255:.3f} {blu / 255:.3f} 1"

    act = build["actuator"]
    jnt = build["joint"]
    palm = build["palm_contact"]
    dflt = build["default_geom"]
    radius_overrides = {
        int(k): float(v) for k, v in build["capsule_radius_pct_overrides"].items()
    }

    asset_xml: list[str] = []
    actuator_xml: list[str] = []
    body_xml: list[str] = [""] * 16

    for i in range(16):
        if i not in bodies:
            continue
        b = bodies[i]
        mesh_name = f"mano_bone_{i}"
        asset_xml.append(
            f'<mesh name="{mesh_name}" '
            f'file="{b["obj_path"].relative_to(mjcf_dir).as_posix()}"/>'
        )

        cm = b["cm"]
        geom_pos = np.zeros(3)
        is_palm = PARENTS[i] == -1

        if is_palm:
            # Palm body frame at MANO wrist origin (not centroid) so wrist
            # slide ctrl maps to wrist-joint world position. Compensate the
            # mesh offset + center-of-mass with centers[i].
            pos = np.zeros(3)
            cm = b["cm"] + centers[i]
            geom_pos = centers[i]
            joint_xml = (
                f'<joint name="wrist_tx" type="slide" axis="1 0 0" '
                f'pos="0 0 0" limited="false" damping="{jnt["wrist_trans_damping"]}"/>\n      '
                f'<joint name="wrist_ty" type="slide" axis="0 1 0" '
                f'pos="0 0 0" limited="false" damping="{jnt["wrist_trans_damping"]}"/>\n      '
                f'<joint name="wrist_tz" type="slide" axis="0 0 1" '
                f'pos="0 0 0" limited="false" damping="{jnt["wrist_trans_damping"]}"/>\n      '
                f'<joint name="wrist_rot" type="ball" pos="0 0 0" '
                f'limited="false" damping="{jnt["wrist_rot_damping"]}" '
                f'armature="{jnt["wrist_rot_armature"]}"/>'
            )
            for lbl in "xyz":
                actuator_xml.append(
                    f'<position name="wrist_motor_t{lbl}" '
                    f'joint="wrist_t{lbl}" kp="{act["wrist_trans_kp"]}" '
                    f'dampratio="{act["dampratio"]}"/>'
                )
            for axis, lbl in zip(("1 0 0", "0 1 0", "0 0 1"), "xyz"):
                actuator_xml.append(
                    f'<position name="wrist_motor_r{lbl}" '
                    f'joint="wrist_rot" gear="{axis}" '
                    f'kp="{act["wrist_rot_kp"]}" '
                    f'dampratio="{act["wrist_rot_dampratio"]}"/>'
                )
        else:
            parent_origin = np.zeros(3) if PARENTS[i] == 0 else centers[PARENTS[i]]
            pos = centers[i] - parent_origin
            jpos = joints[i] - centers[i]
            joint_xml = (
                f'<joint name="mano_joint_{i}" type="ball" '
                f'pos="{jpos[0]:.6g} {jpos[1]:.6g} {jpos[2]:.6g}" '
                f'limited="false" damping="{jnt["finger_damping"]}" '
                f'armature="{jnt["finger_armature"]}"/>'
            )
            for axis, lbl in zip(("1 0 0", "0 1 0", "0 0 1"), "xyz"):
                actuator_xml.append(
                    f'<position name="mano_motor_{i}_{lbl}" '
                    f'joint="mano_joint_{i}" gear="{axis}" '
                    f'kp="{act["finger_kp"]}" dampratio="{act["dampratio"]}"/>'
                )

        pos_str = " ".join(f"{x:.6g}" for x in pos)
        cm_str = " ".join(f"{x:.6g}" for x in cm)
        geom_pos_str = " ".join(f"{x:.6g}" for x in geom_pos)

        if collision_shape == "mesh" or is_palm:
            palm_soft = (
                f' priority="{palm["priority"]}" solimp="{palm["solimp"]}"'
                if is_palm
                else ""
            )
            geoms_xml = (
                f'<geom type="mesh" mesh="{mesh_name}" '
                f'pos="{geom_pos_str}" rgba="{rgba}"{palm_soft}/>'
            )
        else:
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
            radius = _fit_capsule_radius(b["v_local"], p_a, p_b, radius_pct=pct)
            # Tip landmark sits on the mesh surface (outermost vertex), so the
            # hemispherical cap would poke past the fingertip. Pull p_b back
            # along the axis by `radius` so the cap apex lands on the landmark.
            if not kids:
                seg = p_b - p_a
                seg_len = float(np.linalg.norm(seg))
                if seg_len > 1e-9:
                    p_b = p_b - (radius / seg_len) * seg
            if np.linalg.norm(p_b - p_a) < 1e-5:
                center_local = 0.5 * (p_a + p_b) + geom_pos
                center_str = " ".join(f"{x:.6g}" for x in center_local)
                geoms_xml = (
                    f'<geom type="sphere" size="{radius:.6g}" '
                    f'pos="{center_str}" rgba="{rgba}"/>'
                )
            else:
                a_world = p_a + geom_pos
                b_world = p_b + geom_pos
                fromto_str = " ".join(
                    f"{x:.6g}" for x in np.concatenate([a_world, b_world])
                )
                geoms_xml = (
                    f'<geom type="capsule" fromto="{fromto_str}" '
                    f'size="{radius:.6g}" rgba="{rgba}"/>'
                )

        body_xml[i] = (
            f'<body name="mano_body_{i}" pos="{pos_str}">\n'
            f"      {joint_xml}\n"
            f'      <inertial pos="{cm_str}" mass="{b["mass"]:.6g}" '
            f'fullinertia="{_inertia_str_mjcf(b["inertia"])}"/>\n'
            f"      {geoms_xml}"
        )

    closing = [""] * 16
    for i in range(15, -1, -1):
        if not body_xml[i]:
            continue
        closing[i] += "\n    </body>"
        p = PARENTS[i]
        if p != -1:
            closing[p] = "\n    " + body_xml[i] + closing[i] + closing[p]
    root_xml = body_xml[0] + closing[0]

    xml = (
        '<mujoco model="mano_hand">\n'
        '  <compiler autolimits="true" angle="radian"/>\n'
        "  <default>\n"
        f'    <geom solimp="{dflt["solimp"]}" solref="{dflt["solref"]}"/>\n'
        "  </default>\n"
        "  <asset>\n    " + "\n    ".join(asset_xml) + "\n  </asset>\n"
        "  <worldbody>\n    " + root_xml + "\n  </worldbody>\n"
        "  <actuator>\n    " + "\n    ".join(actuator_xml) + "\n  </actuator>\n"
        "</mujoco>\n"
    )
    mjcf_path.write_text(xml)


def _build_side(
    side: Literal["right", "left"],
    out_dir: Path,
    shape: np.ndarray,
    mano_root: Path,
    build: dict,
) -> None:
    """Forward ManoLayer at zero pose for side and emit both MJCFs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = out_dir / BONE_MESHES_SUBDIR

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

    submeshes = _segment_by_bone(verts, faces, weights)
    bones_found = sorted(sm["joint_idx"] for sm in submeshes)
    console.print(
        f"[cyan]Segmented {len(submeshes)}/16 bones[/cyan] indices={bones_found}"
    )

    bodies, centers = _prepare_bodies(submeshes, build["total_mass_kg"], mesh_dir)

    mesh_path = out_dir / MESH_HAND_XML
    capsule_path = out_dir / CAPSULE_HAND_XML
    _emit_mjcf(bodies, centers, joints, tips, mesh_path, "mesh", build)
    _emit_mjcf(bodies, centers, joints, tips, capsule_path, "capsule", build)
    console.print(f"[green]Wrote {mesh_path}[/green]")
    console.print(f"[green]Wrote {capsule_path}[/green]")
    console.print(f"[green]Meshes in {mesh_dir} ({len(submeshes)} files)[/green]")


def main(config: Path = DEFAULT_CONFIG_PATH) -> None:
    """Build mesh + capsule MJCFs for both MANO hands.

    Args:
        config: Path to the YAML config file.
    """
    cfg = yaml.safe_load(Path(config).read_text())
    build = cfg["build"]
    name = cfg["name"]

    shape_file = resolve_path(cfg["shape_file"]) if cfg.get("shape_file") else None
    out_root = OUTPUT_ROOT / name
    rhand_out = out_root / RHAND_SUBDIR
    lhand_out = out_root / LHAND_SUBDIR

    shape = _resolve_shape(cfg.get("shape"), shape_file)

    for side, out_dir in (("right", rhand_out), ("left", lhand_out)):
        _build_side(side, out_dir, shape, MANO_MODELS_DIR, build)


if __name__ == "__main__":
    tyro.cli(main)
