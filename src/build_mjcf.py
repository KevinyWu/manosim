"""Build mjcf files for a MANO hand.

Emits two MJCFs per side under `<output_root>/<name>/mano_{r,l}hand/`:
  - mesh_hand.xml - one mesh geometry per bone
  - capsule_hand.xml - simplified capsule per phalanx with a mesh palm
"""

from pathlib import Path
from typing import Literal

import numpy as np
import tyro
import yaml
from rich.console import Console

from .mano_build import PARENTS, capsule_fit, forward_mano, resolve_shape
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
            p_a, p_b, radius = capsule_fit(i, b, centers, joints, tips, build)
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

    bodies, centers, joints, tips = forward_mano(
        side, shape, mano_root, build["total_mass_kg"], mesh_dir
    )

    mesh_path = out_dir / MESH_HAND_XML
    capsule_path = out_dir / CAPSULE_HAND_XML
    _emit_mjcf(bodies, centers, joints, tips, mesh_path, "mesh", build)
    _emit_mjcf(bodies, centers, joints, tips, capsule_path, "capsule", build)
    console.print(f"[green]Wrote {mesh_path}[/green]")
    console.print(f"[green]Wrote {capsule_path}[/green]")
    console.print(f"[green]Meshes in {mesh_dir} ({len(bodies)} files)[/green]")


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

    shape = resolve_shape(cfg.get("shape"), shape_file)

    for side, out_dir in (("right", rhand_out), ("left", lhand_out)):
        _build_side(side, out_dir, shape, MANO_MODELS_DIR, build)


if __name__ == "__main__":
    tyro.cli(main)
