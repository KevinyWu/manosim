"""Live MJCF physics with MANO overlay (LBS)."""

import types
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro
import viser
import yaml
from manotorch.manolayer import ManoLayer
from mjviser import Viewer
from rich.console import Console

from .paths import (
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

mujoco.set_mju_user_warning(lambda msg: None)

# MANO overlay color; mesh + capsule hand colors are built into the MJCFs
MANO_COLOR = (0.35, 0.78, 1.00)
SKELETON_COLOR = (0.2, 0.8, 1.0)
KEYPOINT_COLOR = (0.0, 0.5, 1.0)

# Initial hand opacity
MANO_OVERLAY_OPACITY = 0.6
CAPSULE_HAND_OPACITY = 0.6
MESH_HAND_OPACITY = 0.6

# Initial camera pose
CAMERA_POSITION = (0.0, -0.4, 0.1)
CAMERA_LOOK_AT = (0.0, 0.0, 0.1)
CAMERA_UP = (0.0, 0.0, 1.0)

# Geom-group convention used by the merged MJCF
GROUP_MESH = 0
GROUP_CAPSULE = 1

# Stand both hands vertical (fingers pointing world +z)
_S = 2**0.5 / 2
QUAT_RIGHT = (_S, 0.0, _S, 0.0)
QUAT_LEFT = (_S, 0.0, -_S, 0.0)
R_RIGHT = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
R_LEFT = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

# MANO 21-landmark skeleton edges (manotorch joint order)
MANO_SKELETON_PAIRS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11),
    (11, 12),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
)

# MJCF attributes whose values are MuJoCo names
NAME_ATTRS = ("name", "joint", "mesh")

# Fingertip vertex indices
TIP_IDX_RIGHT = (745, 317, 444, 556, 673)
TIP_IDX_LEFT = (745, 317, 445, 556, 673)
JOINT_REORDER = (
    0,
    13,
    14,
    15,
    16,
    1,
    2,
    3,
    17,
    4,
    5,
    6,
    18,
    10,
    11,
    12,
    19,
    7,
    8,
    9,
    20,
)


@dataclass
class _SideRest:
    """Per-side cache used by the render hook to deform the MANO overlay."""

    body_ids: np.ndarray  # (16,) MJCF body ids for mano_body_0..15
    palm_id: int
    weights: np.ndarray  # (778, 16) LBS weights
    j_regressor: np.ndarray  # (16, 778)
    tip_idx: np.ndarray  # (5,) vertex indices for fingertip joints
    reorder: np.ndarray  # (21,) joint reorder for SNAP layout
    v_canonical: np.ndarray  # (16, 778, 4) bone-local homogeneous verts
    pair_arr: np.ndarray  # (E, 2) skeleton edge index pairs
    overlay_handle: object
    skeleton_handle: object
    keypoint_handle: object
    wrist_frame_handle: object
    label_handles: list  # (21,) landmark number labels
    body_label_handles: list  # (16,) MuJoCo body number labels


def _resolve_shape(
    yaml_shape: list[float] | None, shape_file: Path | None
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


def _load_mano_mesh(
    side: str, shape: np.ndarray, mano_root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward ManoLayer at zero pose."""
    layer = ManoLayer(
        rot_mode="axisang",
        side=side,
        center_idx=0,
        mano_assets_root=str(mano_root),
        use_pca=False,
        flat_hand_mean=True,
    )
    if side == "left":
        layer.th_shapedirs[:, 0, :] *= -1
    pose_coeffs = torch.zeros((1, 48), dtype=torch.float32)
    shape_t = torch.tensor(shape, dtype=torch.float32)
    out = layer(pose_coeffs, shape_t)
    verts = out.verts.detach().cpu().numpy()[0]
    # Closed faces seal the open wrist stump (side flip handled internally)
    faces = layer.get_mano_closed_faces().detach().cpu().numpy()
    joints = out.joints.detach().cpu().numpy()[0]
    weights = layer.th_weights.detach().cpu().numpy()
    j_regressor = layer.th_J_regressor.detach().cpu().numpy()
    return verts, faces, joints, weights, j_regressor


def _load_combined_tree(
    hand_dir: Path, capsule_xml: str, mesh_xml: str
) -> ET.ElementTree:
    """Build per-side tree: capsule physics + mesh visual overlay."""
    cap = ET.parse(hand_dir / capsule_xml)
    cap_root = cap.getroot()
    # Set group for capsule geoms
    for geom in cap_root.iter("geom"):
        geom.set("group", str(GROUP_CAPSULE))

    # Parse mesh tree and add mesh geoms to matching capsule bodies
    mesh_tree = ET.parse(hand_dir / mesh_xml)
    cap_bodies = {b.get("name"): b for b in cap_root.iter("body")}
    for mbody in mesh_tree.getroot().iter("body"):
        name = mbody.get("name")
        target = cap_bodies.get(name)
        if target is None:
            continue
        for mgeom in mbody.findall("geom"):
            g = deepcopy(mgeom)
            g.set("group", str(GROUP_MESH))
            # Set contype/conaffinity to 0 so mesh geoms don't collide
            g.set("contype", "0")
            g.set("conaffinity", "0")
            target.append(g)
    return cap


def _prefix_and_offset(
    tree: ET.ElementTree,
    mjcf_dir: Path,
    prefix: str,
    offset: np.ndarray,
    quat_wxyz: tuple[float, float, float, float],
) -> ET.ElementTree:
    """Prefix names, absolutize mesh paths, set palm pos/quat on the tree."""
    root = tree.getroot()
    for mesh in root.iter("mesh"):
        f = mesh.get("file")
        if f and not Path(f).is_absolute():
            mesh.set("file", str((mjcf_dir / f).resolve()))

    # Shorten actuator display names in mjviser
    for act in root.iter("position"):
        nm = act.get("name")
        if nm is None:
            continue
        if nm.startswith("mano_motor_"):
            act.set("name", f"joint_{nm[len('mano_motor_') :]}")
        elif nm.startswith("wrist_motor_"):
            act.set("name", f"wrist_{nm[len('wrist_motor_') :]}")

    for el in root.iter():
        for attr in NAME_ATTRS:
            v = el.get(attr)
            if v is not None:
                el.set(attr, f"{prefix}{v}")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("source MJCF has no <worldbody>")
    for body in worldbody.findall("body"):
        cur = np.array([float(x) for x in body.get("pos", "0 0 0").split()])
        body.set("pos", " ".join(f"{x:.6g}" for x in cur + offset))
        body.set("quat", " ".join(f"{x:.6g}" for x in quat_wxyz))
    return tree


def _merge_two_hands(
    right_dir: Path,
    left_dir: Path,
    spacing_m: float,
    capsule_xml: str,
    mesh_xml: str,
) -> str:
    """Compose a single MJCF: capsule+mesh per side, offset along world x."""
    sides = (
        ("r_", right_dir, np.array([+spacing_m / 2, 0.0, 0.0]), QUAT_RIGHT),
        ("l_", left_dir, np.array([-spacing_m / 2, 0.0, 0.0]), QUAT_LEFT),
    )

    out = ET.Element("mujoco", attrib={"model": "mano_two_hands"})
    compiler = ET.SubElement(out, "compiler")
    compiler.set("autolimits", "true")
    compiler.set("angle", "radian")
    opt = ET.SubElement(out, "option")
    opt.set("integrator", "implicitfast")
    asset = ET.SubElement(out, "asset")
    worldbody = ET.SubElement(out, "worldbody")
    actuator = ET.SubElement(out, "actuator")

    for prefix, hand_dir, offset, quat in sides:
        tree = _load_combined_tree(hand_dir, capsule_xml, mesh_xml)
        src = _prefix_and_offset(tree, hand_dir, prefix, offset, quat).getroot()
        for tag, dest in (
            ("asset", asset),
            ("worldbody", worldbody),
            ("actuator", actuator),
        ):
            sect = src.find(tag)
            if sect is not None:
                for child in sect:
                    dest.append(deepcopy(child))

    ET.indent(out, space="  ")
    return ET.tostring(out, encoding="unicode")


def _body_transforms(data: mujoco.MjData, body_ids: np.ndarray) -> np.ndarray:
    """Pack (xpos, xmat) for given body ids into (N, 4, 4) homogeneous."""
    n = body_ids.shape[0]
    g = np.zeros((n, 4, 4), dtype=np.float64)
    g[:, :3, :3] = data.xmat[body_ids].reshape(n, 3, 3)
    g[:, :3, 3] = data.xpos[body_ids]
    g[:, 3, 3] = 1.0
    return g


def _deform_overlay(rest: _SideRest, data: mujoco.MjData) -> None:
    """LBS deform the MANO mesh using current MJCF body transforms and push to viser."""
    g_cur = _body_transforms(data, rest.body_ids)  # (16, 4, 4)
    v_per_bone = np.einsum("bij,bvj->bvi", g_cur, rest.v_canonical)  # (16, 778, 4)
    v_def = np.einsum("vb,bvi->vi", rest.weights, v_per_bone)[:, :3]  # (778, 3)
    v_def_f32 = v_def.astype(np.float32)

    j16 = rest.j_regressor @ v_def  # (16, 3)
    tips = v_def[rest.tip_idx]  # (5, 3)
    j21 = np.concatenate([j16, tips], axis=0)[rest.reorder]  # (21, 3)
    j21_f32 = j21.astype(np.float32)

    rest.overlay_handle.vertices = v_def_f32
    rest.skeleton_handle.points = j21_f32[rest.pair_arr]
    rest.keypoint_handle.points = j21_f32
    rest.wrist_frame_handle.position = tuple(data.xpos[rest.palm_id].astype(np.float32))
    rest.wrist_frame_handle.wxyz = tuple(data.xquat[rest.palm_id].astype(np.float32))
    for i, lbl in enumerate(rest.label_handles):
        lbl.position = tuple(j21_f32[i])

    body_pos = data.xpos[rest.body_ids].astype(np.float32)  # (16, 3)
    for i, lbl in enumerate(rest.body_label_handles):
        lbl.position = tuple(body_pos[i])


def _setup_grouped_actuator_sliders(self) -> None:
    """Split actuator sliders into Right/Left Hand folders"""
    if self.model.nu == 0:
        return

    groups: dict[str, list] = {"Right Hand": [], "Left Hand": [], "Other": []}
    for i in range(self.model.nu):
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        name = name or f"actuator_{i}"
        limited = bool(self.model.actuator_ctrllimited[i])
        lo, hi = self.model.actuator_ctrlrange[i] if limited else (-1.0, 1.0)
        lo, hi = round(float(lo), 3), round(float(hi), 3)
        if name.startswith("r_"):
            groups["Right Hand"].append((i, name[2:], lo, hi))
        elif name.startswith("l_"):
            groups["Left Hand"].append((i, name[2:], lo, hi))
        else:
            groups["Other"].append((i, name, lo, hi))

    total = sum(len(v) for v in groups.values())
    if total > self._MAX_SLIDERS:
        self._server.gui.add_markdown(
            f"*Actuator sliders disabled ({total} actuators exceed"
            f" limit of {self._MAX_SLIDERS}).*"
        )
        return

    for title, items in groups.items():
        if not items:
            continue
        with self._server.gui.add_folder(title):
            for act_id, label, lo, hi in items:
                val = float(np.clip(self.data.ctrl[act_id], lo, hi))
                slider = self._server.gui.add_slider(
                    label,
                    min=lo,
                    max=hi,
                    step=round((hi - lo) / 200, 4),
                    initial_value=round(val, 3),
                    disabled=False,
                )
                self._actuator_sliders.append((slider, act_id))

                def _on_update(_, _id=act_id, _sl=slider) -> None:
                    with self._lock:
                        self.data.ctrl[_id] = _sl.value

                slider.on_update(_on_update)


def main(
    config: Path = DEFAULT_CONFIG_PATH,
    spacing_m: float = 0.2,
    port: int = 8080,
) -> None:
    """Drive both MANO MJCFs under physics with a static MANO overlay.

    Args:
        config: Path to the YAML config file.
        spacing_m: Separation along world x between right (+x) and left (-x) wrists.
        port: Viser port.
    """
    cfg = yaml.safe_load(Path(config).read_text())
    name = cfg["name"]

    mano_root = MANO_MODELS_DIR
    shape_file = resolve_path(cfg["shape_file"]) if cfg.get("shape_file") else None
    out_root = OUTPUT_ROOT / name
    rhand_out = out_root / RHAND_SUBDIR
    lhand_out = out_root / LHAND_SUBDIR
    capsule_xml = CAPSULE_HAND_XML
    mesh_xml = MESH_HAND_XML

    shape = _resolve_shape(cfg.get("shape"), shape_file)

    sides = [
        (
            "right",
            rhand_out,
            np.array([+spacing_m / 2, 0.0, 0.0]),
            R_RIGHT,
            QUAT_RIGHT,
        ),
        (
            "left",
            lhand_out,
            np.array([-spacing_m / 2, 0.0, 0.0]),
            R_LEFT,
            QUAT_LEFT,
        ),
    ]
    for _, hand_dir, _, _, _ in sides:
        for fname in (capsule_xml, mesh_xml):
            p = hand_dir / fname
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} missing — run `python -m manosim.build_mjcf` first."
                )

    combined_xml = _merge_two_hands(
        rhand_out, lhand_out, spacing_m, capsule_xml, mesh_xml
    )
    model = mujoco.MjModel.from_xml_string(combined_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    server = viser.ViserServer(port=port, verbose=False)

    rest_per_side: list[_SideRest] = []
    pair_arr = np.asarray(MANO_SKELETON_PAIRS, dtype=np.int32)
    reorder_arr = np.asarray(JOINT_REORDER, dtype=np.int32)
    for side, _, offset, R, quat in sides:
        verts, faces, joints, weights, j_regressor = _load_mano_mesh(
            side, shape, mano_root
        )
        verts_w = (verts @ R.T + offset).astype(np.float64)
        joints_w = (joints @ R.T + offset).astype(np.float32)

        prefix = "r_" if side == "right" else "l_"
        body_ids = np.array(
            [
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}mano_body_{i}"
                )
                for i in range(16)
            ],
            dtype=np.int32,
        )
        if (body_ids < 0).any():
            raise RuntimeError(f"missing {prefix}mano_body_* bodies in merged MJCF")
        palm_id = int(body_ids[0])

        g_rest = _body_transforms(data, body_ids)  # (16, 4, 4)
        g_rest_inv = np.linalg.inv(g_rest)  # (16, 4, 4)
        v_h = np.concatenate([verts_w, np.ones((verts_w.shape[0], 1))], axis=1)
        v_canonical = np.einsum("bij,vj->bvi", g_rest_inv, v_h)  # (16, 778, 4)

        h = server.scene.add_mesh_simple(
            f"/{side}/mano",
            vertices=verts_w.astype(np.float32),
            faces=faces,
            color=MANO_COLOR,
            opacity=MANO_OVERLAY_OPACITY,
            flat_shading=False,
        )
        h.visible = False

        segs = joints_w[pair_arr]
        colors = np.tile(SKELETON_COLOR, (len(pair_arr), 2, 1)).astype(np.float32)
        s = server.scene.add_line_segments(
            f"/{side}/skeleton",
            points=segs,
            colors=colors,
            line_width=4.0,
            visible=False,
        )

        k = server.scene.add_point_cloud(
            f"/{side}/keypoints",
            points=joints_w,
            colors=np.tile(KEYPOINT_COLOR, (len(joints_w), 1)).astype(np.float32),
            point_size=0.005,
            point_shape="rounded",
            visible=False,
        )

        wf = server.scene.add_frame(
            f"/{side}/wrist_frame",
            wxyz=quat,
            position=tuple(offset.tolist()),
            axes_length=0.05,
            axes_radius=0.002,
        )
        wf.visible = True

        labels = [
            server.scene.add_label(
                f"/{side}/joint_labels/{i}",
                text=str(i),
                position=tuple(joints_w[i]),
                visible=False,
            )
            for i in range(len(joints_w))
        ]

        # MuJoCo actuation body (mano_body_0..15) number labels
        bodies_w = data.xpos[body_ids].astype(np.float32)  # (16, 3)
        body_labels = [
            server.scene.add_label(
                f"/{side}/body_labels/{i}",
                text=str(i),
                position=tuple(bodies_w[i]),
            )
            for i in range(len(bodies_w))
        ]

        tip_idx = np.asarray(
            TIP_IDX_RIGHT if side == "right" else TIP_IDX_LEFT, dtype=np.int32
        )
        rest_per_side.append(
            _SideRest(
                body_ids=body_ids,
                palm_id=palm_id,
                weights=weights.astype(np.float64),
                j_regressor=j_regressor.astype(np.float64),
                tip_idx=tip_idx,
                reorder=reorder_arr,
                v_canonical=v_canonical,
                pair_arr=pair_arr,
                overlay_handle=h,
                skeleton_handle=s,
                keypoint_handle=k,
                wrist_frame_handle=wf,
                label_handles=labels,
                body_label_handles=body_labels,
            )
        )

    overlay_handles = [r.overlay_handle for r in rest_per_side]
    skeleton_handles = [r.skeleton_handle for r in rest_per_side]
    keypoint_handles = [r.keypoint_handle for r in rest_per_side]
    wrist_frame_handles = [r.wrist_frame_handle for r in rest_per_side]
    label_handles = [lbl for r in rest_per_side for lbl in r.label_handles]
    body_label_handles = [lbl for r in rest_per_side for lbl in r.body_label_handles]

    gui_skel = server.gui.add_checkbox("MANO Skeleton", initial_value=False)
    gui_kp = server.gui.add_checkbox("MANO Keypoints", initial_value=False)
    gui_mano = server.gui.add_checkbox("MANO Overlay", initial_value=False)

    @gui_mano.on_update
    def _(_event) -> None:
        for h in overlay_handles:
            h.visible = gui_mano.value

    @gui_skel.on_update
    def _(_event) -> None:
        for s in skeleton_handles:
            s.visible = gui_skel.value

    @gui_kp.on_update
    def _(_event) -> None:
        for k in keypoint_handles:
            k.visible = gui_kp.value

    mesh_hand_handles: list = []
    capsule_hand_handles: list = []

    def _patched_bmt(name, mesh, batched_wxyzs, batched_positions, **kwargs):
        n = int(np.asarray(batched_wxyzs).shape[0])
        try:
            vc = np.asarray(mesh.visual.vertex_colors)
            rgb = vc[0, :3].astype(np.uint8)
        except Exception:
            rgb = np.array([90, 200, 255], dtype=np.uint8)
        init_op = (
            MESH_HAND_OPACITY if f"group{GROUP_MESH}" in name else CAPSULE_HAND_OPACITY
        )
        h = server.scene.add_batched_meshes_simple(
            name,
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.int32),
            batched_wxyzs=batched_wxyzs,
            batched_positions=batched_positions,
            batched_scales=kwargs.get("batched_scales"),
            batched_colors=np.tile(rgb, (n, 1)),
            batched_opacities=np.full(n, init_op, dtype=np.float32),
            lod=kwargs.get("lod", "auto"),
            cast_shadow=kwargs.get("cast_shadow", True),
            receive_shadow=kwargs.get("receive_shadow", True),
            visible=kwargs.get("visible", True),
        )
        if f"group{GROUP_MESH}" in name:
            mesh_hand_handles.append(h)
        elif f"group{GROUP_CAPSULE}" in name:
            capsule_hand_handles.append(h)
        return h

    server.scene.add_batched_meshes_trimesh = _patched_bmt

    camera_set_clients: set[int] = set()

    def _set_camera(client: viser.ClientHandle) -> None:
        client.camera.position = CAMERA_POSITION
        client.camera.look_at = CAMERA_LOOK_AT
        client.camera.up_direction = CAMERA_UP

    def _render_fn(scene) -> None:
        scene.update_from_mjdata(data)
        for rest in rest_per_side:
            _deform_overlay(rest, data)
        for cid, client in server.get_clients().items():
            if cid not in camera_set_clients:
                _set_camera(client)
                camera_set_clients.add(cid)

    viewer = Viewer(model, data, server=server, render_fn=_render_fn)
    viewer.scene.camera_tracking_enabled = False
    # Split actuator sliders into Right/Left Hand folders (runs in viewer.run())
    viewer._setup_actuator_sliders = types.MethodType(
        _setup_grouped_actuator_sliders, viewer
    )

    # Default: capsule physics hand on, mesh visual hand off
    viewer.scene.geom_groups_visible[GROUP_MESH] = False
    viewer.scene.geom_groups_visible[GROUP_CAPSULE] = True
    viewer.scene._sync_visibilities()

    gui_capsule_hand = server.gui.add_checkbox("Capsule Hand", initial_value=True)
    gui_mesh_hand = server.gui.add_checkbox("Mesh Hand", initial_value=False)
    gui_mano_opacity = server.gui.add_slider(
        "MANO Overlay Opacity",
        min=0.0,
        max=1.0,
        step=0.05,
        initial_value=MANO_OVERLAY_OPACITY,
    )
    gui_capsule_opacity = server.gui.add_slider(
        "Capsule Hand Opacity",
        min=0.0,
        max=1.0,
        step=0.05,
        initial_value=CAPSULE_HAND_OPACITY,
    )
    gui_mesh_opacity = server.gui.add_slider(
        "Mesh Hand Opacity",
        min=0.0,
        max=1.0,
        step=0.05,
        initial_value=MESH_HAND_OPACITY,
    )
    gui_labels = server.gui.add_checkbox("Landmark Numbers", initial_value=False)
    gui_body_labels = server.gui.add_checkbox("Body Numbers", initial_value=True)
    gui_wf = server.gui.add_checkbox("Wrist Frame", initial_value=True)

    @gui_labels.on_update
    def _(_event) -> None:
        for lbl in label_handles:
            lbl.visible = gui_labels.value

    @gui_body_labels.on_update
    def _(_event) -> None:
        for lbl in body_label_handles:
            lbl.visible = gui_body_labels.value

    @gui_wf.on_update
    def _(_event) -> None:
        for wf in wrist_frame_handles:
            wf.visible = gui_wf.value

    @gui_mesh_hand.on_update
    def _(_event) -> None:
        viewer.scene.geom_groups_visible[GROUP_MESH] = gui_mesh_hand.value
        viewer.scene._sync_visibilities()

    @gui_capsule_hand.on_update
    def _(_event) -> None:
        viewer.scene.geom_groups_visible[GROUP_CAPSULE] = gui_capsule_hand.value
        viewer.scene._sync_visibilities()

    @gui_mano_opacity.on_update
    def _(_event) -> None:
        a = float(gui_mano_opacity.value)
        for h in overlay_handles:
            h.opacity = a

    @gui_mesh_opacity.on_update
    def _(_event) -> None:
        a = float(gui_mesh_opacity.value)
        for h in mesh_hand_handles:
            n = int(h.batched_positions.shape[0])
            h.batched_opacities = np.full(n, a, dtype=np.float32)

    @gui_capsule_opacity.on_update
    def _(_event) -> None:
        a = float(gui_capsule_opacity.value)
        for h in capsule_hand_handles:
            n = int(h.batched_positions.shape[0])
            h.batched_opacities = np.full(n, a, dtype=np.float32)

    viewer.run()


if __name__ == "__main__":
    tyro.cli(main)
