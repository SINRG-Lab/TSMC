import os
import re
import cv2
import open3d as o3d
import numpy as np
from collections import deque

def flip_uv_v_inplace(mesh: o3d.geometry.TriangleMesh):
    if not mesh.has_triangle_uvs():
        return
    uvs = np.asarray(mesh.triangle_uvs)
    uvs[:, 1] = 1.0 - uvs[:, 1]
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)

def select_viewpoints(mesh, gt_mesh, num_views=4, width=1080, height=1920):
    """
    Interactively select viewpoints for rendering.

    Args:
        mesh (o3d.geometry.TriangleMesh): Mesh to visualize.
        gt_mesh (o3d.geometry.TriangleMesh): Ground truth mesh.
        num_views (int): Number of viewpoints to select.

    Returns:
        list: List of PinholeCameraParameters for selected viewpoints.
    """
    viewpoints = []
    print(f"Select {num_views} viewpoints. Adjust the view, press 'Q' to close the window, then choose to save via console.")

    for i in range(num_views * 2):  # Allow extra attempts
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height)
        vis.add_geometry(mesh)
        vis.add_geometry(gt_mesh)
        vis.run()

        # Get current view parameters
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        vis.destroy_window()

        # Prompt user to save viewpoint
        #response = input(f"Save viewpoint {len(viewpoints)+1}? (y/n): ").strip().lower()
        #if response == 'y':
            # Check if viewpoint is unique
        is_unique = True
        for existing_param in viewpoints:
            if np.allclose(param.extrinsic, existing_param.extrinsic, atol=1e-3):
                print("Viewpoint is too similar to a previous one. Please select a different view.")
                is_unique = False
                break
        if is_unique:
            viewpoints.append(param)
            print(f"Viewpoint {len(viewpoints)} saved. Extrinsic: {param.extrinsic}")
        #else:
        #    print(f"Viewpoint skipped. Select another.")

        if len(viewpoints) >= num_views:
            break

    if len(viewpoints) < num_views:
        print(f"Warning: Only {len(viewpoints)} viewpoints saved. Proceeding with available views.")

    return viewpoints[:num_views]

def render_mesh_texture(mesh, width=512, height=512, camera_params=None, enable_lighting=False):
    """
    Render textured RGB + depth using OffscreenRenderer (Open3D 0.19.x).
    Returns: (rgb_uint8 HxWx3, depth_uint8 HxW, (dmin, dmax))
    """
    mesh_copy = mesh

    if enable_lighting and not mesh_copy.has_vertex_normals():
        mesh_copy.compute_vertex_normals()

    # ---- Make a "held" texture image if present ----
    tex_img = None
    if hasattr(mesh_copy, "textures") and len(mesh_copy.textures) > 0:
        # Convert to numpy and re-wrap -> ensures a held instance owned by Python
        tex_np = np.asarray(mesh_copy.textures[0])
        if tex_np is not None and tex_np.size > 0:
            tex_img = o3d.geometry.Image(np.ascontiguousarray(tex_np))

    use_texture = (tex_img is not None) and mesh_copy.has_triangle_uvs()

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit" if enable_lighting else "defaultUnlit"
    mat.base_color = [1.0, 1.0, 1.0, 1.0]

    if use_texture:
        mat.albedo_img = tex_img
        if enable_lighting:
            mat.roughness = 1.0
            mat.metallic = 0.0

    renderer.scene.add_geometry("mesh", mesh_copy, mat)

    # Camera
    if camera_params is not None:
        intrinsic = camera_params.intrinsic.intrinsic_matrix
        extrinsic = camera_params.extrinsic
        renderer.setup_camera(intrinsic, extrinsic, width, height)
    else:
        center = mesh_copy.get_center()
        eye = center + np.array([0, 0, 2.0])
        renderer.scene.camera.look_at(center, eye, [0, 1, 0])

    rgb_img = np.asarray(renderer.render_to_image(), dtype=np.uint8)
    depth_f = np.asarray(renderer.render_to_depth_image())

    dmin, dmax = float(depth_f.min()), float(depth_f.max())
    if dmax > dmin:
        depth_img = ((depth_f - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)
    else:
        depth_img = np.zeros_like(depth_f, dtype=np.uint8)

    renderer.scene.clear_geometry()
    del renderer

    return rgb_img, depth_img, (dmin, dmax)


def render_mesh_geometry(
    mesh,
    width=512,
    height=512,
    camera_params=None,
    background_color=(1.0, 1.0, 1.0, 1.0),
    base_color=(0.7, 0.7, 0.7, 1.0),
):
    """
    Render a non-textured mesh with plain shaded geometry, similar to draw_geometries([mesh]).
    Returns: (rgb_uint8 HxWx3, depth_uint8 HxW, (dmin, dmax))
    """
    mesh_copy = o3d.geometry.TriangleMesh(mesh)

    if not mesh_copy.has_vertex_normals():
        mesh_copy.compute_vertex_normals()

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(list(background_color))

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = list(base_color)
    mat.base_roughness = 1.0
    mat.base_reflectance = 0.0
    mat.base_metallic = 0.0

    if mesh_copy.has_vertex_colors():
        mat.shader = "defaultLit"

    renderer.scene.add_geometry("mesh", mesh_copy, mat)

    if camera_params is not None:
        intrinsic = camera_params.intrinsic.intrinsic_matrix
        extrinsic = camera_params.extrinsic
        renderer.setup_camera(intrinsic, extrinsic, width, height)
    else:
        bbox = mesh_copy.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = max(float(np.max(bbox.get_extent())), 1e-3)
        eye = center + np.array([0.0, 0.0, 2.5 * extent])
        renderer.scene.camera.look_at(center, eye, [0.0, 1.0, 0.0])

    rgb_img = np.asarray(renderer.render_to_image(), dtype=np.uint8)
    depth_f = np.asarray(renderer.render_to_depth_image())

    finite = np.isfinite(depth_f)
    if np.any(finite):
        dmin = float(depth_f[finite].min())
        dmax = float(depth_f[finite].max())
    else:
        dmin = 0.0
        dmax = 0.0

    if dmax > dmin:
        depth_norm = np.zeros_like(depth_f, dtype=np.float32)
        depth_norm[finite] = (depth_f[finite] - dmin) / (dmax - dmin)
        depth_img = (depth_norm * 255.0).astype(np.uint8)
    else:
        depth_img = np.zeros((height, width), dtype=np.uint8)

    renderer.scene.clear_geometry()
    del renderer

    return rgb_img, depth_img, (dmin, dmax)


def render_mesh_auto(mesh, width=512, height=512, camera_params=None, enable_lighting=True):
    """
    Render a mesh using textures when available, otherwise fall back to plain shaded geometry.
    """
    has_textures = hasattr(mesh, "textures") and len(mesh.textures) > 0
    has_uvs = mesh.has_triangle_uvs()
    if has_textures and has_uvs:
        return render_mesh_texture(
            mesh,
            width=width,
            height=height,
            camera_params=camera_params,
            enable_lighting=enable_lighting,
        )
    return render_mesh_geometry(
        mesh,
        width=width,
        height=height,
        camera_params=camera_params,
    )


def _make_single_material_submesh(mesh, triangle_indices, material_id):
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    submesh = o3d.geometry.TriangleMesh()
    submesh.vertices = o3d.utility.Vector3dVector(vertices.copy())
    submesh.triangles = o3d.utility.Vector3iVector(triangles[triangle_indices].copy())

    if mesh.has_vertex_normals():
        submesh.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals).copy())
    else:
        submesh.compute_vertex_normals()

    if mesh.has_vertex_colors():
        submesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors).copy())

    if mesh.has_triangle_uvs():
        triangle_uvs = np.asarray(mesh.triangle_uvs).reshape(-1, 3, 2)
        submesh.triangle_uvs = o3d.utility.Vector2dVector(
            triangle_uvs[triangle_indices].reshape(-1, 2).copy()
        )

    if hasattr(mesh, "textures") and len(mesh.textures) > material_id:
        tex_np = np.asarray(mesh.textures[material_id])
        if tex_np is not None and tex_np.size > 0:
            submesh.textures = [o3d.geometry.Image(np.ascontiguousarray(tex_np))]

    return submesh


def render_mesh_materials(
    mesh,
    width=512,
    height=512,
    camera_params=None,
    background_color=(1.0, 1.0, 1.0, 1.0),
    enable_lighting=True,
):
    """
    Render a legacy TriangleMesh with multiple textures by splitting it per material id.
    Returns: (rgb_uint8 HxWx3, depth_uint8 HxW, (dmin, dmax))
    """
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
    if not mesh_copy.has_vertex_normals():
        mesh_copy.compute_vertex_normals()

    if not mesh_copy.has_triangle_material_ids() or not mesh_copy.has_textures():
        return render_mesh_auto(
            mesh_copy,
            width=width,
            height=height,
            camera_params=camera_params,
            enable_lighting=enable_lighting,
        )

    triangle_material_ids = np.asarray(mesh_copy.triangle_material_ids).reshape(-1)
    unique_material_ids = sorted(int(mid) for mid in np.unique(triangle_material_ids) if mid >= 0)
    if not unique_material_ids:
        return render_mesh_auto(
            mesh_copy,
            width=width,
            height=height,
            camera_params=camera_params,
            enable_lighting=enable_lighting,
        )

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(list(background_color))

    for material_id in unique_material_ids:
        triangle_indices = np.flatnonzero(triangle_material_ids == material_id)
        if triangle_indices.size == 0:
            continue

        submesh = _make_single_material_submesh(mesh_copy, triangle_indices, material_id)

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit" if enable_lighting else "defaultUnlit"
        mat.base_color = [1.0, 1.0, 1.0, 1.0]

        if hasattr(submesh, "textures") and len(submesh.textures) > 0:
            tex_np = np.asarray(submesh.textures[0])
            if tex_np is not None and tex_np.size > 0:
                mat.albedo_img = o3d.geometry.Image(np.ascontiguousarray(tex_np))
        elif submesh.has_vertex_colors():
            mat.shader = "defaultLit"
        else:
            mat.base_color = [0.7, 0.7, 0.7, 1.0]

        renderer.scene.add_geometry(f"mesh_mat_{material_id}", submesh, mat)

    if camera_params is not None:
        intrinsic = camera_params.intrinsic.intrinsic_matrix
        extrinsic = camera_params.extrinsic
        renderer.setup_camera(intrinsic, extrinsic, width, height)
    else:
        bbox = mesh_copy.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = max(float(np.max(bbox.get_extent())), 1e-3)
        eye = center + np.array([0.0, 0.0, 2.5 * extent])
        renderer.scene.camera.look_at(center, eye, [0.0, 1.0, 0.0])

    rgb_img = np.asarray(renderer.render_to_image(), dtype=np.uint8)
    depth_f = np.asarray(renderer.render_to_depth_image())

    finite = np.isfinite(depth_f)
    if np.any(finite):
        dmin = float(depth_f[finite].min())
        dmax = float(depth_f[finite].max())
    else:
        dmin = 0.0
        dmax = 0.0

    if dmax > dmin:
        depth_norm = np.zeros_like(depth_f, dtype=np.float32)
        depth_norm[finite] = (depth_f[finite] - dmin) / (dmax - dmin)
        depth_img = (depth_norm * 255.0).astype(np.uint8)
    else:
        depth_img = np.zeros((height, width), dtype=np.uint8)

    renderer.scene.clear_geometry()
    del renderer

    return rgb_img, depth_img, (dmin, dmax)


def render_triangle_model(
    model_or_path,
    width=512,
    height=512,
    camera_params=None,
    background_color=(1.0, 1.0, 1.0, 1.0),
):
    """
    Render a multi-material OBJ/FBX/GLTF using Open3D's TriangleMeshModel path.

    This is the correct path for meshes with multiple materials / texture maps,
    since ``read_triangle_mesh()`` + a single MaterialRecord only uses one material.
    Returns: (rgb_uint8 HxWx3, depth_uint8 HxW, (dmin, dmax))
    """
    if isinstance(model_or_path, (str, os.PathLike)):
        model = o3d.io.read_triangle_model(str(model_or_path), print_progress=False)
    else:
        model = model_or_path

    if len(model.meshes) == 0:
        raise ValueError("TriangleMeshModel is empty")

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(list(background_color))
    renderer.scene.add_model("model", model)

    if camera_params is not None:
        intrinsic = camera_params.intrinsic.intrinsic_matrix
        extrinsic = camera_params.extrinsic
        renderer.setup_camera(intrinsic, extrinsic, width, height)
    else:
        all_vertices = []
        for mesh_info in model.meshes:
            vertices = np.asarray(mesh_info.mesh.vertices)
            if vertices.size:
                all_vertices.append(vertices)
        if not all_vertices:
            raise ValueError("TriangleMeshModel contains no vertices")
        vertices = np.concatenate(all_vertices, axis=0)
        center = vertices.mean(axis=0)
        extent = max(float(np.max(vertices.max(axis=0) - vertices.min(axis=0))), 1e-3)
        eye = center + np.array([0.0, 0.0, 2.5 * extent])
        renderer.scene.camera.look_at(center, eye, [0.0, 1.0, 0.0])

    rgb_img = np.asarray(renderer.render_to_image(), dtype=np.uint8)
    depth_f = np.asarray(renderer.render_to_depth_image())

    finite = np.isfinite(depth_f)
    if np.any(finite):
        dmin = float(depth_f[finite].min())
        dmax = float(depth_f[finite].max())
    else:
        dmin = 0.0
        dmax = 0.0

    if dmax > dmin:
        depth_norm = np.zeros_like(depth_f, dtype=np.float32)
        depth_norm[finite] = (depth_f[finite] - dmin) / (dmax - dmin)
        depth_img = (depth_norm * 255.0).astype(np.uint8)
    else:
        depth_img = np.zeros((height, width), dtype=np.uint8)

    renderer.scene.clear_geometry()
    del renderer

    return rgb_img, depth_img, (dmin, dmax)


_RENDER_FRAME_PATTERN = re.compile(r"(?P<frame>\d+)_v_(?P<view>\d+)\.(?:jpe?g|png)$", re.IGNORECASE)


def index_render_sequence(render_dir):
    """Index a flattened render folder named like ``00_v_03.jpg``."""
    records = []
    for name in os.listdir(render_dir):
        match = _RENDER_FRAME_PATTERN.fullmatch(name)
        if match is None:
            continue
        frame_id = int(match.group("frame"))
        view_id = int(match.group("view"))
        records.append((frame_id, view_id, os.path.join(render_dir, name)))

    if not records:
        raise ValueError(f"No frame-view renders found in {render_dir}")

    frame_ids = sorted({frame_id for frame_id, _, _ in records})
    view_ids = sorted({view_id for _, view_id, _ in records})
    frame_to_pos = {frame_id: idx for idx, frame_id in enumerate(frame_ids)}
    view_to_pos = {view_id: idx for idx, view_id in enumerate(view_ids)}

    paths_by_view = {view_id: [None] * len(frame_ids) for view_id in view_ids}
    flat_paths = [None] * (len(frame_ids) * len(view_ids))

    for frame_id, view_id, path in records:
        frame_pos = frame_to_pos[frame_id]
        view_pos = view_to_pos[view_id]
        paths_by_view[view_id][frame_pos] = path
        flat_paths[frame_pos * len(view_ids) + view_pos] = path

    missing = [
        (frame_ids[frame_pos], view_id)
        for view_id, view_paths in paths_by_view.items()
        for frame_pos, path in enumerate(view_paths)
        if path is None
    ]
    if missing:
        raise ValueError(f"Missing rendered images for frame/view pairs: {missing[:5]}")

    return {
        "frame_ids": frame_ids,
        "view_ids": view_ids,
        "frame_to_pos": frame_to_pos,
        "view_to_pos": view_to_pos,
        "paths_by_view": paths_by_view,
        "flat_paths": flat_paths,
        "num_frames": len(frame_ids),
        "num_views": len(view_ids),
    }


def _load_rgb_image(path, downscale):
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if downscale != 1.0:
        image_rgb = cv2.resize(
            image_rgb,
            dsize=None,
            fx=downscale,
            fy=downscale,
            interpolation=cv2.INTER_AREA,
        )
    return image_rgb


def _extract_largest_component(mask, min_area):
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask, dtype=bool)

    best_label = 0
    best_area = 0
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= min_area and area > best_area:
            best_label = label_idx
            best_area = area

    if best_label == 0:
        return np.zeros_like(mask, dtype=bool)
    return labels == best_label


def _pick_positive_points(mask, count):
    if count <= 0:
        return []

    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    work = dist.copy()
    if not np.any(work > 0):
        return []

    radius = max(8, int(np.sqrt(mask.sum() / max(count, 1)) * 0.35))
    points = []
    for _ in range(count):
        y, x = np.unravel_index(np.argmax(work), work.shape)
        if work[y, x] <= 0:
            break
        points.append([int(x), int(y)])
        cv2.circle(work, (int(x), int(y)), radius, 0, thickness=-1)

    return points


def _pick_negative_points(mask, count):
    if count <= 0 or not np.any(mask):
        return []

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    height, width = mask.shape
    pad = max(12, int(0.15 * max(y1 - y0 + 1, x1 - x0 + 1)))

    candidates = [
        [max(0, x0 - pad), max(0, y0 - pad)],
        [min(width - 1, x1 + pad), max(0, y0 - pad)],
        [max(0, x0 - pad), min(height - 1, y1 + pad)],
        [min(width - 1, x1 + pad), min(height - 1, y1 + pad)],
        [max(0, x0 - pad), (y0 + y1) // 2],
        [min(width - 1, x1 + pad), (y0 + y1) // 2],
    ]

    negatives = []
    for x, y in candidates:
        if not mask[y, x]:
            negatives.append([int(x), int(y)])
        if len(negatives) >= count:
            break
    return negatives


def _analyze_view_motion(
    frame_paths,
    downscale=0.5,
    white_threshold=245,
    motion_percentile=96,
    min_component_area=96,
):
    if len(frame_paths) < 2:
        raise ValueError("Need at least two frames per view to estimate motion")

    kernel = np.ones((3, 3), dtype=np.uint8)
    first_rgb = _load_rgb_image(frame_paths[0], downscale)
    prev_gray = cv2.cvtColor(first_rgb, cv2.COLOR_RGB2GRAY)
    prev_valid = np.any(first_rgb < white_threshold, axis=2)

    heatmap = np.zeros_like(prev_gray, dtype=np.float32)
    best_mask = np.zeros_like(prev_gray, dtype=bool)
    best_score = 0.0
    best_frame_pos = 0

    for frame_pos, path in enumerate(frame_paths[1:], start=1):
        curr_rgb = _load_rgb_image(path, downscale)
        curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)
        curr_valid = np.any(curr_rgb < white_threshold, axis=2)

        diff = cv2.absdiff(curr_gray, prev_gray).astype(np.float32)
        valid = prev_valid | curr_valid
        diff[~valid] = 0.0
        diff = cv2.GaussianBlur(diff, (0, 0), 1.2)
        heatmap += diff

        valid_diff = diff[valid]
        if valid_diff.size == 0:
            prev_gray = curr_gray
            prev_valid = curr_valid
            continue

        threshold = max(8.0, float(np.percentile(valid_diff, motion_percentile)))
        motion_mask = diff >= threshold
        motion_mask = cv2.morphologyEx(motion_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel).astype(bool)
        component_mask = _extract_largest_component(motion_mask, min_area=min_component_area)

        if np.any(component_mask):
            component_score = float(diff[component_mask].sum())
            if component_score > best_score:
                best_score = component_score
                best_mask = component_mask
                best_frame_pos = frame_pos

        prev_gray = curr_gray
        prev_valid = curr_valid

    return {
        "best_mask": best_mask,
        "best_score": best_score,
        "best_frame_pos": best_frame_pos,
        "heatmap": heatmap,
        "image_shape": best_mask.shape,
    }


def auto_select_sam3_prompt(
    render_dir,
    prompt_frame_pos=0,
    num_positive_points=3,
    num_negative_points=2,
    downscale=0.5,
    white_threshold=245,
    motion_percentile=96,
    min_component_area=96,
):
    """
    Automatically propose SAM3 point prompts from temporal motion in one reference view.
    """
    render_index = index_render_sequence(render_dir)
    view_analyses = {}
    best_view_id = None
    best_view_score = -1.0

    for view_id, frame_paths in render_index["paths_by_view"].items():
        analysis = _analyze_view_motion(
            frame_paths,
            downscale=downscale,
            white_threshold=white_threshold,
            motion_percentile=motion_percentile,
            min_component_area=min_component_area,
        )
        view_analyses[view_id] = analysis
        if analysis["best_score"] > best_view_score:
            best_view_id = view_id
            best_view_score = analysis["best_score"]

    if best_view_id is None or best_view_score <= 0:
        raise ValueError("Failed to find a dynamic region from temporal motion")

    best_analysis = view_analyses[best_view_id]
    prompt_frame_pos = int(np.clip(prompt_frame_pos, 0, render_index["num_frames"] - 1))
    prompt_mask = best_analysis["best_mask"].copy()
    if not np.any(prompt_mask):
        raise ValueError(f"No usable motion proposal found for reference view {best_view_id}")

    reference_frame_path = render_index["paths_by_view"][best_view_id][prompt_frame_pos]
    reference_rgb = _load_rgb_image(reference_frame_path, downscale=1.0)
    reference_height, reference_width = reference_rgb.shape[:2]
    prompt_rgb = _load_rgb_image(reference_frame_path, downscale=downscale)
    prompt_foreground = np.any(prompt_rgb < white_threshold, axis=2)
    prompt_mask = prompt_mask & prompt_foreground
    if not np.any(prompt_mask):
        prompt_mask = best_analysis["best_mask"]

    prompt_height, prompt_width = prompt_mask.shape
    scale_x = reference_width / prompt_width
    scale_y = reference_height / prompt_height

    positive_points = _pick_positive_points(prompt_mask, num_positive_points)
    negative_points = _pick_negative_points(prompt_mask, num_negative_points)
    all_points = positive_points + negative_points
    all_labels = [1] * len(positive_points) + [0] * len(negative_points)

    if not all_points:
        raise ValueError("Automatic prompt generation produced no points")

    points_abs = np.array(
        [
            [
                int(np.clip(round(x * scale_x), 0, reference_width - 1)),
                int(np.clip(round(y * scale_y), 0, reference_height - 1)),
            ]
            for x, y in all_points
        ],
        dtype=np.int32,
    )
    labels = np.array(all_labels, dtype=np.int32)
    flat_frame_index = prompt_frame_pos * render_index["num_views"] + render_index["view_to_pos"][best_view_id]

    return {
        "reference_view_id": best_view_id,
        "analysis_frame_pos": best_analysis["best_frame_pos"],
        "prompt_frame_pos": prompt_frame_pos,
        "frame_id": render_index["frame_ids"][prompt_frame_pos],
        "flat_frame_index": flat_frame_index,
        "points_abs": points_abs,
        "labels": labels,
        "prompt_mask": prompt_mask,
        "heatmap": best_analysis["heatmap"],
        "reference_frame_path": reference_frame_path,
        "view_scores": {view_id: analysis["best_score"] for view_id, analysis in view_analyses.items()},
        "render_index": render_index,
    }

def get_prim_id_image(mesh_legacy: o3d.geometry.TriangleMesh,
                      camera_params: o3d.camera.PinholeCameraParameters,
                      width: int, height: int):
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)

    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tmesh)

    K = camera_params.intrinsic.intrinsic_matrix.astype(np.float64)
    E = camera_params.extrinsic.astype(np.float64)

    Kt = o3d.core.Tensor(K, dtype=o3d.core.Dtype.Float64)
    Et = o3d.core.Tensor(E, dtype=o3d.core.Dtype.Float64)

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=Kt,
        extrinsic_matrix=Et,
        width_px=width,
        height_px=height
    )

    ans = scene.cast_rays(rays)
    prim_id = ans["primitive_ids"].numpy().astype(np.int32)
    t_hit = ans["t_hit"].numpy().astype(np.float32)

    return prim_id, t_hit



def vote_faces_from_mask(prim_id: np.ndarray, mask: np.ndarray, num_faces: int):
    face_ids = prim_id[mask]
    face_ids = face_ids[face_ids >= 0]
    votes = np.bincount(face_ids, minlength=num_faces).astype(np.int32)

    visible_ids = prim_id[prim_id >= 0]
    visible = np.bincount(visible_ids, minlength=num_faces).astype(np.int32)

    return votes, visible


def fuse_views_to_face_mask(mesh, camera_params_list, sam_masks_list, width, height, tau=0.6):
    num_faces = np.asarray(mesh.triangles).shape[0]
    total_votes = np.zeros(num_faces, dtype=np.int64)
    total_vis = np.zeros(num_faces, dtype=np.int64)

    for cam, mask in zip(camera_params_list, sam_masks_list):
        prim_id, _ = get_prim_id_image(mesh, cam, width, height)
        votes, vis = vote_faces_from_mask(prim_id, mask.astype(bool), num_faces)
        total_votes += votes
        total_vis += vis

    ratio = total_votes / np.maximum(total_vis, 1)

    face_is_dynamic = ratio >= tau
    face_is_unseen = total_vis == 0
    face_is_low_vis = total_vis < 4   # tune this

    return face_is_dynamic, ratio, total_vis, face_is_unseen, face_is_low_vis


def load_masks_for_mesh_frame(mask_root, mesh_frame_id, num_views=16):
    frame_dir = os.path.join(mask_root, f"frame_{mesh_frame_id:04d}")
    masks = []
    for view_id in range(num_views):
        masks.append(np.load(os.path.join(frame_dir, f"view_{view_id:02d}.npy")).astype(bool))
    return masks



def split_mesh_by_face_mask(mesh: o3d.geometry.TriangleMesh, face_mask: np.ndarray):
    tris = np.asarray(mesh.triangles)
    verts = np.asarray(mesh.vertices)

    dyn_tris = tris[face_mask]
    stat_tris = tris[~face_mask]

    dyn = o3d.geometry.TriangleMesh()
    dyn.vertices = o3d.utility.Vector3dVector(verts)
    dyn.triangles = o3d.utility.Vector3iVector(dyn_tris)
    dyn.remove_unreferenced_vertices()
    dyn.remove_degenerate_triangles()
    dyn.remove_duplicated_triangles()
    dyn.remove_duplicated_vertices()
    dyn.compute_vertex_normals()

    stat = o3d.geometry.TriangleMesh()
    stat.vertices = o3d.utility.Vector3dVector(verts)
    stat.triangles = o3d.utility.Vector3iVector(stat_tris)
    stat.remove_unreferenced_vertices()
    stat.remove_degenerate_triangles()
    stat.remove_duplicated_triangles()
    stat.remove_duplicated_vertices()
    stat.compute_vertex_normals()

    return dyn, stat

def build_face_adjacency(mesh):
    tris = np.asarray(mesh.triangles)
    edge_to_faces = {}

    for fi, tri in enumerate(tris):
        edges = [
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[2], tri[0]))),
        ]

        for e in edges:
            edge_to_faces.setdefault(e, []).append(fi)

    neighbors = [[] for _ in range(len(tris))]

    for faces in edge_to_faces.values():
        if len(faces) == 2:
            a, b = faces
            neighbors[a].append(b)
            neighbors[b].append(a)

    return neighbors


def dilate_dynamic_on_mesh(
    face_is_dynamic,
    neighbors,
    expandable_mask=None,
    num_iters=2
):
    """
    Expand dynamic labels over the mesh graph.

    Args:
        face_is_dynamic: bool array, initial dynamic face mask.
        neighbors: face adjacency list.
        expandable_mask: bool array. If provided, only these faces can be newly added.
                         Useful for only expanding into unseen or low-visible faces.
        num_iters: number of adjacency expansion steps.

    Returns:
        refined dynamic mask.
    """
    refined = face_is_dynamic.copy()

    for _ in range(num_iters):
        new_refined = refined.copy()

        dynamic_ids = np.where(refined)[0]

        for f in dynamic_ids:
            for nb in neighbors[f]:
                if refined[nb]:
                    continue

                if expandable_mask is not None and not expandable_mask[nb]:
                    continue

                new_refined[nb] = True

        refined = new_refined

    return refined

def connected_components_faces(face_ids, neighbors):
    face_set = set(face_ids)
    visited = set()
    components = []

    for f in face_ids:
        if f in visited:
            continue

        comp = []
        queue = deque([f])
        visited.add(f)

        while queue:
            cur = queue.popleft()
            comp.append(cur)

            for nb in neighbors[cur]:
                if nb in face_set and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        components.append(comp)

    return components

def remove_small_static_islands(face_is_dynamic, neighbors, max_static_component_size=100):
    """
    Move small disconnected static components into the dynamic region.
    """
    refined = face_is_dynamic.copy()

    static_ids = np.where(~refined)[0]
    comps = connected_components_faces(static_ids, neighbors)

    for comp in comps:
        if len(comp) <= max_static_component_size:
            refined[comp] = True

    return refined


def build_local_band_around_dynamic(face_is_dynamic, neighbors, band_iters=2):
    """
    Return a bool mask of faces that are close to the initial dynamic region.

    band_iters=1 means direct neighbors of dynamic faces.
    band_iters=2 means neighbors of neighbors.
    """
    band = face_is_dynamic.copy()
    frontier = set(np.where(face_is_dynamic)[0].tolist())

    for _ in range(band_iters):
        new_frontier = set()

        for f in frontier:
            for nb in neighbors[f]:
                if not band[nb]:
                    band[nb] = True
                    new_frontier.add(nb)

        frontier = new_frontier

        if not frontier:
            break

    # only candidate faces, excluding already dynamic ones
    candidate_band = band & (~face_is_dynamic)

    return candidate_band

def local_expand_dynamic(
    face_is_dynamic,
    neighbors,
    candidate_band,
    expandable_mask,
    min_dynamic_neighbors=2,
    num_iters=1
):
    """
    Conservative local expansion.
    Only faces inside candidate_band can be changed.
    """
    refined = face_is_dynamic.copy()

    for _ in range(num_iters):
        new_refined = refined.copy()

        candidates = np.where((~refined) & candidate_band & expandable_mask)[0]

        for f in candidates:
            dyn_count = sum(refined[nb] for nb in neighbors[f])

            if dyn_count >= min_dynamic_neighbors:
                new_refined[f] = True

        refined = new_refined

    return refined


def keep_only_components_touching_initial_dynamic(
    refined_dynamic,
    initial_dynamic,
    neighbors,
    min_touch_faces=1
):
    """
    Keep dynamic components only if they touch the initial dynamic region.
    This removes unconnected floor/background pieces added during refinement.
    """
    visited = np.zeros(len(refined_dynamic), dtype=bool)
    final_dynamic = np.zeros_like(refined_dynamic, dtype=bool)

    dynamic_ids = np.where(refined_dynamic)[0]

    for start in dynamic_ids:
        if visited[start]:
            continue

        queue = deque([start])
        visited[start] = True
        comp = []

        while queue:
            f = queue.popleft()
            comp.append(f)

            for nb in neighbors[f]:
                if refined_dynamic[nb] and not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

        comp = np.array(comp, dtype=np.int64)

        # Keep this dynamic component only if it contains/touches initial dynamic faces.
        touch_count = np.sum(initial_dynamic[comp])

        if touch_count >= min_touch_faces:
            final_dynamic[comp] = True

    return final_dynamic

def save_sam_masks_multi_frames(outputs_all, save_root, num_frames=None, num_views=16,
                                H=1080, W=1920, instance_id=0):


    os.makedirs(save_root, exist_ok=True)

    total = len(outputs_all)
    if total % num_views != 0:
        raise ValueError(f"len(outputs_all)={total} not divisible by num_views={num_views}")

    inferred_frames = total // num_views
    if num_frames is None:
        num_frames = inferred_frames
    else:
        if num_frames != inferred_frames:
            raise ValueError(f"num_frames={num_frames} but inferred {inferred_frames} from total/num_views")

    for frame_id in range(num_frames):
        frame_dir = os.path.join(save_root, f"frame_{frame_id:04d}")
        os.makedirs(frame_dir, exist_ok=True)

        for view_id in range(num_views):
            k = frame_id * num_views + view_id
            frame_dict = outputs_all[k]

            if not frame_dict:
                mask = np.zeros((H, W), dtype=bool)
            else:
                if instance_id in frame_dict:
                    mask = frame_dict[instance_id].astype(bool)
                else:

                    mask = frame_dict[next(iter(frame_dict))].astype(bool)

            np.save(os.path.join(frame_dir, f"view_{view_id:02d}.npy"), mask)

    print(f"Saved masks to: {save_root}")
    print(f"Frames: {num_frames}, Views/frame: {num_views}, Total saved: {num_frames * num_views}")

def remove_small_dynamic_components(face_is_dynamic, neighbors, min_component_faces=200):
    """
    Move small disconnected dynamic components back to static.

    Args:
        face_is_dynamic: bool array, True = dynamic face.
        neighbors: face adjacency list from build_face_adjacency(mesh).
        min_component_faces: dynamic components with fewer faces than this
                             will be moved back to static.

    Returns:
        refined bool mask.
    """
    refined = face_is_dynamic.copy()
    visited = np.zeros(len(face_is_dynamic), dtype=bool)

    dynamic_ids = np.where(face_is_dynamic)[0]

    for start in dynamic_ids:
        if visited[start]:
            continue

        queue = deque([start])
        visited[start] = True
        comp = []

        while queue:
            f = queue.popleft()
            comp.append(f)

            for nb in neighbors[f]:
                if face_is_dynamic[nb] and not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

        if len(comp) < min_component_faces:
            refined[comp] = False

    return refined