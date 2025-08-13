from scipy.spatial.distance import directed_hausdorff
from PIL import Image
import numpy as np
from rdp import rdp
import cv2
from shapely.geometry import Polygon, LineString
import shapely
from scipy.interpolate import splprep, splev
from itertools import combinations
from collections import defaultdict


image = Image.open("IMG_9561.jpg")
def find_array_with_points(sides, pts):
    pt1 = np.array(pts[0])
    pt2 = np.array(pts[1])
    for arr in sides:
        has_pt1 = np.any(np.all(arr == pt1, axis=1))
        has_pt2 = np.any(np.all(arr == pt2, axis=1))
        if has_pt1 and has_pt2:
            return arr
    raise ValueError("No array contains both points.")


def binarize_image(img, thresh=100, erode_iterations=2, kernel_size=3):
    img = img.convert('L')
    img_np = np.array(img)
    binarized_np = (img_np > thresh).astype(np.uint8) * 255

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_np = cv2.erode(binarized_np, kernel, iterations=erode_iterations)

    return Image.fromarray(eroded_np), eroded_np // 255


def get_blobs(binarized_np, min_area=50000, draw_on=None):
    binary = (binarized_np * 255).astype(np.uint8) if binarized_np.max() <= 1 else binarized_np.copy()
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        # Skip small areas
        if cv2.contourArea(contour) < min_area:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        boxes.append(box)

        if draw_on is not None:
            cv2.drawContours(draw_on, [box], 0, (0, 255, 0), 2)

    print(f"Detected {len(boxes)} rotated blobs.")
    return boxes


# noinspection PyTypeChecker
def get_edge(binarized_np, boxes, sample_every=5):
    geometries = []

    for box in boxes:
        mask = np.zeros_like(binarized_np, dtype=np.uint8)
        cv2.drawContours(mask, [box], 0, color=255, thickness=-1)

        blob = cv2.bitwise_and(binarized_np, mask)

        cnts, _ = cv2.findContours(blob, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            largest = largest.reshape(-1, 2)
            sampled = largest[::sample_every]

            if len(sampled) >= 3:
                poly = Polygon(sampled)
                print(f"Box → Contour points: {len(sampled)} | Valid: {poly.is_valid} | Area: {poly.area:.2f}")
                if not poly.is_valid:
                    print("⚠ Invalid polygon — attempting fix...")
                    poly = poly.buffer(0)

                if poly.is_valid:
                    geometries.append(poly)
            elif len(sampled) >= 2:
                geometries.append(LineString(sampled))

    return geometries


def get_centroid(polygon):
    return shapely.centroid(polygon)


def get_polygon_size(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny
    return height, width


def label_box_corners(points):

    points = np.array(points)

    sorted_by_y = points[np.argsort(points[:, 1])]

    top = sorted_by_y[2:]
    bottom = sorted_by_y[:2]

    top_left, top_right = top[np.argsort(top[:, 0])]
    bottom_left, bottom_right = bottom[np.argsort(bottom[:, 0])]

    return {
        1: tuple(top_left),
        2: tuple(bottom_left),
        3: tuple(bottom_right),
        4: tuple(top_right)
    }

def get_side_corners(combo):
    side_ends = [[combo[1], combo[2]], [combo[2], combo[3]], [combo[3], combo[4]], [combo[4], combo[1]]]
    return side_ends
# noinspection PyTypeChecker
def detect_polygon_corners_by_rdp(polygon, ax2=None, epsilon=125, std_tol=0.2, min_length=550, ax=None):
    coords = np.array(polygon.exterior.coords[:-1])  # Remove duplicate end point
    simplified = rdp(coords, epsilon=epsilon)
    candidate_corners = list(enumerate(simplified))
    if ax:
        for idx, pt in candidate_corners:
            ax.plot(pt[0], pt[1], 'ro', markersize=3)

    if len(candidate_corners) < 4:
        return [], candidate_corners, {}

    best_combo = None
    lowest_std = float('inf')

    for combo in combinations(candidate_corners, 4):
        points = [pt for _, pt in combo]
        sorted_points = sorted(
            points, key=lambda p: np.arctan2(p[1] - polygon.centroid.y, p[0] - polygon.centroid.x)
        )
        # ✅ Side length check
        side_lengths = [
            np.linalg.norm(sorted_points[i] - sorted_points[(i + 1) % 4])
            for i in range(4)
        ]

        if any(length < min_length for length in side_lengths):
            continue
        std_dev = np.std(side_lengths) / np.mean(side_lengths)
        if std_dev < std_tol and std_dev < lowest_std:
            best_combo = sorted_points
            lowest_std = std_dev

    if best_combo is None:
        return [], candidate_corners, {}
    rdp_indices = [np.argmin(np.linalg.norm(coords - corner, axis=1)) for corner in best_combo]
    sorted_pairs = sorted(zip(rdp_indices, best_combo), key=lambda x: x[0])
    ordered_indices = [idx for idx, _ in sorted_pairs]

    side_points = []
    for i in range(4):
        start = ordered_indices[i]
        end = ordered_indices[(i + 1) % 4]

        if start < end:
            segment = coords[start:end + 1]
        else:
            segment = np.concatenate((coords[start:], coords[:end + 1]), axis=0)

        side_points.append(np.array(segment))

    point_to_sides = defaultdict(list)
    for side_idx, side in enumerate(side_points):
        for pt in map(tuple, side):
            point_to_sides[pt].append(side_idx)

    for pt, sides in point_to_sides.items():
        if len(sides) > 1:
            if ax2:
                ax2.plot(pt[0], pt[1], 'yx', markersize=10)

    print(f"Detected {len(side_points)} sides")
    labeled = label_box_corners(best_combo)
    side_index_dict = {}
    for j, i in enumerate(get_side_corners(labeled)):
        side_index_dict[j] = find_array_with_points(side_points, i)
    print("SIDE INDEX DICTIONARY:", side_index_dict)
    return side_points, candidate_corners, side_index_dict


def classify_side_shape(side):
    side.adjust_angle()
    side.side_points = normalized(side)
    start = np.array(side.side_points[0])

    is_horizontal = side.side_index in (1, 3)
    baseline_value = start[1] if is_horizontal else start[0]  # y for horizontal, x for vertical

    deviations = []
    for p in side.side_points[1:-1]:
        coord = p[1] if is_horizontal else p[0]
        deviations.append(coord - baseline_value)
    deviations = np.array(deviations)

    max_dev = np.max(deviations)
    max_abs_dev = max_dev

    # Debug print, remove later if desired
    print(f"Side {side.side_index}: max_abs_dev={max_abs_dev}")

    if max_abs_dev < 50:
        return 'flat'

    # Determine inward/outward by side and deviation sign
    if side.side_index == 0:  # left vertical side
        return 'inward' if max_dev > 0 else 'outward'
    elif side.side_index == 1:  # bottom horizontal side
        return 'outward' if max_dev < 0 else 'inward'
    elif side.side_index == 2:  # right vertical side
        return 'outward' if max_dev > 0 else 'inward'
    elif side.side_index == 3:  # top horizontal side
        return 'inward' if max_dev < 0 else 'outward'



# noinspection PyTypeChecker,PyTupleAssignmentBalance
def normalized(side, num_points=50):
    points = np.array(side.side_points)

    if len(points) <= 3:
        print(f"⚠️ Not enough points to interpolate (got {len(points)} points). Skipping normalization.")
        return points

    try:
        x, y = points[:, 0], points[:, 1]
        tck, _ = splprep([x, y], s=0)
        u = np.linspace(0, 1, num_points)
        x_i, y_i = splev(u, tck)
        curve = np.stack([x_i, y_i], axis=1)

        current_angle = side.angle
        rotation_deg = -current_angle
        rotated_curve = rotate_points(curve, origin=side.p1, angle_deg=rotation_deg)

        rotated_curve[:, 0] -= rotated_curve[0, 0]

        rotated_curve[:, 1] -= np.mean(rotated_curve[:, 1])

        return rotated_curve

    except Exception as e:
        print(f"❌ splprep failed on side: {e}")
        return points


# noinspection PyTypeChecker,PyTupleAssignmentBalance
def norm_align(side, num_points=50):
    side.adjust_angle()
    points = np.array(side.side_points)

    if len(points) <= 3:
        print(f"⚠️ Not enough points to interpolate (got {len(points)} points). Skipping alignment.")
        return points

    try:
        x, y = points[:, 0], points[:, 1]
        tck, _ = splprep([x, y], s=0)
        u = np.linspace(0, 1, num_points)
        x_i, y_i = splev(u, tck)
        curve = np.stack([x_i, y_i], axis=1)

        # Determine dominant direction
        dx = abs(side.p2[0] - side.p1[0])
        dy = abs(side.p2[1] - side.p1[1])

        if dy > dx:
            # More vertical → align y to start at 0, center x
            curve[:, 1] -= curve[0, 1]
            curve[:, 0] -= np.mean(curve[:, 0])
        else:
            # More horizontal → align x to start at 0, center y
            curve[:, 0] -= curve[0, 0]
            curve[:, 1] -= np.mean(curve[:, 1])

        return curve

    except Exception as e:
        print(f"❌ splprep failed on side: {e}")
        return points


def matching_score(side_a, side_b):
    norm_a = normalized(side_a)
    norm_b = normalized(side_b)

    ref_point = np.array(side_a.p1)
    norm_a += ref_point - norm_a[0]  # shift side_a's curve so p1 aligns to real p1

    variations = {
        "original": norm_b,
        "flip_x": norm_b * np.array([-1, 1]),
        "flip_y": norm_b * np.array([1, -1]),
        "flip_xy": norm_b * np.array([-1, -1]),
    }

    best_score = float("inf")
    best_points = norm_b

    for mode, variant in variations.items():
        variant_shifted = variant - variant[0] + ref_point

        forward = directed_hausdorff(norm_a, variant_shifted)[0]
        backward = directed_hausdorff(variant_shifted, norm_a)[0]
        score = max(forward, backward)

        # print(f"Score ({mode}): {score:.2f}")
        if score < best_score:
            best_score = score
            best_points = variant_shifted.copy()
    """print("A p1:", norm_a[0][0])
    print("B p1:", best_points[0][0])
    print("A type:", side_a.type)
    print("B type:", side_b.type)
    print(f"✅ Best match: {best_mode} with score {best_score:.2f}")"""

    return best_score, best_points


def sanity_check_polygon(polygon):
    if not polygon.is_valid:
        print("Invalid polygon geometry!")
        return False
    if polygon.area == 0:
        print("Zero area polygon!")
        return False
    if len(polygon.exterior.coords) < 4:
        print("Too few points for a valid shape!")
        return False
    return True


def side_to_dict(p1, p2, side_index, side_points):
    return {
        "p1": p1.tolist(),
        "p2": p2.tolist(),
        "side_index": side_index,
        "points": list(tuple(i) for i in side_points.tolist()),
    }


# used some AI to explain this one
def rotate_points(points, origin, angle_deg):
    angle_rad = np.radians(angle_deg)
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    return (points - origin) @ rot_matrix.T + origin


# noinspection PyPep8Naming
def rotate_points_to_nearest_right_angle(points):
    # Use PCA to find the main axis direction
    centered = points - np.mean(points, axis=0)
    u, s, vh = np.linalg.svd(centered)
    direction = vh[0]  # principal component direction

    angle = np.degrees(np.arctan2(direction[1], direction[0]))
    target = min([0, 90, 180, -90], key=lambda x: abs((x - angle + 180) % 360 - 180))
    theta = np.radians((target - angle + 180) % 360 - 180)

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    return centered @ R.T + np.mean(points, axis=0)


def split_list(lst, chunk_size=4):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def orient_sides_from_corners(corners):
    """
    Given 4 corners (unsorted), return ordered sides as a list of point arrays.
    Side 1 is always the right side.
    """

    corners = np.array(corners)

    # Step 1: Sort corners by y to get top/bottom
    sorted_by_y = corners[np.argsort(corners[:, 1])]
    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]

    # Step 2: Sort left/right within each
    top_left, top_right = top_two[np.argsort(top_two[:, 0])]
    bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:, 0])]

    # Now assemble corners in clockwise order starting from top-left
    ordered_corners = [top_left, top_right, bottom_right, bottom_left]

    # Step 3: Extract 4 sides from consecutive corners
    sides = []
    for i in range(4):
        p1 = ordered_corners[i]
        p2 = ordered_corners[(i + 1) % 4]
        side = np.array([p1, p2])
        sides.append(side)

    return sides  # [top, right, bottom, left] clockwise
