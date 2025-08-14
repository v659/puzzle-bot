from utils import *
import matplotlib
matplotlib.use("TkAgg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from side import Side
# === Run pipeline ===
from shapely.geometry import Polygon, MultiPolygon, LineString

def plot_geometries(geometries):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_title("Visualized Geometries")

    for geom in geometries:
        if isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            ax.plot(x, y, 'b')
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, 'g')
        else:
            print(f"⚠️ Skipping unknown geometry type: {type(geom)}")

    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
def getsides():
    import json
    piece_count = 0  # Label for each piece
    binarized, img_np = binarize_image(image)

    # === Show binary image ===
    fig_bin, ax_bin = plt.subplots()
    ax_bin.set_title("Binarized Image")
    ax_bin.imshow(binarized, cmap='gray')
    ax_bin.axis('off')
    fig_bin.tight_layout()
    fig_bin.savefig("binarized_image.png")

    img_draw = np.array(image.convert("RGB"))  # For visualization
    boxes = get_blobs(img_np, draw_on=img_draw)
    geometries = get_edge(img_np, boxes, sample_every=5)
    print(f"Extracted {len(geometries)} geometries")
    # === Plotting ===
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax.set_aspect('equal')
    ax2.set_aspect('equal')
    ax.set_title("Piece Shapes with Labels")
    ax2.set_title("Detected Sides with Index Labels")
    fig3, ax3 = plt.subplots()
    ax3.set_title("All Rotated Sides")
    ax3.set_aspect('equal')
    ax3.grid(True)  # <-- Add this line to show grid
    ax3.legend(fontsize=6)

    sides_list = []
    sides_list_copy = []
    # Clear data file
    with open("pieces_data.json", "w") as f:
        json.dump([], f)
    with open("types_data.json", "w") as f:
        json.dump([], f)
    for geom in geometries:

        if isinstance(geom, Polygon):
            polygons = [geom]
        elif isinstance(geom, MultiPolygon):
            polygons = list(geom.geoms)
        elif isinstance(geom, LineString):
            coords = list(geom.coords)
            x_vals = [pt[0] for pt in coords]
            y_vals = [pt[1] for pt in coords]
            ax.plot(x_vals, y_vals, color='gray', linewidth=1, linestyle='--')
            continue
        else:
            print(f"⚠️ Skipping unknown geometry type: {type(geom)}")
            continue

        for poly in polygons:
            piece_count += 1
            # Draw the outline of the piece
            x, y = poly.exterior.xy
            ax.plot(x, y, color='black', linewidth=2)

            # Label piece number at its centroid
            centroid = poly.centroid
            ax.text(centroid.x, centroid.y, str(piece_count), fontsize=10, fontweight='bold',
                    color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

            # Detect sides

            sides, candidate_corners, side_index_dict = detect_polygon_corners_by_rdp(poly, ax2)
            # Plot corners
            for idx, pt in candidate_corners:
                ax.plot(pt[0], pt[1], 'ro', markersize=4)
                ax2.plot(pt[0], pt[1], 'ro', markersize=4)

            if not sides:
                print("❌ No valid quad for this polygon")
                continue
            cmap = plt.get_cmap('tab20')
            for i in side_index_dict:
                side_arr = np.array(side_index_dict[i])
                side_arr_copy = rotate_points_to_nearest_right_angle(side_arr)
                if len(side_arr) < 2:
                    print(f"️ Skipping side {i} with too few points")
                    continue

                side_obj = Side(list(tuple(i) for i in side_arr.tolist()), i)
                side_obj_copy = Side(list(tuple(i) for i in side_arr_copy.tolist()), i)
                sides_list_copy.append(side_obj_copy)
                sides_list.append(side_obj)

                # Color based on index
                color = cmap(i % 20)

                ax2.plot(side_arr[:, 0], side_arr[:, 1], color=color, linewidth=2)

                # === Label side with type ===
                mid_idx = len(side_arr) // 2
                mid_pt = side_arr[mid_idx]
                label = str(side_obj.side_index)  # 'F', 'I', 'O'

                ax2.text(mid_pt[0], mid_pt[1], label, fontsize=8, color='blue',
                         ha='center', va='center',
                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # === Save side metadata
    sides_list = split_list(sides_list)
    figures = []
    axes = []
    with open("pieces_data.json", "r") as f:
        data = json.load(f)
    with open("types_data.json", "r") as f:
        data2 = json.load(f)
    for j, i in enumerate(sides_list_copy):
        side_arr2 = np.array(i.side_points)
        cmap = plt.get_cmap('tab20')
        color = cmap(j % 20)
        ax3.plot(side_arr2[:, 0], side_arr2[:, 1], label=f"Side {j} - Type {classify_side_shape(i)}, Index {i.side_index}", color=color)
        mid_pt = side_arr2[len(side_arr2) // 2]
        ax3.text(mid_pt[0], mid_pt[1], str(i.side_index), fontsize=8)

        data2.append(classify_side_shape(i))

    ax3.set_title("All Rotated Sides")
    ax3.set_aspect('equal')
    ax3.legend(fontsize=6)
    fig3.tight_layout()

    with open("types_data.json", "w") as f:
        json.dump(data2, f, indent=4)
    for i, piece_sides in enumerate(sides_list):
        piece_data = {
            "piece_number": int(i),
            "sides": []
        }

        for r in piece_sides:
            ax2.plot(r.x, r.y, 'ko')  # optional: show a dot at center
            ax2.text(r.x, r.y, str(r.side_index), fontsize=10, color='red')
            side_data = {
                "side_points": [list(map(float, pt)) for pt in r.side_points],
                "side_index": int(r.side_index),
                "side_length": float(r.length),
                "side_angle": float(r.angle),
                "side_normalized": [list(map(float, pt)) for pt in r.normalized_points] if r.normalized_points is not None
 else [],
                "p1": list(map(float, r.p1)),
                "p2": list(map(float, r.p2)),
                "side_id": r.side_id,
                "side_data": str(r)
            }
            piece_data["sides"].append(side_data)

        data.append(piece_data)

    with open("pieces_data.json", "w") as f:
        json.dump(data, f, indent=4)

    # === Show and save plots ===
    fig.tight_layout()
    fig2.tight_layout()
    fig.gca().invert_yaxis()
    fig2.gca().invert_yaxis()
    fig3.gca().invert_yaxis()
    print('done')
    # Optional: save figures as images
    fig.savefig("piece_shapes.png")
    fig2.savefig("side_segments.png")
    # Show plots
    plt.ioff()  # interactive mode
    plt.show(block=True)


if __name__ == "__main__":
    getsides()
