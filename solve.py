import json
import matplotlib.pyplot as plt
from side import Side
from utils import *
# from rdp import rdp
# from getsides import getsides

# getsides()
print("g")
# Load data
with open("pieces_data.json", "r") as f:
    data = json.load(f)
print("l")

all_sides = []
for piece in data:
    print("in data?")
    piece_num = piece["piece_number"]
    for s in piece["sides"]:
        side = Side(s["side_points"], s["side_index"], piece_number=piece_num)
        side.adjust_angle()
        print(side.type)
        all_sides.append(side)


for i, side_a in enumerate(all_sides):
    for j, side_b in enumerate(all_sides):
        if i >= j:
            continue  # avoid duplicates & self-pairs
        if side_a.piece_number == side_b.piece_number:
            continue  # skip same piece

        score, bpoints = matching_score(side_a, side_b)

        types_different = side_a.type != side_b.type
        lengths_close = abs(side_a.length - side_b.length) < 7
        score_good = score < 30
        not_flat = side_a.type != 'flat' and side_b.type != 'flat'
        is_match = types_different and lengths_close and score_good and not_flat

        status = " MATCH" if is_match else " NO MATCH"
        print(f"{status} | A: {side_a.piece_number}-{side_a.side_index}  ↔  B: {side_b.piece_number}-{side_b.side_index}"
              f" | Score: {score:.2f} | Types: {side_a.type, side_b.type} | Diff_lengths: {abs(side_a.length - side_b.length):.2f}")

        if is_match:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            fig.suptitle(f"MATCH: {side_a.piece_number}-{side_a.side_index} ↔ {side_b.piece_number}-{side_b.side_index}", fontsize=12)

            # Plot side A
            a_pts = side_a.normalized_points
            axes[1].plot(*zip(*a_pts), color='red', marker='o')
            axes[1].set_title(f"A: Piece {side_a.piece_number} Side {side_a.side_index}")
            axes[1].axis("equal")
            axes[1].invert_yaxis()



            b_pts = side_b.normalized_points
            axes[1].plot(*zip(*b_pts), color='green', marker='o')

            axes[1].set_title(f"B: Piece {side_b.piece_number} Side {side_b.side_index}")
            axes[1].axis("equal")
            axes[1].invert_yaxis()


