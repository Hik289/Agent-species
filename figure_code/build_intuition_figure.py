"""Draw the Agent Species overview from the released ALFWorld records."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
INK = "#202c40"
MUTED = "#516179"
LINE = "#a4b5c8"
VIOLET = "#6748a6"
TEAL = "#127768"
BLUE = "#2d61a3"
AMBER = "#936126"


def make_figure():
    fig = plt.figure(figsize=(12, 4.2))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1080), ylim=(405, 0))
    ax.set_axis_off()

    def label(x, y, text, *, size=17, color=INK, weight="normal", align="left"):
        ax.text(x, y, text, ha=align, va="center", fontsize=size * 0.8,
                weight=weight, color=color, zorder=6)

    def line(points, color=LINE, width=1, style="-"):
        x, y = zip(*points)
        ax.plot(x, y, color=color, lw=width, ls=style, solid_capstyle="round", zorder=3)

    def arrow(start, end, color=BLUE, width=1.25):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", shrinkA=0, shrinkB=0,
                                     mutation_scale=9, lw=width, color=color, zorder=4))

    def box(x, y, w, h, fill, edge, radius=7, width=0.9, z=2):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle=f"round,pad=0,rounding_size={radius}",
                                    facecolor=fill, edgecolor=edge, lw=width, zorder=z))

    def dot(x, y, radius, fill, edge, width=1, z=5):
        ax.add_patch(Circle((x, y), radius, facecolor=fill, edgecolor=edge, lw=width, zorder=z))

    def polygon(points, fill, edge):
        ax.add_patch(Polygon(points, facecolor=fill, edgecolor=edge, lw=1.2,
                             joinstyle="round", zorder=4))

    def panel(x, y, w, h, letter, title, fill, edge, accent):
        box(x, y, w, h, fill, edge, radius=12, width=1.1)
        box(x + 13, y + 11, 24, 24, accent, accent, radius=5)
        label(x + 25, y + 23, letter, size=16, color="white", weight="bold", align="center")
        label(x + 46, y + 23, title, size=19, weight="bold")

    def document(x, y, color, fill, *, strategy=False):
        polygon([(x - 15, y - 20), (x + 7, y - 20), (x + 15, y - 12),
                 (x + 15, y + 20), (x - 15, y + 20)], fill, color)
        ax.plot([x + 7, x + 7, x + 15], [y - 20, y - 12, y - 12],
                lw=0.9, color=color, zorder=5)
        for offset, length in ((-6, 15), (1, 15), (8, 10)):
            ax.plot([x - 4, x - 4 + length], [y + offset] * 2, color=color,
                    lw=1, solid_capstyle="round", zorder=5)
            if strategy:
                dot(x - 9, y + offset, 1.35, color, color, width=0)

    def model(x, y, color, fill):
        box(x - 20, y - 16, 40, 32, fill, color, radius=6, width=1.3)
        line([(x, y - 16), (x, y - 23)], color)
        dot(x, y - 24, 2.2, color, color, width=0)
        dot(x - 7, y - 3, 2.4, color, color, width=0)
        dot(x + 7, y - 3, 2.4, color, color, width=0)
        line([(x - 6, y + 7), (x + 6, y + 7)], color, 1.2)

    def household(x, y):
        polygon([(x - 24, y - 6), (x, y - 25), (x + 24, y - 6),
                 (x + 19, y - 6), (x + 19, y + 17), (x - 19, y + 17),
                 (x - 19, y - 6)], "#e0f2e8", TEAL)
        ax.add_patch(Rectangle((x - 5, y + 2), 10, 15, facecolor="white",
                               edgecolor=TEAL, lw=1.0, zorder=5))
        dot(x + 2, y + 9, 1, TEAL, TEAL, width=0)

    panel(9, 9, 265, 271, "A", "Swap model roles", "#f5f2fc", "#c3b4e3", VIOLET)
    panel(290, 9, 458, 271, "B", "Plan once, execute with feedback",
          "#f0f6fc", "#afc9e5", BLUE)
    panel(764, 9, 307, 271, "C", "Matched evaluation",
          "#f1f8f4", "#accdbb", TEAL)
    arrow((270, 146), (294, 146), VIOLET)
    arrow((744, 146), (768, 146), BLUE)

    # A. The selected cells are a readable slice of the 4-by-4 experiment.
    label(141, 67, "Executor", weight="bold")
    label(134, 94, "Opus", size=15.5, align="center", color=VIOLET, weight="bold")
    label(211, 94, "GPT-sol", size=15.5, align="center", color=TEAL, weight="bold")
    label(25, 105, "Planner", size=16, weight="bold")
    label(25, 138, "Opus", size=16, color=VIOLET)
    label(25, 195, "GPT-sol", size=16, color=TEAL)
    for y, planner_color in ((138, VIOLET), (195, TEAL)):
        for x, executor_color in ((134, VIOLET), (211, TEAL)):
            highlighted = y == 138 and x == 211
            box(x - 32, y - 22, 64, 44, "#e1f3ea" if highlighted else "white",
                TEAL if highlighted else "#c9bde0", radius=6, width=1.5 if highlighted else 0.85)
            arrow((x - 8, y), (x + 7, y), MUTED, 1)
            dot(x - 16, y, 6.5, planner_color, planner_color, width=0)
            dot(x + 16, y, 6.5, executor_color, executor_color, width=0)
    label(141, 250, "4 of 16 role assignments", color=MUTED, size=16, align="center")

    # B. The environment result feeds both the next turn and the recorded history.
    document(336, 93, BLUE, "white")
    model(426, 93, VIOLET, "#ede5fa")
    document(518, 93, VIOLET, "#f1eafb", strategy=True)
    arrow((356, 93), (400, 93))
    arrow((452, 93), (498, 93), VIOLET)
    label(336, 126, "Task input", align="center", size=16)
    label(426, 126, "Planner", align="center", size=16, weight="bold")
    label(518, 126, "Strategy", align="center", size=16, weight="bold", color=VIOLET)
    arrow((518, 137), (518, 152), VIOLET)
    box(600, 77, 132, 31, "#eae2f7", "#cbbbdf", radius=6)
    label(666, 93, "Once at reset", size=15.5, color=VIOLET, weight="bold", align="center")

    model(518, 181, TEAL, "#e1f1ed")
    household(680, 181)
    arrow((546, 169), (649, 169))
    arrow((649, 194), (546, 194), TEAL)
    label(598, 154, "Action", size=15.5, color=BLUE, align="center")
    label(605, 207, "Observation", size=15.5, color=TEAL, align="center")
    label(518, 219, "Executor", size=17, weight="bold", align="center")
    label(680, 219, "ALFWorld", size=17, weight="bold", align="center")

    line([(680, 230), (680, 250), (361, 250), (361, 181), (487, 181)],
         TEAL, 1.25, (0, (4, 3)))
    arrow((473, 181), (491, 181), TEAL)
    label(378, 166, "Each turn", size=15, color=TEAL, weight="bold")
    box(404, 236, 234, 28, "#e0f2ec", "#a8cdbc", radius=6, z=4)
    label(521, 250, "Recent action–outcome history", size=14.5, color=TEAL, align="center")

    # C. Position the points with values from the same 250-task records.
    def load(path):
        return json.loads((ROOT / "data" / path).read_text(encoding="utf-8"))

    baseline = load("final_audit.json")["raw_checks"]["n250_single-opus"]
    mixed = load("matrix_cells/n250_opus_sol.json")
    homogeneous = load("matrix_cells/n250_opus_opus.json")

    def plot_x(cost):
        return 822 + (cost - 0.03) / 0.27 * 219

    def plot_y(rate):
        return 223 - (100 * rate - 84) / 13 * 126

    label(779, 60, "Selected results · 250 held-out tasks", size=15.5, color=MUTED)
    label(779, 82, "Success (%)", size=15.5, weight="bold")
    for rate in (85, 90, 95):
        y = plot_y(rate / 100)
        line([(822, y), (1041, y)], "#c8dbd0", width=0.7)
        label(813, y, str(rate), size=14, color=MUTED, align="right")
    line([(822, 96), (822, 223), (1041, 223)], "#79958b", 0.95)
    for cost in (0.05, 0.15, 0.25):
        x = plot_x(cost)
        line([(x, 223), (x, 228)], "#79958b", 0.95)
        label(x, 238, f"{cost:.2f}", size=14, color=MUTED, align="center")
    label(931, 259, "USD per success", size=15.5, weight="bold", align="center")
    for row, cost_key, name, color, align, dx, dy in (
        (baseline, "CoP_from_vectors", "Single Opus", MUTED, "right", -11, 0),
        (mixed, "CoP", "Opus → GPT-sol", TEAL, "left", 11, -8),
        (homogeneous, "CoP", "Opus → Opus", VIOLET, "right", -11, -7),
    ):
        x, y = plot_x(row[cost_key]), plot_y(row["SR"])
        dot(x, y, 5.9, color, "white", width=1.1)
        label(x + dx, y + dy, name, size=14.5, color=color, weight="bold", align=align)

    # D and E use independent tasks and experimental protocols.
    panel(9, 295, 525, 101, "D", "Development search",
          "#fbf7ee", "#e0c99c", AMBER)
    panel(550, 295, 521, 101, "E", "Typed offspring assay",
          "#f5f2fa", "#ccbce1", VIOLET)
    for i, fill in enumerate(("#f1e1ba", "#f7ebcd", "white")):
        box(29 + i * 5, 346 + i * 5, 28, 24, fill, AMBER, radius=3)
    label(74, 367, "Topology, models, harnesses", size=15.5)
    arrow((303, 367), (332, 367), AMBER)
    label(343, 359, "Rank on 8 tasks", size=15.5, weight="bold")
    label(343, 378, "By success and cost", size=14.5, color=MUTED)

    label(566, 366, "Exchange subgraphs", size=16.5)
    arrow((743, 366), (769, 366), VIOLET)
    dot(793, 366, 7, "#dff2e8", TEAL, width=1.1)
    line([(801, 366), (816, 366)], TEAL, 1.05)
    dot(824, 366, 7, "#dff2e8", TEAL, width=1.1)
    label(839, 366, "Pass", size=16, color=TEAL, weight="bold")
    dot(920, 366, 7, "#fce9e6", "#a95048", width=1.1)
    line([(928, 366), (943, 366)], "#a95048", 1.05, (0, (2, 2)))
    ax.add_patch(Rectangle((944, 359), 14, 14, facecolor="#fce9e6",
                           edgecolor="#a95048", lw=1.1, zorder=5))
    label(969, 366, "Mismatch", size=16, color="#a95048", weight="bold")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", type=Path, default=ROOT / "figures" / "intuition.png")
    parser.add_argument("--pdf", type=Path, help="Also write a vector PDF.")
    args = parser.parse_args()

    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "text.usetex": False}):
        fig = make_figure()
        fig.savefig(args.png, dpi=320)
        if args.pdf:
            fig.savefig(args.pdf, metadata={"Title": "Agent Species overview"})
        plt.close(fig)


if __name__ == "__main__":
    main()
