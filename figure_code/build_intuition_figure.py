"""Draw the plan-once overview and a separate typed-crossover schematic."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path as DrawPath


ROOT = Path(__file__).resolve().parents[1]
INK = "#263244"
MUTED = "#506471"
PLUM = "#67418d"
TEAL = "#176e5f"
PINK = "#ad5051"


def canvas(height):
    fig = plt.figure(figsize=(12, height / 90))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1080), ylim=(height, 0), aspect="equal")
    ax.set_axis_off()
    return fig, ax


def label(ax, x, y, text, size=19, *, color=INK, bold=False, align="center"):
    ax.text(x, y, text, fontsize=size * 0.8, color=color,
            weight="bold" if bold else "normal", ha=align, va="center", zorder=8)


def stroke(ax, points, color, *, width=1.45, style="-", z=4):
    x, y = zip(*points)
    ax.plot(x, y, color=color, lw=width, ls=style, solid_capstyle="round", zorder=z)


def curve(ax, start, segments, *, edge="none", fill="none", width=1.45, close=False):
    vertices = [start]
    codes = [DrawPath.MOVETO]
    for segment in segments:
        vertices.extend(segment)
        codes.extend([DrawPath.CURVE4] * 3)
    if close:
        vertices.append(start)
        codes.append(DrawPath.CLOSEPOLY)
    ax.add_patch(PathPatch(DrawPath(vertices, codes), facecolor=fill, edgecolor=edge,
                           lw=width, capstyle="round", joinstyle="round", zorder=2))


def arrow(ax, start, end, color, *, bend=0, width=1.4, style="-"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9.5,
                                connectionstyle=f"arc3,rad={bend}", shrinkA=0, shrinkB=0,
                                color=color, lw=width, linestyle=style, zorder=5))


def rounded(ax, x, y, width, height, fill, edge="none", *, radius=5, lw=1.25):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                                boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor=fill, edgecolor=edge, lw=lw, zorder=3))


def circle(ax, x, y, radius, fill, edge="none", *, width=1.25):
    ax.add_patch(Circle((x, y), radius, facecolor=fill, edgecolor=edge, lw=width, zorder=5))


def apple(ax, x, y):
    curve(ax, (x, y - 13), [
        ((x - 15, y - 24), (x - 29, y - 8), (x - 22, y + 9)),
        ((x - 18, y + 23), (x - 9, y + 26), (x, y + 20)),
        ((x + 9, y + 26), (x + 18, y + 23), (x + 22, y + 9)),
        ((x + 29, y - 8), (x + 15, y - 24), (x, y - 13)),
    ], fill="#f4bcb4", edge=PINK, close=True)
    curve(ax, (x, y - 14), [((x - 4, y - 23), (x + 3, y - 32), (x + 17, y - 29)),
                           ((x + 16, y - 18), (x + 6, y - 15), (x, y - 14))],
          fill="#a6cfae", edge=TEAL, width=1.15, close=True)
    curve(ax, (x - 13, y - 7), [((x - 16, y - 3), (x - 16, y + 1), (x - 14, y + 4))],
          edge="#fff7f2", width=1.75)


def droplet(ax, x, y):
    curve(ax, (x, y - 28), [
        ((x - 6, y - 16), (x - 23, y - 2), (x - 23, y + 8)),
        ((x - 23, y + 32), (x + 23, y + 32), (x + 23, y + 8)),
        ((x + 23, y - 2), (x + 6, y - 16), (x, y - 28)),
    ], fill="#bce1eb", edge="#38798a", close=True)
    curve(ax, (x - 14, y + 7), [((x - 14, y + 14), (x - 11, y + 18), (x - 6, y + 19))],
          edge="#f5fcff", width=1.7)
    for dx, dy, size in ((32, -12, 5), (-32, 1, 3.5), (32, 9, 3)):
        circle(ax, x + dx, y + dy, size, "#d2edf4", "#68a2ac", width=0.85)


def fridge(ax, x, y):
    rounded(ax, x - 21, y - 30, 42, 61, "#dcebf7", "#466f94", radius=6)
    stroke(ax, [(x - 20, y - 7), (x + 20, y - 7)], "#466f94", width=1.2)
    stroke(ax, [(x - 11, y - 20), (x - 11, y - 13)], "#466f94")
    stroke(ax, [(x - 11, y + 1), (x - 11, y + 13)], "#466f94")
    stroke(ax, [(x - 14, y + 31), (x - 14, y + 35)], "#466f94")
    stroke(ax, [(x + 14, y + 31), (x + 14, y + 35)], "#466f94")
    circle(ax, x + 10, y + 11, 4, "#f1bd93", "#ac6953", width=0.75)


def plan_note(ax):
    curve(ax, (40, 169), [
        ((35, 113), (62, 96), (114, 97)),
        ((179, 89), (206, 117), (205, 165)),
        ((204, 214), (170, 249), (111, 244)),
        ((65, 247), (37, 213), (40, 169)),
    ], fill="#f0e8f8", close=True)
    points = [(86, 119), (145, 112), (164, 127), (174, 211), (97, 221)]
    ax.add_patch(Polygon([(x + 5, y + 5) for x, y in points],
                         facecolor="#dfcdee", edgecolor="none", zorder=2))
    ax.add_patch(Polygon(points, facecolor="#fffcff", edgecolor=PLUM,
                         lw=1.45, joinstyle="round", zorder=3))
    stroke(ax, [(145, 112), (147, 130), (164, 127)], PLUM, width=1.2)
    for x, y, length in ((104, 153, 39), (107, 174, 35), (109, 195, 28)):
        stroke(ax, [(x, y), (x + 4, y + 4), (x + 10, y - 4)], TEAL, width=1.2)
        stroke(ax, [(x + 15, y - 1), (x + 15 + length, y - 5)], "#9878b4", width=1.15)
    stroke(ax, [(63, 133), (63, 148)], "#a788be")
    stroke(ax, [(56, 140), (70, 140)], "#a788be")


def executor_call(ax, x):
    circle(ax, x, 132, 24, "#dbefe5")
    rounded(ax, x - 15, 123, 30, 18, "#f9fffc", TEAL, radius=5, lw=1.15)
    for dx in (-6, 0, 6):
        circle(ax, x + dx, 132, 1.55, TEAL)


def overview():
    fig, ax = canvas(304)
    label(ax, 121, 38, "Opus", 30, color=PLUM, bold=True)
    label(ax, 121, 69, "Plan once", 20, color=PLUM)
    plan_note(ax)

    label(ax, 489, 38, "GPT-sol", 30, color=TEAL, bold=True)
    label(ax, 489, 69, "Act each turn", 20, color=TEAL)
    curve(ax, (174, 150), [((214, 148), (208, 98), (255, 98)),
                          ((370, 98), (575, 98), (666, 98))], edge="#a285bd", width=1.4)
    label(ax, 264, 84, "Fixed plan", 15.5, color=PLUM, align="left")
    curve(ax, (259, 227), [
        ((281, 192), (380, 199), (476, 199)),
        ((574, 198), (669, 186), (719, 217)),
        ((761, 257), (664, 270), (493, 268)),
        ((345, 269), (248, 273), (259, 227)),
    ], fill="#e9f5ed", close=True)

    for x, scene, action in ((311, apple, "Find"), (489, droplet, "Clean"), (666, fridge, "Store")):
        arrow(ax, (x, 98), (x, 108), "#a285bd", width=1.2)
        executor_call(ax, x)
        arrow(ax, (x - 9, 155), (x - 9, 186), TEAL, bend=0.15)
        arrow(ax, (x + 9, 186), (x + 9, 155), "#608b9d", bend=0.15, width=1.15)
        scene(ax, x, 226)
        label(ax, x, 280, action, 18, color=TEAL)
    label(ax, 293, 172, "Action", 14.5, color=TEAL, align="right")
    label(ax, 331, 172, "Feedback", 14.5, color="#496f80", align="left")
    arrow(ax, (352, 227), (445, 227), "#609585", style=(0, (2, 3)))
    arrow(ax, (532, 227), (624, 227), "#609585", style=(0, (2, 3)))

    # Compare executor substitutions with the Opus planner held fixed.
    costs = []
    for executor in ("opus", "sol"):
        path = ROOT / "data" / "matrix_cells" / f"n250_opus_{executor}.json"
        costs.append(json.loads(path.read_text(encoding="utf-8"))["CoP"])

    stroke(ax, [(766, 38), (766, 271)], "#d5e1dd", width=1)
    label(ax, 801, 49, "Cost per success", 23, bold=True, align="left")
    for y, name, cost, color, fill in (
        (100, "Opus executor", costs[0], PLUM, "#a78aba"),
        (154, "GPT-sol executor", costs[1], TEAL, "#419483"),
    ):
        label(ax, 801, y, name, 17, color=color, align="left")
        label(ax, 1047, y, f"${cost:.4f}", 17, color=color, bold=True, align="right")
        rounded(ax, 801, y + 15, 246 * cost / max(costs), 12, fill, radius=4)
    reduction = 1 - costs[1] / costs[0]
    label(ax, 801, 226, f"{reduction:.0%} lower", 30, color=TEAL, bold=True, align="left")
    label(ax, 801, 253, "Same Opus planner", 17, color=MUTED, align="left")
    return fig


def population():
    fig, ax = canvas(230)

    def module(x, y, kind, slot):
        color, fill = (PLUM, "#efe4f7") if kind == "round" else (TEAL, "#dfefe5")
        rounded(ax, x - 30, y - 17, 60, 34, fill, color, radius=7)
        for offset in (-4, 4):
            stroke(ax, [(x - 10, y + offset), (x + 10, y + offset)], color, width=1)
        px = x + slot * 31
        if kind == "round":
            circle(ax, px, y, 6, "white", color)
        else:
            rounded(ax, px - 5.5, y - 5.5, 11, 11, "white", color, radius=0.8)

    def workflow(x, y, left, right):
        color = PINK if left != right else MUTED
        style = (0, (2, 3)) if left != right else "-"
        stroke(ax, [(x + 37, y), (x + 81, y)], color, style=style)
        module(x, y, left, 1)
        module(x + 118, y, right, -1)

    label(ax, 148, 36, "Parent workflows", 23, bold=True)
    workflow(89, 93, "round", "round")
    workflow(89, 163, "square", "square")

    label(ax, 425, 36, "Swap subgraphs", 23, bold=True)
    curve(ax, (251, 93), [((296, 93), (322, 97), (354, 126))], edge="#a28bb4")
    curve(ax, (251, 163), [((296, 163), (322, 159), (354, 130))], edge="#71a59a")
    arrow(ax, (379, 115), (473, 115), PLUM, bend=-0.20)
    arrow(ax, (473, 141), (379, 141), TEAL, bend=-0.20)
    curve(ax, (499, 128), [((535, 128), (536, 94), (568, 93))], edge="#78918e")
    curve(ax, (499, 128), [((535, 128), (536, 162), (568, 163))], edge="#78918e")
    arrow(ax, (567, 93), (581, 93), "#78918e")
    arrow(ax, (567, 163), (581, 163), "#78918e")

    label(ax, 800, 36, "Offspring type check", 23, bold=True)
    workflow(620, 93, "round", "round")
    workflow(620, 163, "round", "square")
    circle(ax, 815, 93, 12, "#e4f2e9", TEAL, width=1)
    stroke(ax, [(809, 93), (813, 97), (821, 89)], TEAL, width=1.3, z=6)
    circle(ax, 815, 163, 12, "#f8e9e8", PINK, width=1)
    stroke(ax, [(811, 159), (819, 167)], PINK, width=1.25, z=6)
    stroke(ax, [(811, 167), (819, 159)], PINK, width=1.25, z=6)
    label(ax, 838, 93, "Matching types", 19, color=TEAL, bold=True, align="left")
    label(ax, 838, 163, "Type mismatch", 19, color=PINK, bold=True, align="left")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", type=Path, default=ROOT / "figures" / "intuition.png")
    parser.add_argument("--pdf", type=Path, help="Also write the overview as a vector PDF.")
    parser.add_argument("--population-png", type=Path,
                        default=ROOT / "figures" / "typed_crossover.png")
    parser.add_argument("--population-pdf", type=Path,
                        help="Also write the typed-crossover schematic as a vector PDF.")
    args = parser.parse_args()

    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "text.usetex": False}):
        for draw, png, pdf, title in (
            (overview, args.png, args.pdf, "Plan once, act each turn"),
            (population, args.population_png, args.population_pdf, "Typed crossover"),
        ):
            fig = draw()
            fig.savefig(png, dpi=320)
            if pdf:
                fig.savefig(pdf, metadata={"Title": title})
            plt.close(fig)


if __name__ == "__main__":
    main()
