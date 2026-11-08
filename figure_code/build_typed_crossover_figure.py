"""Draw how agent parts are exchanged and their connections checked."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
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

    label(ax, 148, 36, "Parent agents", 23, bold=True)
    workflow(89, 93, "round", "round")
    workflow(89, 163, "square", "square")

    label(ax, 425, 36, "Exchange parts", 23, bold=True)
    curve(ax, (251, 93), [((296, 93), (322, 97), (354, 126))], edge="#a28bb4")
    curve(ax, (251, 163), [((296, 163), (322, 159), (354, 130))], edge="#71a59a")
    arrow(ax, (379, 115), (473, 115), PLUM, bend=-0.20)
    arrow(ax, (473, 141), (379, 141), TEAL, bend=-0.20)
    curve(ax, (499, 128), [((535, 128), (536, 94), (568, 93))], edge="#78918e")
    curve(ax, (499, 128), [((535, 128), (536, 162), (568, 163))], edge="#78918e")
    arrow(ax, (567, 93), (581, 93), "#78918e")
    arrow(ax, (567, 163), (581, 163), "#78918e")

    label(ax, 800, 36, "Check connections", 23, bold=True)
    workflow(620, 93, "round", "round")
    workflow(620, 163, "round", "square")
    circle(ax, 815, 93, 12, "#e4f2e9", TEAL, width=1)
    stroke(ax, [(809, 93), (813, 97), (821, 89)], TEAL, width=1.3, z=6)
    circle(ax, 815, 163, 12, "#f8e9e8", PINK, width=1)
    stroke(ax, [(811, 159), (819, 167)], PINK, width=1.25, z=6)
    stroke(ax, [(811, 167), (819, 159)], PINK, width=1.25, z=6)
    label(ax, 838, 93, "Types match", 19, color=TEAL, bold=True, align="left")
    label(ax, 838, 163, "Types differ", 19, color=PINK, bold=True, align="left")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", type=Path, default=ROOT / "figures" / "typed_crossover.png")
    parser.add_argument("--pdf", type=Path, help="Also write a vector PDF.")
    args = parser.parse_args()

    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "text.usetex": False}):
        fig = population()
        fig.savefig(args.png, dpi=320)
        if args.pdf:
            fig.savefig(args.pdf, metadata={"Title": "Typed crossover"})
        plt.close(fig)


if __name__ == "__main__":
    main()
