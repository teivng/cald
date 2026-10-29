"""CALD, the 3b1b cut — manim scene (draft 2).

Storyboard (user-directed):
  1  the supervised problem: class callouts -> legend; X_train = context set
  2  "A PFN predicts in-context": heatmap + boundary fade in
  --- title card: Calibration ---
  3  heatmap fades; X_anchor materializes (class-coloured triangles -> dark
     yellow), X_eval materializes (class-coloured squares -> green); pan left;
     the objective ℒ(X_anchor)↓ + ℒ(X_eval)↑ appears and docks above the
     plane; heatmap returns; round 1 finetune (boundary nuked + live λ curve)
     -> first histogram bar; round 2 faster -> second bar
  4  montage to K = 50; histogram centered on the record; α-quantile threshold
  --- title card: Deployment ---
  5  unknown ID batch (green circles) -> full finetune curve -> λ* logged as a
     vertical line inside the record -> no alarm
  6  rotated (harmful) batch -> λ* far below -> histogram zooms out -> alarm

Render:   manim -qm --disable_caching scene.py CALDStory
Needs:    assets/ produced by gen_data.py first.
"""

import os
import pickle

import numpy as np
from manim import (
    Axes, Circle, Create, DashedLine, Dot, FadeIn, FadeOut, GrowFromEdge,
    ImageMobject, Line, Rectangle, ReplacementTransform, Scene, Square,
    SurroundingRectangle, Text, Triangle, VGroup, VMobject, Write,
    config, DOWN, ITALIC, LEFT, ORIGIN, RIGHT, UP, PI,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

# ---- catppuccin mocha ------------------------------------------------------
BG = "#1e1e2e"; TEXT = "#cdd6f4"; SUBTEXT = "#a6adc8"; SURFACE = "#45475a"
C0 = "#f38ba8"; C1 = "#89b4fa"; DGOLD = "#df8e1d"; GOLD = "#f9e2af"
GREEN = "#a6e3a1"; RED = "#f38ba8"; TEAL = "#94e2d5"

config.background_color = BG

D = np.load(os.path.join(ASSETS, "data.npz"))
with open(os.path.join(ASSETS, "contours.pkl"), "rb") as f:
    CONTOURS = pickle.load(f)

N_STEPS = 50
K = len(D["cal_lams"])
THRESH = float(D["threshold"])
CAL_LAMS = D["cal_lams"]
LAM_ID = float(D["dep_id_traj"][-1])
LAM_SH = float(D["dep_sh_traj"][-1])

# sparse display subsets: a handful of points reads better than the full set
RNG = np.random.default_rng(7)
SUP_IDX = RNG.choice(len(D["X_sup"]), 20, replace=False)
ANC_IDX = RNG.choice(len(D["X_anc"]), 10, replace=False)
EV_IDX = RNG.choice(64, 14, replace=False)      # eval batches display subset

# centered histogram over the calibration record (zoomed out in seq 6)
H_LO = float(CAL_LAMS.min()) - 0.012
H_HI = float(CAL_LAMS.max()) + 0.012
HIST_BINS = np.linspace(H_LO, H_HI, 13)
MAX_COUNT = int(np.histogram(CAL_LAMS, bins=HIST_BINS)[0].max())

PLANE_C = LEFT * 3.55 + DOWN * 0.25     # plane center after the pan
LOSS_C = RIGHT * 3.55 + UP * 1.75
HIST_C = RIGHT * 3.55 + DOWN * 2.0


def fps3(pts, k=3):
    """Farthest-point pick of k diverse samples."""
    sel = [int(np.argmin(pts[:, 0] + pts[:, 1]))]
    for _ in range(k - 1):
        d = np.min(
            ((pts[:, None, :] - pts[sel][None, :, :]) ** 2).sum(-1), axis=1)
        sel.append(int(np.argmax(d)))
    return pts[sel]


class CALDStory(Scene):
    # ------------------------------------------------------------ helpers
    def d2p(self, xy):
        return self.plane.c2p(xy[0], xy[1])

    def heat(self, key):
        prefix, idx = key.rsplit("_", 1)
        img = ImageMobject(os.path.join(ASSETS, f"{prefix}_{int(idx):02d}.png"))
        w = np.linalg.norm(self.plane.c2p(4, 0) - self.plane.c2p(-4, 0))
        img.width = w
        img.move_to(self.plane.c2p(0, 0))
        img.set_z_index(-2)
        return img

    def boundary(self, key):
        grp = VGroup()
        for seg in CONTOURS[key]:
            if len(seg) < 2:
                continue
            v = VMobject(stroke_color=TEXT, stroke_width=3)
            v.set_points_as_corners([self.d2p(p) for p in seg])
            grp.add(v)
        grp.set_z_index(-1)
        return grp

    def swap_field(self, key):
        new_h, new_b = self.heat(key), self.boundary(key)
        self.remove(self.heat_m, self.bound_m)
        self.add(new_h, new_b)
        self.heat_m, self.bound_m = new_h, new_b

    def set_title(self, s, color=TEXT, t2c=None):
        t = Text(s, font_size=30, color=color, t2c=t2c or {}) \
            .to_edge(UP, buff=0.25)
        if getattr(self, "title_m", None):
            self.play(ReplacementTransform(self.title_m, t), run_time=0.6)
        else:
            self.play(FadeIn(t), run_time=0.6)
        self.title_m = t

    def title_card(self, s):
        veil = Rectangle(width=config.frame_width, height=config.frame_height,
                         fill_color=BG, fill_opacity=0.94, stroke_width=0)
        veil.set_z_index(10)
        word = Text(s, font_size=64, color=TEXT).set_z_index(11)
        self.play(FadeIn(veil), run_time=0.7)
        self.play(Write(word), run_time=1.0)
        self.wait(1.1)
        self.play(FadeOut(word), FadeOut(veil), run_time=0.7)

    def msub(self, base, sub, font_size=20, color=TEXT):
        b = Text(base, font_size=font_size, slant=ITALIC, color=color)
        s = Text(sub, font_size=int(font_size * 0.62), color=color)
        s.next_to(b, RIGHT, buff=0.03, aligned_edge=DOWN)
        return VGroup(b, s)

    def loss_expr(self, sub, color, font_size=34):
        g = VGroup(
            Text("ℒ(", font_size=font_size, color=color),
            self.msub("X", sub, font_size=int(font_size * 0.9), color=color),
            Text(")", font_size=font_size, color=color),
        ).arrange(RIGHT, buff=0.07)
        return g

    # ------------------------------------------------------------- legend
    def legend_marker(self, kind):
        if kind == "dot0":
            return Dot(radius=0.06, color=C0)
        if kind == "dot1":
            return Dot(radius=0.06, color=C1)
        if kind == "tri0":
            return Triangle(color=C0, fill_color=C0, fill_opacity=0.9,
                            stroke_width=0).scale(0.1).rotate(PI)
        if kind == "tri1":
            return Triangle(color=C1, fill_color=C1, fill_opacity=0.9,
                            stroke_width=0).scale(0.1).rotate(PI)
        if kind == "tri":
            return Triangle(color=DGOLD, fill_color=DGOLD, fill_opacity=0.9,
                            stroke_width=0).scale(0.1).rotate(PI)
        if kind == "sq":
            return Square(side_length=0.17, color=GREEN, fill_color=GREEN,
                          fill_opacity=0.9, stroke_width=0)
        return Circle(radius=0.09, color=GREEN, stroke_width=3.0)   # "circ"

    def build_legend(self, grid=False):
        rows = VGroup()
        for kinds, base, sub in self.legend_spec:
            marks = VGroup(*[self.legend_marker(k) for k in kinds])
            marks.arrange(RIGHT, buff=0.07)
            rows.add(VGroup(marks, self.msub(base, sub, font_size=19))
                     .arrange(RIGHT, buff=0.14))
        if grid:
            rows.arrange_in_grid(cols=3, buff=0.35, col_alignments="lll")
        else:
            rows.arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        box = SurroundingRectangle(rows, color=SURFACE, buff=0.16,
                                   corner_radius=0.08, stroke_width=1.5,
                                   fill_color=BG, fill_opacity=0.85)
        return VGroup(box, rows)

    def relegend(self, grid=True, run_time=0.6):
        new = self.build_legend(grid=grid).move_to(LEFT * 3.55 + DOWN * 3.58)
        self.play(ReplacementTransform(self.legend, new), run_time=run_time)
        self.legend = new

    def verdict_card(self, line1, line2, color):
        txt = VGroup(
            Text(line1, font_size=30, color=TEXT),
            Text(line2, font_size=36, color=color),
        ).arrange(DOWN, buff=0.25)
        box = SurroundingRectangle(txt, color=color, corner_radius=0.12,
                                   buff=0.35, stroke_width=2.5,
                                   fill_color=BG, fill_opacity=0.93)
        card = VGroup(box, txt).move_to(ORIGIN)
        card.set_z_index(12)
        self.play(FadeIn(box), Write(txt[0]), run_time=0.8)
        self.play(Write(txt[1]), run_time=0.7)
        return card

    # ---------------------------------------------------------- construct
    def construct(self):
        self.title_m = None
        self.seq1()
        self.seq2()
        self.title_card("Calibration")
        self.seq3()
        self.seq4()
        self.title_card("Deployment")
        self.seq5()
        self.seq6()

    # ------------------------------------------------------------- seq 1
    def seq1(self):
        self.plane = Axes(
            x_range=[-4, 4, 2], y_range=[-4, 4, 2],
            x_length=6.4, y_length=6.4, tips=False,
            axis_config={"color": SURFACE, "include_ticks": False},
        ).shift(DOWN * 0.3)
        frame = SurroundingRectangle(self.plane, color=SURFACE, buff=0,
                                     stroke_width=2)
        self.play(Create(self.plane), Create(frame), run_time=1.2)
        self.frame_m = frame

        Xs, ys = D["X_sup"][SUP_IDX], D["y_sup"][SUP_IDX]
        self.dots = VGroup(*[
            Dot(self.d2p(p), radius=0.05, color=C1 if c == 1 else C0,
                fill_opacity=0.9)
            for p, c in zip(Xs, ys)
        ])
        self.play(FadeIn(self.dots, lag_ratio=0.01), run_time=1.6)

        # call out three DIVERSE points per class, then collapse to a legend
        def callout(cls, color, pos, from_right):
            sel = fps3(Xs[ys == cls])
            label = Text(f"class {cls}", font_size=26, color=color).move_to(pos)
            start = label.get_left() if from_right else label.get_right()
            lines = VGroup(*[
                Line(start, self.d2p(p), stroke_color=color,
                     stroke_width=1.6, stroke_opacity=0.8)
                for p in sel
            ])
            return label, lines

        lab0, ln0 = callout(0, C0, LEFT * 5.2 + DOWN * 0.6, from_right=False)
        lab1, ln1 = callout(1, C1, RIGHT * 5.2 + UP * 1.6, from_right=True)
        self.play(Write(lab0), Create(ln0), run_time=1.0)
        self.play(Write(lab1), Create(ln1), run_time=1.0)
        self.wait(0.5)

        # these labelled points are the context set, X_train
        ctx = VGroup(
            self.msub("X", "train", font_size=30, color=TEXT),
            Text("— the context samples", font_size=24, color=SUBTEXT),
        ).arrange(RIGHT, buff=0.2).to_edge(UP, buff=0.35)
        self.play(Write(ctx), run_time=1.0)
        self.wait(0.9)

        self.legend_spec = [(["dot0", "dot1"], "X", "train")]
        self.legend = self.build_legend().to_corner(UP + RIGHT, buff=0.3) \
            .shift(DOWN * 0.7)
        self.play(
            ReplacementTransform(VGroup(lab0, lab1, ctx), self.legend),
            FadeOut(ln0), FadeOut(ln1), run_time=1.2,
        )
        self.wait(0.5)

    # ------------------------------------------------------------- seq 2
    def seq2(self):
        self.set_title("A PFN predicts in-context")
        self.heat_m = self.heat("cal0_00")
        self.bound_m = self.boundary("cal0_0")
        self.play(FadeIn(self.heat_m), run_time=1.4)
        self.play(Create(self.bound_m), run_time=1.6)
        self.wait(1.2)

    # ------------------------------------------------------------- seq 3
    def seq3(self):
        self.set_title("Calibration")

        # heatmap steps aside while the partition is introduced
        self.play(FadeOut(self.heat_m), FadeOut(self.bound_m), run_time=0.8)

        # anchors: class-coloured triangles, then all turn dark yellow
        anc_pts = D["X_anc"][ANC_IDX]
        anc_y = D["y_anc"][ANC_IDX]
        self.anchor_m = VGroup(*[
            Triangle(color=C1 if c == 1 else C0,
                     fill_color=C1 if c == 1 else C0,
                     fill_opacity=0.95, stroke_width=0)
            .scale(0.12).rotate(PI).move_to(self.d2p(p))
            for p, c in zip(anc_pts, anc_y)
        ])
        cap_anc = VGroup(
            self.msub("X", "anchor", font_size=26, color=TEXT),
            Text("held out, in-distribution", font_size=21, color=SUBTEXT),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        cap_anc.next_to(self.legend, DOWN, buff=0.3).align_to(self.legend,
                                                              RIGHT)
        self.legend_spec.append((["tri0", "tri1"], "X", "anchor"))
        new_leg = self.build_legend().move_to(self.legend)
        self.play(FadeIn(self.anchor_m, lag_ratio=0.02), FadeIn(cap_anc),
                  ReplacementTransform(self.legend, new_leg), run_time=1.2)
        self.legend = new_leg
        self.wait(0.8)
        # -> dark yellow: the classes no longer matter, only "trusted ID"
        self.legend_spec[-1] = (["tri"], "X", "anchor")
        new_leg = self.build_legend().move_to(self.legend)
        self.play(
            self.anchor_m.animate.set_fill(DGOLD).set_color(DGOLD),
            ReplacementTransform(self.legend, new_leg),
            run_time=0.9,
        )
        self.legend = new_leg
        self.wait(0.5)
        self.play(FadeOut(cap_anc), run_time=0.4)

        # eval batch: green squares; flash their PFN pseudolabels, then back
        ev_pts = D["cal_evals"][0][EV_IDX]
        ev_ps = D["pseudo0"][EV_IDX]
        self.eval_m = VGroup(*[
            Square(side_length=0.2, color=GREEN, fill_color=GREEN,
                   fill_opacity=0.95, stroke_width=0).move_to(self.d2p(p))
            for p in ev_pts
        ])
        cap_ev = VGroup(
            self.msub("X", "eval", font_size=26, color=TEXT),
            Text("bootstrapped ID eval batch", font_size=21, color=SUBTEXT),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        cap_ev.next_to(self.legend, DOWN, buff=0.3).align_to(self.legend,
                                                             RIGHT)
        self.legend_spec.append((["sq"], "X", "eval"))
        new_leg = self.build_legend().move_to(self.legend)
        self.play(FadeIn(self.eval_m, lag_ratio=0.02), FadeIn(cap_ev),
                  ReplacementTransform(self.legend, new_leg), run_time=1.2)
        self.legend = new_leg
        self.wait(0.6)
        pseudo_cap = Text("pseudolabeled by the PFN", font_size=21,
                          color=TEXT)
        pseudo_cap.next_to(cap_ev, DOWN, buff=0.18) \
            .align_to(cap_ev, LEFT)
        self.play(
            FadeIn(pseudo_cap),
            *[sq.animate.set_fill(C1 if p == 1 else C0)
              .set_color(C1 if p == 1 else C0)
              for sq, p in zip(self.eval_m, ev_ps)],
            run_time=0.8,
        )
        self.wait(1.0)
        self.play(
            *[sq.animate.set_fill(GREEN).set_color(GREEN)
              for sq in self.eval_m],
            run_time=0.8,
        )
        self.wait(0.3)
        self.play(FadeOut(cap_ev), FadeOut(pseudo_cap), run_time=0.4)

        # pan left; legend becomes a compact strip under the plane
        plane_grp = VGroup(self.plane, self.frame_m, self.dots,
                           self.anchor_m, self.eval_m)
        legend_target = self.build_legend(grid=True) \
            .move_to(LEFT * 3.55 + DOWN * 3.58)
        self.play(
            plane_grp.animate.scale(0.85).move_to(PLANE_C),
            ReplacementTransform(self.legend, legend_target),
            run_time=1.4,
        )
        self.legend = legend_target

        # the optimization objective, revealed one piece at a time, then
        # the prefix goes away and the losses dock above the plane
        prefix = Text("Optimization objective:", font_size=27, color=TEXT)
        prefix.move_to(RIGHT * 3.55 + UP * 2.35)
        core_a = self.loss_expr("anchor", DGOLD)
        arr_a = Text("↓", font_size=34, color=DGOLD)
        core_e = self.loss_expr("eval", GREEN)
        arr_e = Text("↑", font_size=34, color=GREEN)
        obj = VGroup(
            VGroup(core_a, arr_a).arrange(RIGHT, buff=0.12),
            VGroup(core_e, arr_e).arrange(RIGHT, buff=0.12),
        ).arrange(RIGHT, buff=0.8).move_to(RIGHT * 3.55 + UP * 1.5)
        self.play(Write(prefix), run_time=0.8)
        self.play(Write(core_a), run_time=0.8)
        self.play(Write(arr_a), run_time=0.5)
        self.wait(0.3)
        self.play(Write(core_e), run_time=0.8)
        self.play(Write(arr_e), run_time=0.5)
        self.wait(0.9)
        self.play(
            FadeOut(prefix),
            obj.animate.scale(0.62).move_to(LEFT * 3.55 + UP * 2.93),
            run_time=0.9,
        )
        self.objective_m = obj

        # heatmap returns; loss axes materialize
        self.heat_m = self.heat("cal0_00")
        self.bound_m = self.boundary("cal0_0")
        self.play(FadeIn(self.heat_m), Create(self.bound_m), run_time=1.0)

        self.loss_ax = Axes(
            x_range=[0, N_STEPS, 10], y_range=[0, 0.85, 0.2],
            x_length=5.4, y_length=2.4, tips=False,
            axis_config={"color": SURFACE, "font_size": 18,
                         "include_ticks": True},
        ).move_to(LOSS_C)
        xl = Text("finetune step", font_size=18, color=SUBTEXT)
        xl.next_to(self.loss_ax, DOWN, buff=0.1)
        yl = Text("anchor loss λ", font_size=18, color=SUBTEXT)
        yl.rotate(PI / 2).next_to(self.loss_ax, LEFT, buff=0.1)
        self.loss_deco = VGroup(xl, yl)
        self.play(Create(self.loss_ax), FadeIn(self.loss_deco), run_time=1.0)

        # rounds 1..3 animate the anchor-loss curve (round 1 slow, rounds 2-3
        # a bit faster); the montage takes over from round 4
        tempos = [(1, 0.07, 1.0), (2, 0.045, 0.7), (2, 0.045, 0.7)]
        for k, (stride, wt, tempo) in enumerate(tempos):
            if k == 1:
                # make the re-bootstrap explicit: new batch, labelled
                self.set_eval_squares(D["cal_evals"][k][EV_IDX], tempo=1.0)
                btxt = VGroup(
                    Text("each round, ", font_size=21, color=TEXT),
                    self.msub("X", "eval", font_size=21, color=GREEN),
                    Text(" is a fresh ID bootstrap", font_size=21,
                         color=TEXT),
                ).arrange(RIGHT, buff=0.08)
                bbox = SurroundingRectangle(
                    btxt, color=SURFACE, buff=0.14, corner_radius=0.08,
                    stroke_width=1.5, fill_color=BG, fill_opacity=0.9)
                boot = VGroup(bbox, btxt).move_to(RIGHT * 3.55 + DOWN * 0.2)
                boot.set_z_index(8)
                sq_pts = np.array(
                    [sq.get_center()[:2] for sq in self.eval_m])
                ptrs = VGroup(*[
                    Line(boot.get_left(), [p[0], p[1], 0],
                         stroke_color=GREEN, stroke_width=1.6,
                         stroke_opacity=0.8)
                    for p in fps3(sq_pts)
                ])
                self.play(FadeIn(boot), Create(ptrs), run_time=1.0)
                self.wait(1.2)
                self.play(FadeOut(boot), FadeOut(ptrs), run_time=0.5)
            elif k > 1:
                self.set_eval_squares(D["cal_evals"][k][EV_IDX], tempo)
            self.run_round(f"cal{k}", D["cal_trajs"][k],
                           step_stride=stride, step_wait=wt,
                           note=f"calibration round {k + 1} of {K}",
                           tempo=tempo)
            if k == 0:
                self.hist_ax = Axes(
                    x_range=[H_LO, H_HI, 0.04],
                    y_range=[0, MAX_COUNT + 1, 3],
                    x_length=5.4, y_length=2.1, tips=False,
                    axis_config={"color": SURFACE, "font_size": 18},
                ).move_to(HIST_C)
                hxl = Text("final anchor loss λ", font_size=18, color=SUBTEXT)
                hxl.next_to(self.hist_ax, DOWN, buff=0.08)
                self.hist_deco = hxl
                self.play(Create(self.hist_ax), FadeIn(hxl), run_time=0.9)
                self.hist_vals = []
                self.hist_bars = VGroup()
                self.add(self.hist_bars)
            if tempo >= 0.5:
                self.push_hist(CAL_LAMS[k], animate_from=self.end_dot)
            else:
                self.remove(self.end_dot)
                self.push_hist(CAL_LAMS[k], fast=True)
        self.wait(0.3)

    def set_eval_squares(self, pts, tempo):
        new = VGroup(*[
            Square(side_length=0.2, color=GREEN, fill_color=GREEN,
                   fill_opacity=0.95, stroke_width=0).move_to(self.d2p(p))
            for p in pts
        ])
        if tempo >= 0.6:
            self.play(FadeOut(self.eval_m), run_time=0.3)
            self.play(FadeIn(new, lag_ratio=0.01), run_time=0.4)
        else:
            self.remove(self.eval_m)
            self.add(new)
        self.eval_m = new

    def run_round(self, key, traj, step_stride, step_wait, note,
                  curve_color=GREEN, tempo=1.0):
        note_m = Text(note, font_size=20, color=curve_color)
        note_m.next_to(self.loss_ax, UP, buff=0.12)
        if tempo >= 0.6:
            self.play(FadeIn(note_m), run_time=0.5 * tempo)
        else:
            self.add(note_m)

        curve = VMobject(stroke_color=curve_color, stroke_width=3.5)
        pts = [self.loss_ax.c2p(0, traj[0])]
        curve.set_points_as_corners(pts * 2)
        self.add(curve)
        for s in range(0, N_STEPS, step_stride):
            self.swap_field(f"{key}_{s + 1}")
            pts.append(self.loss_ax.c2p(s + 1, traj[s]))
            curve.set_points_as_corners(pts)
            self.wait(step_wait)
        # land on the exact final step
        self.swap_field(f"{key}_{N_STEPS}")
        pts.append(self.loss_ax.c2p(N_STEPS, traj[-1]))
        curve.set_points_as_corners(pts)
        self.end_dot = Dot(pts[-1], radius=0.07, color=GOLD)
        if tempo >= 0.5:
            ring = Circle(radius=0.16, color=GOLD, stroke_width=2.5)
            ring.move_to(pts[-1])
            self.play(FadeIn(self.end_dot), Create(ring),
                      run_time=0.6 * tempo)
            self.wait(0.3 * tempo)
            self.play(FadeOut(note_m), FadeOut(ring), FadeOut(curve),
                      run_time=0.5 * tempo)
        else:
            self.add(self.end_dot)
            self.wait(0.1)
            self.remove(note_m, curve)

    def hist_geom(self, vals, ax):
        counts, _ = np.histogram(vals, bins=HIST_BINS)
        bars = VGroup()
        for c, lo, hi in zip(counts, HIST_BINS[:-1], HIST_BINS[1:]):
            if c == 0:
                continue
            p0, p1 = ax.c2p(lo, 0), ax.c2p(hi, c)
            bar = Rectangle(width=p1[0] - p0[0], height=p1[1] - p0[1],
                            fill_color=C1, fill_opacity=0.8,
                            stroke_color=BG, stroke_width=1)
            bar.move_to([(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, 0])
            bars.add(bar)
        return bars

    def push_hist(self, lam, animate_from=None, fast=False):
        old_counts = np.histogram(self.hist_vals, bins=HIST_BINS)[0]
        self.hist_vals.append(lam)
        new_bars = self.hist_geom(self.hist_vals, self.hist_ax)
        if animate_from is not None:
            # the dot becomes ONLY the new chunk, stacked on what is already
            # there; the full histogram is then swapped in pixel-identically
            b = int(np.clip(np.digitize(lam, HIST_BINS) - 1, 0,
                            len(HIST_BINS) - 2))
            lo, hi = HIST_BINS[b], HIST_BINS[b + 1]
            p0 = self.hist_ax.c2p(lo, old_counts[b])
            p1 = self.hist_ax.c2p(hi, old_counts[b] + 1)
            chunk = Rectangle(width=p1[0] - p0[0], height=p1[1] - p0[1],
                              fill_color=C1, fill_opacity=0.8,
                              stroke_color=BG, stroke_width=1)
            chunk.move_to([(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, 0])
            self.play(ReplacementTransform(animate_from, chunk),
                      run_time=0.7)
            self.remove(self.hist_bars, chunk)
            self.add(new_bars)
        else:
            self.remove(self.hist_bars)
            self.add(new_bars)
            if not fast:
                self.wait(0.1)
        self.hist_bars = new_bars

    # ------------------------------------------------------------- seq 4
    def seq4(self):
        self.set_title(f"Repeat on K = {K} known-ID batches")
        counter = Text(f"round 4 / {K}", font_size=20, color=GREEN)
        counter.next_to(self.hist_ax, UP, buff=0.12)
        self.add(counter)
        for k in range(3, K):
            self.set_eval_squares(D["cal_evals_all"][k][EV_IDX], tempo=0.1)
            self.swap_field(f"calfin_{k}")
            new_c = Text(f"round {k + 1} / {K}", font_size=20, color=GREEN)
            new_c.next_to(self.hist_ax, UP, buff=0.12)
            self.remove(counter); self.add(new_c); counter = new_c
            self.push_hist(CAL_LAMS[k], fast=True)
            self.wait(0.07)
        self.remove(counter)
        self.play(FadeOut(self.eval_m), run_time=0.4)
        self.eval_m = None

        base = self.hist_ax.c2p(THRESH, 0)
        y_top = self.hist_ax.c2p(THRESH, MAX_COUNT + 0.6)[1]
        self.thresh_m = DashedLine(base, [base[0], y_top, 0],
                                   color=GOLD, stroke_width=3)
        tl = Text("α-quantile\nthreshold", font_size=15, color=GOLD,
                  line_spacing=0.8)
        tl.move_to([base[0], base[1] - 0.55, 0])
        self.play(Create(self.thresh_m), FadeIn(tl), run_time=0.9)
        self.thresh_lab = tl
        self.wait(1.0)

    # ------------------------------------------------------- deployment
    def deploy_batch(self, pts, key, traj, note, curve_color):
        """Materialize an unknown batch, run the full finetune curve, and
        return the final λ."""
        self.swap_field("cal0_0")            # pristine field again
        batch = VGroup(*[
            Circle(radius=0.12, color=GREEN, stroke_width=3.2)
            .move_to(self.d2p(p)) for p in pts[EV_IDX]
        ])
        self.play(FadeIn(batch, lag_ratio=0.02), run_time=1.0)
        self.run_round(key, traj, step_stride=1, step_wait=0.05,
                       note=note, curve_color=curve_color)
        return batch

    def seq5(self):
        self.set_title("Deployment: an unknown (benign) batch arrives",
                       t2c={"benign": TEAL})
        # unknown labels -> green circles; update the legend marker
        self.legend_spec[-1] = (["circ"], "X", "eval")
        self.relegend(run_time=0.5)
        self.dep_batch = self.deploy_batch(
            D["X_dep_id"], "depid", D["dep_id_traj"],
            "deployment: unknown batch", TEAL)
        x = self.hist_ax.c2p(LAM_ID, 0)
        line = Line(x, [x[0], self.hist_ax.c2p(0, MAX_COUNT + 0.4)[1], 0],
                    color=TEAL, stroke_width=5)
        self.play(ReplacementTransform(self.end_dot, line), run_time=0.8)
        self.id_line = line
        self.id_verdict = self.verdict_card(
            "λ* above the α-quantile threshold", "verdict: no alarm", TEAL)
        self.wait(1.6)

    def seq6(self):
        self.set_title(
            "Deployment: an unknown (harmfully shifted) batch arrives",
            t2c={"harmfully shifted": RED})
        self.play(FadeOut(self.dep_batch), FadeOut(self.id_verdict),
                  run_time=0.5)
        batch = self.deploy_batch(
            D["X_ood"], "depsh", D["dep_sh_traj"],
            f"deployment: rotated batch", RED)

        # λ* is so low the histogram has to zoom out to show it
        new_ax = Axes(
            x_range=[0, H_HI, 0.2], y_range=[0, MAX_COUNT + 1, 3],
            x_length=5.4, y_length=2.1, tips=False,
            axis_config={"color": SURFACE, "font_size": 18},
        ).move_to(HIST_C)
        new_bars = self.hist_geom(self.hist_vals, new_ax)
        nb = new_ax.c2p(THRESH, 0)
        new_thresh = DashedLine(
            nb, [nb[0], new_ax.c2p(THRESH, MAX_COUNT + 0.6)[1], 0],
            color=GOLD, stroke_width=3)
        new_tl = Text("α-quantile\nthreshold", font_size=15, color=GOLD,
                      line_spacing=0.8).move_to([nb[0], nb[1] - 0.55, 0])
        xi = new_ax.c2p(LAM_ID, 0)
        new_id_line = Line(
            xi, [xi[0], new_ax.c2p(0, MAX_COUNT + 0.4)[1], 0],
            color=TEAL, stroke_width=4)
        self.play(
            ReplacementTransform(self.hist_ax, new_ax),
            ReplacementTransform(self.hist_bars, new_bars),
            ReplacementTransform(self.thresh_m, new_thresh),
            ReplacementTransform(self.thresh_lab, new_tl),
            ReplacementTransform(self.id_line, new_id_line),
            run_time=1.2,
        )
        self.hist_ax, self.hist_bars = new_ax, new_bars
        self.thresh_m, self.thresh_lab = new_thresh, new_tl

        x = self.hist_ax.c2p(LAM_SH, 0)
        line = Line(x, [x[0], self.hist_ax.c2p(0, MAX_COUNT + 0.4)[1], 0],
                    color=RED, stroke_width=5)
        self.play(ReplacementTransform(self.end_dot, line), run_time=0.8)
        self.verdict_card(
            "λ* below the α-quantile threshold", "verdict: harmful shift",
            RED)
        alarm = SurroundingRectangle(VGroup(self.hist_ax), color=RED,
                                     buff=0.25, stroke_width=5)
        self.play(Create(alarm), run_time=0.5)
        self.play(FadeOut(alarm), run_time=0.4)
        plane_alarm = SurroundingRectangle(VGroup(self.frame_m), color=RED,
                                           buff=0.12, stroke_width=5)
        self.play(Create(plane_alarm), run_time=0.5)
        self.wait(2.0)
