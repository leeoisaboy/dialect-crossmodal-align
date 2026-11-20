"""Render a crisp animated demo GIF for CrossModal Align Lab."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = REPO_ROOT / "studio" / "data" / "lab_bundle.json"
OUT_GIF = REPO_ROOT / "docs" / "assets" / "lab-demo.gif"

W, H = 960, 540
FPS = 16

BG = (251, 251, 253)
INK = (29, 29, 31)
INK_SOFT = (66, 66, 69)
MUTED = (110, 110, 115)
FAINT = (134, 134, 139)
BLUE = (0, 113, 227)
BLUE_SOFT = (228, 240, 255)
TEAL = (43, 156, 148)
TEAL_SOFT = (228, 246, 243)
CORAL = (230, 106, 78)
CORAL_SOFT = (253, 236, 231)
VIOLET = (123, 107, 176)
HAIRLINE = (232, 232, 236)

F_EYEBROW = ImageFont.truetype(r"C:\Windows\Fonts\consolab.ttf", 12)
F_TITLE = ImageFont.truetype(r"C:\Windows\Fonts\msyhbd.ttc", 28)
F_SUB = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 13)
F_MONO = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 11)
F_MONO_BD = ImageFont.truetype(r"C:\Windows\Fonts\consolab.ttf", 12)
F_CN = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 12)
F_CN_BD = ImageFont.truetype(r"C:\Windows\Fonts\msyhbd.ttc", 13)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def ease_in_out(t: float) -> float:
    x = clamp(t, 0.0, 1.0)
    return -(math.cos(math.pi * x) - 1.0) * 0.5


def make_base() -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((36, 16, 58, 38), radius=7, fill=BLUE_SOFT)
    draw.ellipse((42, 24, 48, 30), fill=BLUE)
    draw.text((66, 19), "CrossModal Align", font=F_CN_BD, fill=INK)
    draw.text((196, 21), "Lab", font=F_MONO, fill=MUTED)
    # Trilogy capsule in center
    draw.rounded_rectangle((340, 15, 650, 39), radius=12, fill=(242, 242, 246))
    draw.text((356, 20), "声学几何示波  ·  大模型审计台  ·  ", font=F_CN, fill=MUTED)
    draw.text((544, 20), "跨模态对齐实验室", font=F_CN_BD, fill=BLUE)
    draw.rounded_rectangle((W - 146, 15, W - 36, 39), radius=12, fill=(242, 242, 246))
    draw.ellipse((W - 134, 23, W - 126, 31), fill=TEAL)
    draw.text((W - 118, 20), "v4 白盒沙盒", font=F_CN, fill=INK_SOFT)
    return img


def draw_pill(draw: ImageDraw.ImageDraw, cx: float, cy: float, text: str, bg: tuple[int, int, int], fg: tuple[int, int, int]) -> None:
    l, t, r, b = draw.textbbox((0, 0), text, font=F_MONO_BD)
    pw, ph = (r - l) + 20, (b - t) + 10
    draw.rounded_rectangle((cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2), radius=ph / 2, fill=bg)
    draw.text((cx - (r - l) / 2, cy - (b - t) / 2 - 1), text, font=F_MONO_BD, fill=fg)


def draw_header(draw: ImageDraw.ImageDraw, eyebrow: str, a: str, b: str, sub: str) -> None:
    draw.text((40, 56), eyebrow, font=F_EYEBROW, fill=BLUE)
    draw.text((40, 76), a, font=F_TITLE, fill=INK)
    l, _, r, _ = draw.textbbox((40, 76), a, font=F_TITLE)
    draw.text((r + 4, 76), b, font=F_TITLE, fill=FAINT)
    draw.text((40, 114), sub, font=F_SUB, fill=MUTED)


def render_trap(bundle: dict[str, Any], t: float) -> Image.Image:
    img = make_base()
    draw = ImageDraw.Draw(img)
    rt = bundle["routing_trap"]
    draw_header(
        draw,
        "01 / ROUTING TRAP SIMULATOR",
        "AUROC 0.999，",
        "命中却跌到 4%。",
        "拖动跨域偏移 gate_domain_shift：单轨 Recall@5 与 AUROC 纹丝不动，双轨命中却撞上红线墙断崖塌方。",
    )
    p = ease_in_out((t - 0.12) / 0.65)
    shift = -1.0 * p
    hit = int(round(100.0 - 96.0 * (p ** 2.2)))

    # 3 live gauges
    draw.text((40, 156), f"gate_domain_shift = {shift:.2f}", font=F_MONO_BD, fill=BLUE)
    draw.text((310, 156), "Recall@5: 100%", font=F_MONO_BD, fill=TEAL)
    draw.text((460, 156), "AUROC: 0.999", font=F_MONO_BD, fill=BLUE)
    draw.text((610, 156), f"双轨联合命中: {hit}%", font=F_MONO_BD, fill=CORAL if hit < 50 else TEAL)

    pad_l, pad_r, top_y, bot_y = 60, 60, 210, 456
    plot_w, plot_h = W - pad_l - pad_r, bot_y - top_y
    x_of = lambda v: pad_l + ((v - (-2.2)) / (2.4 - (-2.2))) * plot_w
    thr_x = x_of(rt["threshold"])
    mu_pos = rt["pos_mean_base"] + shift
    mu_neg = rt["neg_mean_base"] + shift
    sigma = rt["std"]
    gauss = lambda x, mu: math.exp(-0.5 * ((x - mu) / sigma) ** 2)

    draw.line((pad_l, bot_y, W - pad_r, bot_y), fill=HAIRLINE, width=1)

    # Neg curve
    neg_pts = [(x_of(-2.2 + 4.6 * i / 120), bot_y - gauss(-2.2 + 4.6 * i / 120, mu_neg) * plot_h * 0.72) for i in range(121)]
    draw.polygon([(pad_l, bot_y), *neg_pts, (W - pad_r, bot_y)], fill=(238, 238, 242))
    draw.line(neg_pts, fill=FAINT, width=2)

    # Pos curve
    pos_pts = [(x_of(-2.2 + 4.6 * i / 120), bot_y - gauss(-2.2 + 4.6 * i / 120, mu_pos) * plot_h * 0.86) for i in range(121)]
    fill_pos = CORAL_SOFT if hit < 50 else TEAL_SOFT
    col_pos = CORAL if hit < 50 else TEAL
    draw.polygon([(pad_l, bot_y), *pos_pts, (W - pad_r, bot_y)], fill=fill_pos)
    draw.line(pos_pts, fill=col_pos, width=3)

    # Red threshold wall
    draw.line((thr_x, top_y - 10, thr_x, bot_y), fill=CORAL, width=3)
    draw_pill(draw, thr_x, top_y - 16, "阈值墙 0.586 (FPR=1%)", CORAL, (255, 255, 255))
    draw_pill(draw, clamp(x_of(mu_pos), 140, W - 140), bot_y - plot_h * 0.90, f"双轨命中 {hit}%", col_pos, (255, 255, 255))

    draw.text((40, H - 32), "eval_routing_trap.py · reports/routing_trap.json · 假高分真崩溃演示", font=F_MONO, fill=FAINT)
    return img


def render_oov(bundle: dict[str, Any], t: float) -> Image.Image:
    img = make_base()
    draw = ImageDraw.Draw(img)
    arena = bundle["oov_arena"]
    draw_header(
        draw,
        "02 / ZERO-SHOT OOV ARENA",
        "音节乐高重组，",
        "击穿闭集上限。",
        "训练词与 OOV 新词互斥但共享音节库：FixedSlotHead 闭集失效 (0%)，LMBridgedHead 零样本飙至 93.8%。",
    )

    is_open = t >= 0.42
    head_key = "lm_bridged" if is_open else "fixed_slot"
    head = arena["heads"][head_key]

    # Left: LEGO seen -> OOV
    syl_cols = {1: BLUE, 2: TEAL, 3: VIOLET, 4: CORAL, 5: (183, 121, 31), 6: (61, 133, 198)}
    oov = arena["oov_words"][0]
    draw.text((40, 158), "SEEN 训练词（Latin-Square 覆盖）", font=F_MONO_BD, fill=FAINT)
    for r, w_obj in enumerate(arena["seen_words"]):
        y = 186 + r * 36
        draw.text((40, y + 4), w_obj["gloss"], font=F_CN, fill=INK_SOFT)
        for p, sid in enumerate(w_obj["syllables"]):
            x = 96 + p * 74
            match = oov["syllables"][p] == sid
            draw.rounded_rectangle((x, y, x + 64, y + 24), radius=6, fill=syl_cols[sid] if match else (240, 240, 244))
            draw.text((x + 24, y + 5), f"S{sid}", font=F_MONO_BD, fill=(255, 255, 255) if match else FAINT)

    draw.text((40, 418), f"未见新词：{oov['name']}「{oov['gloss']}」", font=F_CN_BD, fill=BLUE)
    for p, sid in enumerate(oov["syllables"]):
        x = 96 + p * 74
        draw.rounded_rectangle((x, 444, x + 64, 474), radius=8, fill=syl_cols[sid])
        draw.text((x + 22, 452), f"S{sid}", font=F_MONO_BD, fill=(255, 255, 255))

    # Right: 4x4 similarity matrix
    mx, my, cell = 540, 186, 62
    draw.text((mx, 158), f"{head['name']} · OOV Recall@1: {head['oov_recall1']:.1f}%", font=F_MONO_BD, fill=BLUE if is_open else CORAL)
    for r in range(4):
        for c in range(4):
            val = head["matrix"][r][c]
            cx, cy = mx + c * cell, my + r * cell
            if not is_open:
                fill = (240, 240, 244)
            else:
                fill = BLUE if r == c else (232, 236, 244)
            draw.rounded_rectangle((cx + 3, cy + 3, cx + cell - 3, cy + cell - 3), radius=8, fill=fill)
            if is_open:
                draw.text((cx + 16, cy + 23), f"{val:.2f}", font=F_MONO_BD, fill=(255, 255, 255) if r == c else INK_SOFT)
    if not is_open:
        draw_pill(draw, mx + cell * 2, my + cell * 2, "无锚点槽位 · 闭集失效 (0%)", CORAL, (255, 255, 255))

    draw.text((40, H - 32), "SyntheticDialectWorld · FixedSlotHead vs LMBridgedHead · CosFace 温度退火 10 → 40", font=F_MONO, fill=FAINT)
    return img


def render_domain_fsq(bundle: dict[str, Any], t: float) -> Image.Image:
    img = make_base()
    draw = ImageDraw.Draw(img)
    df = bundle["domain_fsq"]
    draw_header(
        draw,
        "04 / GEOMETRY & FSQ LATTICE",
        "同词跨方言重合，",
        "再落入离散晶格。",
        "形状=词义，颜色=方言域：类条件对齐精准聚合跨方言同义词，FSQ 将特征吸附至 [-1, 1] 离散晶格。",
    )

    p = ease_in_out((t - 0.1) / 0.45)
    fsq_p = ease_in_out((t - 0.62) / 0.25)
    cx, cy, rad = 320, 326, 145
    draw.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), outline=HAIRLINE, width=2)

    if fsq_p > 0.05:
        for g in df["fsq_grid_2d"]:
            gx, gy = cx + g * rad, cy - g * rad
            draw.line((gx, cy - rad, gx, cy + rad), fill=(224, 236, 250), width=1)
            draw.line((cx - rad, gy, cx + rad, gy), fill=(224, 236, 250), width=1)

    dom_cols = [BLUE, TEAL, CORAL]
    for pt in df["points"]:
        x0, y0 = pt["none"]
        x1, y1 = pt["conditional"]
        xf, yf = pt["fsq_cond"]
        x = (x0 * (1 - p) + x1 * p) * (1 - fsq_p) + xf * fsq_p
        y = (y0 * (1 - p) + y1 * p) * (1 - fsq_p) + yf * fsq_p
        px, py = cx + x * rad, cy - y * rad
        col = dom_cols[pt["dom"]]
        if pt["cls"] == 0:
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=col)
        elif pt["cls"] == 1:
            draw.polygon([(px, py - 6), (px + 6, py + 5), (px - 6, py + 5)], fill=col)
        else:
            draw.rectangle((px - 5, py - 5, px + 5, py + 5), fill=col)

    status = "FSQ 离散晶格吸附 · Perplexity 18.4 ✓" if fsq_p > 0.5 else ("类条件对齐 + DANN · 同形跨色重合 ✓" if p > 0.5 else "无域正则 · 按方言域分块")
    draw_pill(draw, cx, 162, status, TEAL if p > 0.5 else FAINT, (255, 255, 255))

    # Right DANN S-curve
    rx0, ry0, rw, rh = 580, 210, 310, 220
    draw.text((rx0, 176), "dann_alpha_schedule (γ = 10.0)", font=F_MONO_BD, fill=INK_SOFT)
    draw.line((rx0, ry0, rx0, ry0 + rh, rx0 + rw, ry0 + rh), fill=HAIRLINE, width=2)
    pts = [(rx0 + rw * i / 30, ry0 + rh - v * rh) for i, v in enumerate(df["dann_curve"])]
    draw.line(pts, fill=BLUE, width=3)
    draw.text((rx0, ry0 + rh + 16), "掩码安全 Perplexity: 18.4 (Unique 76%)", font=F_MONO_BD, fill=TEAL)
    draw.text((rx0, ry0 + rh + 36), "未过滤 padding 反例: 7.1 (虚假拉低)", font=F_MONO, fill=CORAL)

    draw.text((40, H - 32), "domain_alignment_loss(labels=classes) · FSQBottleneck.codebook_metrics(mask=valid)", font=F_MONO, fill=FAINT)
    return img


def main() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    frames = []
    for fn, count in [(render_trap, 42), (render_oov, 40), (render_domain_fsq, 42)]:
        for i in range(count):
            frames.append(fn(bundle, i / max(count - 1, 1)))
    quantized = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(OUT_GIF, save_all=True, append_images=quantized[1:], duration=int(1000 / FPS), loop=0, optimize=True)
    print(f"Wrote {OUT_GIF} ({len(quantized)} frames, {OUT_GIF.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
