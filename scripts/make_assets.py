"""Render README title banners, DCA dog mascot logo, and pipeline diagrams with Pillow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
MASCOT = ASSETS / "mascot.png"

NAVY = (12, 47, 89)
BLUE = (0, 113, 227)
TEAL = (43, 156, 148)
CORAL = (230, 106, 78)
AMBER = (214, 135, 31)
PURPLE = (123, 107, 176)
MUTED = (88, 105, 122)
GRADIENT = (BLUE, PURPLE, TEAL)

FONT_TITLE = Path(r"C:\Windows\Fonts\bahnschrift.ttf")
FONT_UI = Path(r"C:\Windows\Fonts\segoeuib.ttf")
FONT_CN = Path(r"C:\Windows\Fonts\msyhbd.ttc")
FONT_CN_LIGHT = Path(r"C:\Windows\Fonts\msyh.ttc")


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def gradient(size: tuple[int, int], colors: tuple[tuple[int, int, int], ...]) -> Image.Image:
    width, height = size
    stops = np.linspace(0.0, 1.0, len(colors))
    positions = np.linspace(0.0, 1.0, width)
    channels = []
    for channel in range(3):
        values = np.interp(positions, stops, [color[channel] for color in colors])
        channels.append(values.astype(np.uint8))
    row = np.stack(channels, axis=1)[None, :, :].repeat(height, axis=0)
    return Image.fromarray(row, mode="RGB")


def gradient_text(
    canvas: Image.Image,
    xy: tuple[int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    colors: tuple[tuple[int, int, int], ...] = GRADIENT,
) -> None:
    left, top, right, bottom = text_font.getbbox(text)
    width = right - left + 12
    height = bottom - top + 12
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).text((6 - left, 6 - top), text, font=text_font, fill=255)
    canvas.paste(gradient(mask.size, colors), (xy[0] - 6, xy[1] - 6), mask)


def fit_font(path: Path, text: str, max_width: int, start: int, minimum: int) -> ImageFont.FreeTypeFont:
    size = start
    while size > minimum:
        candidate = font(path, size)
        if candidate.getbbox(text)[2] <= max_width:
            return candidate
        size -= 2
    return font(path, minimum)


def shadow_card(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, int, int, int] = (255, 255, 255, 232),
    shadow: int = 12,
) -> None:
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0 + 3, y0 + shadow, x1 + 3, y1 + shadow), radius=radius, fill=(15, 49, 75, 30))
    overlay = overlay.filter(ImageFilter.GaussianBlur(10))
    canvas.alpha_composite(overlay)
    ImageDraw.Draw(canvas).rounded_rectangle(box, radius=radius, fill=fill)


def pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    color: tuple[int, int, int],
) -> int:
    left, top, right, bottom = text_font.getbbox(text)
    width = right - left + 34
    height = bottom - top + 18
    draw.rounded_rectangle(
        (xy[0], xy[1], xy[0] + width, xy[1] + height),
        radius=height // 2,
        fill=(255, 255, 255, 238),
        outline=(*color, 185),
        width=2,
    )
    draw.text((xy[0] + 17, xy[1] + 8), text, font=text_font, fill=color)
    return width


def wave(draw: ImageDraw.ImageDraw, origin: tuple[int, int], width: int, color: tuple[int, int, int]) -> None:
    x0, y0 = origin
    points = []
    for index in range(width):
        x = x0 + index
        phase = index / width
        y = y0 - 16 * np.sin(phase * 6.0 * np.pi) * (0.35 + 0.65 * phase)
        points.append((x, y))
    draw.line(points, fill=(*color, 145), width=3, joint="curve")


def mascot_layer(size: int) -> Image.Image:
    image = Image.open(MASCOT).convert("RGBA")
    ratio = size / image.width
    return image.resize((size, int(image.height * ratio)), Image.Resampling.LANCZOS)


def title_banner(language: str) -> None:
    width, height = 1600, 500
    canvas = Image.new("RGBA", (width, height), (247, 250, 254, 255))
    draw = ImageDraw.Draw(canvas)

    for index in range(height):
        blend = index / height
        color = (int(249 - 8 * blend), int(251 - 4 * blend), int(255 - 5 * blend), 255)
        draw.line((0, index, width, index), fill=color)

    draw.rounded_rectangle((15, 15, width - 15, height - 15), radius=34, outline=(218, 226, 240, 255), width=2)
    draw.ellipse((1175, 65, 1550, 440), fill=(*BLUE, 20))
    draw.ellipse((1228, 110, 1518, 400), outline=(*PURPLE, 68), width=3)

    dog = mascot_layer(265)
    canvas.alpha_composite(dog, (1240, 92))

    label_font = font(FONT_UI, 21)
    draw.text((82, 48), "ZERO-SHOT OOV  /  CROSS-MODAL ALIGNMENT  /  PYTORCH", font=label_font, fill=BLUE)

    title_font = fit_font(FONT_TITLE, "Dialect-CrossModal-Align", 1040, 86, 62)
    gradient_text(canvas, (80, 88), "Dialect-CrossModal-Align", title_font)

    if language == "zh":
        subtitle = "跨模态方言对齐 · 零样本 OOV 与路由陷阱实验室"
        description = "组合式音节零样本泛化、语速不变注意力、类条件 FSQ 晶格与双轨塌方验证"
        chips = ("零样本 OOV", "300ms 相对相位", "类条件 FSQ", "路由陷阱模拟器")
        footer = "一只小狗，四套对齐头，戳破 AUROC 0.999 背后的 4% 跨域塌方"
    else:
        subtitle = "Cross-modal dialect alignment · Zero-shot OOV & routing-trap lab"
        description = "Compositional syllable OOV, speed-invariant attention, class-conditional FSQ & routing collapse"
        chips = ("Zero-Shot OOV", "300ms Relative PE", "FSQ Lattice", "Routing Trap")
        footer = "One dog, four alignment heads, exposing the 4% collapse behind AUROC 0.999"

    subtitle_font = fit_font(FONT_CN if language == "zh" else FONT_UI, subtitle, 1040, 40, 28)
    draw.text((82, 224), subtitle, font=subtitle_font, fill=NAVY)
    desc_font = font(FONT_CN_LIGHT if language == "zh" else FONT_UI, 24)
    draw.text((84, 286), description, font=desc_font, fill=MUTED)

    chip_font = font(FONT_CN if language == "zh" else FONT_UI, 20)
    cursor = 84
    chip_cols = (BLUE, TEAL, PURPLE, CORAL)
    for index, chip in enumerate(chips):
        cursor += pill(draw, (cursor, 342), chip, chip_font, chip_cols[index]) + 12

    wave(draw, (84, 437), 900, PURPLE)
    draw.line((84, 415, 220, 415), fill=(*BLUE, 255), width=5)
    draw.line((220, 415, 440, 415), fill=(*TEAL, 255), width=5)
    draw.line((440, 415, 720, 415), fill=(*CORAL, 255), width=5)
    footer_font = font(FONT_CN_LIGHT if language == "zh" else FONT_UI, 19)
    draw.text((84, 458), footer, font=footer_font, fill=(129, 145, 157))

    canvas.convert("RGB").save(ASSETS / f"title_banner.{language}.png", optimize=True)


def logo() -> None:
    size = 512
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow_card(canvas, (12, 12, 500, 500), radius=112, fill=(252, 253, 255, 255), shadow=10)
    draw = ImageDraw.Draw(canvas)
    draw.ellipse((38, 132, 260, 354), fill=(*BLUE, 22))
    draw.ellipse((45, 139, 253, 347), outline=(*PURPLE, 80), width=3)
    dog = mascot_layer(178)
    canvas.alpha_composite(dog, (52, 150))

    gradient_text(canvas, (258, 132), "DCA", font(FONT_TITLE, 118), GRADIENT)
    draw.text((262, 268), "CROSSMODAL", font=font(FONT_UI, 23), fill=BLUE)
    draw.text((262, 300), "ALIGN LAB", font=font(FONT_UI, 23), fill=TEAL)
    draw.text((262, 365), "OPEN BENCHMARK", font=font(FONT_UI, 16), fill=MUTED)
    draw.ellipse((262, 401, 275, 414), fill=BLUE)
    draw.ellipse((292, 401, 305, 414), fill=TEAL)
    draw.ellipse((322, 401, 335, 414), fill=CORAL)

    canvas.save(ASSETS / "logo.png", optimize=True)


def pipeline(language: str) -> None:
    width, height = 1600, 900
    canvas = Image.new("RGBA", (width, height), (247, 250, 254, 255))
    draw = ImageDraw.Draw(canvas)
    for index in range(height):
        blend = index / height
        draw.line((0, index, width, index), fill=(int(248 - 6 * blend), int(251 - 4 * blend), int(255 - 4 * blend), 255))

    draw.ellipse((1180, 38, 1545, 230), fill=(*BLUE, 18))
    dog = mascot_layer(135)
    canvas.alpha_composite(dog, (1340, 22))

    title = "一套管线，五个白盒切面" if language == "zh" else "One pipeline, five white-box stages"
    draw.text((64, 58), title, font=font(FONT_CN if language == "zh" else FONT_TITLE, 54), fill=NAVY)
    subtitle = (
        "从组合式音节世界到双轨路由塌方验证：每一层数学算子都可独立消融与可视化"
        if language == "zh"
        else "From compositional syllables to dual-track routing verification: every module is ablatable"
    )
    draw.text((68, 135), subtitle, font=font(FONT_CN_LIGHT if language == "zh" else FONT_UI, 23), fill=MUTED)
    draw.line((68, 178, 440, 178), fill=BLUE, width=5)
    draw.line((440, 178, 680, 178), fill=TEAL, width=5)

    if language == "zh":
        cards = (
            ("01", "音节组合世界", ("共享音节声学库", "拉丁方前缀覆盖", "Seen / OOV 互斥"), BLUE),
            ("02", "膨胀时序编码", ("dilations=(1,2,4)", "300 ms 完整声调", "归一化相对 PE [0,1]"), PURPLE),
            ("03", "四类对齐头", ("FixedSlot 对照", "TextAnchored", "LMBridged 注意力"), TEAL),
            ("04", "域对齐与 FSQ", ("类条件对齐+DANN", "FSQ 离散晶格", "掩码安全码本统计"), AMBER),
            ("05", "路由陷阱审计", ("单轨 Recall@5", "门控 AUROC 0.999", "双轨联合命中 4%"), CORAL),
        )
        bullet_font = font(FONT_CN_LIGHT, 20)
        card_title_font = font(FONT_CN, 24)
        footer = "每个模块都有对应 YAML 配置、单元测试与浏览器交互沙盒"
    else:
        cards = (
            ("01", "Syllable World", ("shared acoustic bank", "Latin-square prefix", "disjoint Seen/OOV"), BLUE),
            ("02", "Dilated Temporal", ("dilations=(1,2,4)", "300 ms dialect tone", "normalized PE [0,1]"), PURPLE),
            ("03", "4 Align Heads", ("FixedSlot baseline", "TextAnchored", "LMBridged attention"), TEAL),
            ("04", "Domain & FSQ", ("class-cond + DANN", "FSQ discrete grid", "mask-safe perplexity"), AMBER),
            ("05", "Routing Audit", ("single Recall@5", "gate AUROC 0.999", "dual-track hit 4%"), CORAL),
        )
        bullet_font = font(FONT_UI, 18)
        card_title_font = font(FONT_UI, 23)
        footer = "Every module maps to a runnable YAML config, unit tests, and the browser Lab"

    x_positions = (62, 368, 674, 980, 1286)
    for (number, heading, bullets, color), x in zip(cards, x_positions):
        shadow_card(canvas, (x, 230, x + 252, 652), radius=22)
        draw.rounded_rectangle((x, 230, x + 252, 242), radius=6, fill=color)
        draw.ellipse((x + 22, 265, x + 76, 319), fill=color)
        number_font = font(FONT_UI, 22)
        nw = number_font.getbbox(number)[2]
        draw.text((x + 49 - nw / 2, 280), number, font=number_font, fill=(255, 255, 255))
        draw.text((x + 20, 340), heading, font=card_title_font, fill=NAVY)
        for row, bullet in enumerate(bullets):
            y = 410 + row * 63
            draw.ellipse((x + 24, y + 7, x + 35, y + 18), fill=color)
            bf = fit_font(FONT_CN_LIGHT if language == "zh" else FONT_UI, bullet, 180, bullet_font.size, 14)
            draw.text((x + 48, y), bullet, font=bf, fill=(64, 80, 96))
        draw.line((x + 24, 585, x + 228, 585), fill=(222, 232, 235), width=2)

    for x_conn in (312, 618, 924, 1230):
        draw.line((x_conn, 441, x_conn + 56, 441), fill=(162, 181, 189), width=3)

    shadow_card(canvas, (62, 696, 1538, 834), radius=22, fill=(255, 255, 255, 210), shadow=8)
    sentence = (
        "组合式音节测 OOV，相对位置抗语速，类条件域对齐防坍缩，双轨联合评测照出假高分。"
        if language == "zh"
        else "Compositional syllables test OOV, relative PE resists speed, class-cond alignment stops collapse, dual-track exposes traps."
    )
    draw.text((94, 724), sentence, font=font(FONT_CN if language == "zh" else FONT_UI, 22), fill=NAVY)
    draw.text((94, 774), footer, font=font(FONT_CN_LIGHT if language == "zh" else FONT_UI, 20), fill=MUTED)
    wave(draw, (92, 806), 1160, BLUE)

    canvas.convert("RGB").save(ASSETS / f"pipeline.{language}.png", optimize=True)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    title_banner("zh")
    title_banner("en")
    logo()
    pipeline("zh")
    pipeline("en")
    print("assets written to", ASSETS)
