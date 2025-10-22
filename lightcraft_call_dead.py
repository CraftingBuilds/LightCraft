# ~/Documents/Pyto/lightcraft_call_dead_pure.py
# LightCraft: Call the Dead — iOS-safe frame & GIF renderer (no ffmpeg)

import os, math, numpy as np, imageio
from pathlib import Path
from PIL import Image

# Completely disable ffmpeg backends (prevents hidden imports)
os.environ["IMAGEIO_NO_FFMPEG"] = "1"
os.environ["IMAGEIO_FFMPEG_EXE"] = "disabled"

try:
    import cairosvg
except Exception:
    cairosvg = None


# ---------- Helper Functions ----------
def svg_to_pil(svg_path, scale=1.0, target_size=(1080,1080)):
    """Convert SVG to PIL image (RGBA). Falls back to PNG if no CairoSVG."""
    if cairosvg:
        png_bytes = cairosvg.svg2png(
            url=str(svg_path),
            output_width=target_size[0],
            output_height=target_size[1],
            scale=scale
        )
        from io import BytesIO
        return Image.open(BytesIO(png_bytes)).convert("RGBA")
    png_path = svg_path.with_suffix(".png")
    if png_path.exists():
        return Image.open(png_path).convert("RGBA")
    raise RuntimeError(f"Need cairosvg or a PNG next to {svg_path}")


def exp_ease(x):
    """Smooth exponential easing curve."""
    return 1 - math.exp(-5 * x)


# ---------- Paths ----------
DOCS = Path.home() / "Documents" / "LightCraft"
ASSETS = DOCS / "assets"
OUTDIR = DOCS / "exports"
FRAMES = OUTDIR / "frames_call_dead"
OUTDIR.mkdir(parents=True, exist_ok=True)
FRAMES.mkdir(parents=True, exist_ok=True)

AUDIO_FILE = ASSETS / "Call_the_Dead.wav"
VESICA_SVG = ASSETS / "vesica_piscis_lightcraft.svg"
TRISKELION_SVG = ASSETS / "triskelion_lightcraft.svg"
SEAL_IMG = ASSETS / "SoundCraft Seal.png"
DECAL_IMG = ASSETS / "SoundCraft Decal.png"

DURATION = 61.0
FPS = 30
SIZE = (1080,1080)
N_FRAMES = int(DURATION * FPS)


# ---------- Main ----------
def main():
    print("🎬 Rendering LightCraft: Call the Dead (Pyto-safe)")

    vesica = svg_to_pil(VESICA_SVG)
    triskelion = svg_to_pil(TRISKELION_SVG)
    seal = Image.open(SEAL_IMG).convert("RGBA")
    decal = Image.open(DECAL_IMG).convert("RGBA")

    frames = []

    for i in range(N_FRAMES):
        t = i / FPS
        frame = Image.new("RGBA", SIZE, (0,0,0,255))

        # Geometry switch
        if t < 40:
            rot, scale = 4*t, 1.0 + 0.05 * math.sin(t*2*math.pi/10)
            geom = vesica.copy().rotate(rot)
        else:
            rot, scale = -8*(t-40), 0.98 + 0.08 * math.sin(t*2*math.pi/8)
            geom = triskelion.copy().rotate(rot)

        geom = geom.resize((int(SIZE[0]*scale), int(SIZE[1]*scale)))
        frame.paste(geom, ((SIZE[0]-geom.width)//2, (SIZE[1]-geom.height)//2), geom)

        # Seal overlay
        s = seal.copy().resize((500,500))
        s.putalpha(120)
        frame.paste(s, ((SIZE[0]-s.width)//2, (SIZE[1]-s.height)//2), s)

        # Decal & white hold
        if DURATION-1 < t < DURATION+3.14:
            d = decal.copy()
            fade = exp_ease((t - DURATION + 1) / 3.14)
            d.putalpha(int(255 * fade))
            frame.paste(d, ((SIZE[0]-d.width)//2, (SIZE[1]-d.height)//2), d)

        if DURATION+3.14 <= t < DURATION+3.54:
            frame = Image.new("RGBA", SIZE, (255,255,255,255))

        rgb = frame.convert("RGB")
        frame_path = FRAMES / f"frame_{i:04d}.png"
        rgb.save(frame_path)
        frames.append(rgb)
        if i % 100 == 0:
            print(f"Rendered frame {i}/{N_FRAMES}")

    # ✅ Pure Python GIF creation (safe on iOS)
    gif_path = OUTDIR / "LightCraft_Call_the_Dead.gif"
    imageio.mimsave(gif_path, frames[::5], duration=0.05)

    print("✅ Frames exported:", FRAMES)
    print("✅ GIF preview saved:", gif_path)
    print("⚠️ Convert to MP4 later on desktop:")
    print("   ffmpeg -framerate 30 -i frame_%04d.png -pix_fmt yuv420p LightCraft_Call_the_Dead_Final.mp4")


if __name__ == "__main__":
    main()