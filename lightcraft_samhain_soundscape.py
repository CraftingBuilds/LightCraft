# Samhain_SoundScape_LightCraft.py
# Pyto-safe version without imageio_ffmpeg

import os, math, imageio as iio
from PIL import Image, ImageDraw

out_dir = os.path.expanduser("~/LightCraft/outputs/samhain_soundscape_frames")
os.makedirs(out_dir, exist_ok=True)

frames = []
for t in range(150):  # 150 frames for 2.5 seconds @60fps
    img = Image.new("RGB", (1080, 1080), (10, 0, 20))
    draw = ImageDraw.Draw(img)
    for i in range(6):
        angle = math.pi * 2 * i / 6 + t / 15
        x = 540 + 300 * math.cos(angle)
        y = 540 + 300 * math.sin(angle)
        draw.line([(540, 540), (x, y)], fill=(255, 128, 0), width=2)
    path = os.path.join(out_dir, f"frame_{t:03d}.png")
    img.save(path)
    frames.append(img)

gif_path = os.path.join(out_dir, "Samhain_SoundScape.gif")
iio.mimsave(gif_path, frames, duration=0.04)

print("✅ Frames exported to:", out_dir)
print("✅ GIF preview saved:", gif_path)
print("⚠️ For MP4 conversion, run later on desktop:")
print("   ffmpeg -framerate 25 -i frame_%03d.png -pix_fmt yuv420p Samhain_SoundScape.mp4")