import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import os

# --- Configuration ---
width, height = 500, 500
frames_total = 2100
merge_start = 700
fade_out_start = 1900

# Coordinate Grid
x = np.linspace(-5, 5, width)
y = np.linspace(-5, 5, height)
X, Y = np.meshgrid(x, y)
R_base = np.sqrt(X**2 + Y**2)
A_base = np.arctan2(Y, X)

# --- Starfield Setup ---
num_stars = 200
star_x = np.random.randint(0, width, num_stars)
star_y = np.random.randint(0, height, num_stars)
star_max_bright = np.random.uniform(0.2, 0.7, num_stars)

# --- Colormaps ---
fluoro_maps = [
    np.array([[0,0,0], [0,1,1], [1,0,1]]),
    np.array([[0,0,0], [0.3,0.1,1], [0,1,0]])
]

fib_maps = [
    np.array([[0,0,0], [0.7,0.4,0.2], [1,0.8,0]]),
    np.array([[0,0,0], [0.5,0.2,0], [1,0.5,0]])
]

def blend_colors(cmap_list, t, speed=0.006):
    idx_float = (t * speed) % len(cmap_list)
    idx1 = int(idx_float)
    idx2 = (idx1 + 1) % len(cmap_list)
    blend = idx_float - idx1
    colors = (1 - blend) * cmap_list[idx1] + blend * cmap_list[idx2]
    return LinearSegmentedColormap.from_list("blend", colors)

# --- Figure Setup ---
fig = plt.figure(figsize=(8, 8), facecolor='black')
ax = plt.axes([0, 0, 1, 1], frameon=False)
ax.set_axis_off()
img = ax.imshow(np.zeros((height, width, 3)), interpolation="bilinear")

# --- Animation Update ---
def update(t):
    orbit_progress = min(t / merge_start, 1.0)
    separation = 3.5 * (1.0 - orbit_progress)

    galaxy_factor = min(max(0, (t - merge_start) / 250), 1.0)
    spiral_opacity = 1.0 - (galaxy_factor * 0.95)

    global_fade = 1.0 - max(0, (t - fade_out_start) / (frames_total - fade_out_start))

    orbit_speed = 0.025
    cx = separation * np.cos(t * orbit_speed)
    cy = separation * np.sin(t * orbit_speed)

    cmap_a = blend_colors(fluoro_maps, t)
    cmap_b = blend_colors(fib_maps, t)

    Z_a = np.sin(10*np.sqrt((X-cx)**2 + (Y-cy)**2) +
                 7*np.arctan2(Y-cy, X-cx) - t*0.12)
    rgb_a = cmap_a((np.tanh(Z_a * 2.5) + 1) / 2)[:, :, :3] * spiral_opacity

    Z_b = np.sin(10*np.sqrt((X+cx)**2 + (Y+cy)**2) -
                 13*np.arctan2(Y+cy, X+cx) - t*0.08)
    rgb_b = cmap_b((np.tanh(Z_b * 2.5) + 1) / 2)[:, :, :3] * spiral_opacity

    final_rgb = rgb_a + rgb_b

    if galaxy_factor > 0:
        expansion_t = (t - merge_start) / (frames_total - merge_start)
        reach = 0.6 - (expansion_t * 0.5)

        hue_shift = (t * 0.03) % 1.0
        galaxy_color = plt.cm.nipy_spectral(hue_shift)[:3]

        galaxy_wave = np.sin(reach * R_base - 5 * A_base - t * 0.1)
        glow_radius = 0.25 + (expansion_t * 0.2)
        glow = np.exp(-R_base * glow_radius) * 2.2

        bang_mask = np.clip(np.tanh(galaxy_wave * 2.5) * glow * galaxy_factor, 0, 1.2)
        core = np.exp(-R_base * 1.8) * galaxy_factor * 1.6

        final_rgb += bang_mask[:, :, None] * np.array(galaxy_color)
        final_rgb += core[:, :, None] * np.array([1, 1, 1])

    if t > 400:
        star_fade = min(max(0, (t - 400) / 400), 1.0)
        for i in range(num_stars):
            twinkle = 0.6 + 0.4 * np.sin(t * 0.08 + i)
            brightness = star_max_bright[i] * star_fade * twinkle
            final_rgb[star_y[i], star_x[i]] += brightness

    final_rgb *= global_fade
    final_rgb = np.clip(final_rgb, 0, 1)
    img.set_data(final_rgb)
    return [img]

# --- Animation ---
ani = animation.FuncAnimation(fig, update, frames=frames_total, interval=20, blit=True)

# --- GIF EXPORT ONLY ---
ani.save("convergence.gif", writer="pillow", fps=30)

# --- Show Window ---
manager = plt.get_current_fig_manager()
try:
    manager.window.state('zoomed')
except:
    pass

plt.show()