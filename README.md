# Viewport Shrink

**Viewport Shrink** is a free Blender add-on for non-destructive X/Y/Z viewport compression and geometry inspection.

It allows you to visually compress the Blender viewport along the X, Y, or Z axis without modifying the actual geometry, object scale, or transforms.

This makes subtle problems in curves, silhouettes, proportions, and surface flow easier to see while you continue working on the original model.



https://github.com/user-attachments/assets/1e64869d-7399-4180-9168-257609bbd423



## Features

- Non-destructive X / Y / Z viewport compression
- Independent control for each axis
- Axis locking
- Quick 0.30 and 1.00 presets
- Custom shrink values
- Custom keyboard shortcuts
- Keyboard + LMB slider-drag controls
- One-click viewport reset
- Projection-aware object selection
- Projection-aware vertex and edge selection
- Projected shortest-path selection
- Depth-aware edit vertices and edges
- X-Ray support

## Blender Compatibility

### Blender 4.2 – 5.2

Download:

[**Download Viewport Shrink v1.3 →**](https://github.com/choiform/viewport-shrink/archive/refs/heads/main.zip)

Install through:

**Edit → Preferences → Extensions → Install from Disk**

### Legacy support: Blender 2.80 – 4.1

Designed for compatibility with these versions. Please report version-specific issues.

Download:

[**Download Viewport Shrink v1.3 →**](https://github.com/choiform/viewport-shrink/blob/main/Viewport_Shrink_v1_3.py)

Install through:

**Edit → Preferences → Add-ons → Install**

Blender 2.79 and older are not supported.

## Location

After installation:

**3D Viewport → N Panel → Viewport Shrink**

## Basic Use

Each axis can be visually compressed independently.

- `1.00` = normal viewport
- `0.30` = compressed inspection view

Viewport Shrink only changes the viewport projection.

It does **not** change:

- object scale
- vertex positions
- geometry
- transforms

## Shortcuts

Click **Set Shortcut** inside Viewport Shrink to configure:

- custom shrink presets
- X / Y / Z slider-drag shortcuts
- Reset All shortcut

Custom preset values can be set between **0.10 and 1.00**.

For slider-drag shortcuts:

- Hold your assigned key combination
- Drag with LMB
- Drag inward to shrink
- Drag outward to expand

## Download

Download the latest version from the **Releases** section of this repository.

## About

Created and designed by **Choi Jung Woo**

Maintained by **Choi.form**

Instagram: [@choi.form](https://www.instagram.com/choi.form/)

Website: [choiform.com](https://choiform.com/)

## License

Viewport Shrink is free and open source software licensed under the **GNU General Public License v3.0**.

See the included `LICENSE` file for details.

Copyright © 2026 Choi Jung Woo
