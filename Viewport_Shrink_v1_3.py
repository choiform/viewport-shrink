# -----------------------------------------------------------------------------
# Viewport Shrink
#
# Created and designed by Choi Jung Woo
# Maintained by Choi.form
#
# Copyright © 2026 Choi Jung Woo
# SPDX-FileCopyrightText: 2026 Choi Jung Woo
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Official releases are distributed by Choi.form.
# Official website: https://choiform.com/
# -----------------------------------------------------------------------------

bl_info = {
  "name": "Viewport Shrink",
  "author": "Choi.form",
  "version": (1, 3),
  "blender": (2, 80, 0),
  "location": "3D Viewport > N Panel > Viewport Shrink",
  "description": "Single-viewport non-destructive X/Y/Z viewport shrink",
  "category": "3D View",
}

import bpy
import bmesh
import gpu
import time
import math
import heapq
import uuid
import sys

try:
  import bgl
except Exception:
  bgl = None
from bpy.props import BoolProperty, FloatProperty, StringProperty, CollectionProperty
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector


# -----------------------------------------------------------------------------
# Blender 2.80 -> 5.2 compatibility layer
# -----------------------------------------------------------------------------
# Blender 2.80 is the oldest practical target for this add-on. The `gpu`
# module and the modern Blender add-on/property API start with the 2.80 series.
# Blender 2.79 and older use a substantially different API and are unsupported.
_BLENDER_VERSION = tuple(getattr(bpy.app, "version", (2, 80, 0)))


def _is_macos():
  return sys.platform == "darwin"


def _shader_from_builtin(name):
  """Return a built-in shader across Blender's old/new shader naming eras."""
  candidates = {
    'IMAGE': ('IMAGE', '2D_IMAGE'),
    'UNIFORM_COLOR': ('UNIFORM_COLOR', '2D_UNIFORM_COLOR', '3D_UNIFORM_COLOR'),
    'POLYLINE_UNIFORM_COLOR': (
      'POLYLINE_UNIFORM_COLOR',
      '3D_POLYLINE_UNIFORM_COLOR',
      '3D_UNIFORM_COLOR',
    ),
  }.get(name, (name,))

  last_error = None
  for candidate in candidates:
    try:
      return gpu.shader.from_builtin(candidate)
    except Exception as exc:
      last_error = exc
  if last_error is not None:
    raise last_error
  raise RuntimeError("No compatible GPU shader found")


def _gpu_has_state(name):
  state = getattr(gpu, 'state', None)
  return state is not None and hasattr(state, name)


def _gpu_blend_get():
  if _gpu_has_state('blend_get'):
    return gpu.state.blend_get()
  if bgl is not None:
    try:
      return 'ALPHA' if bgl.glIsEnabled(bgl.GL_BLEND) else 'NONE'
    except Exception:
      pass
  return 'NONE'


def _gpu_blend_set(mode):
  if _gpu_has_state('blend_set'):
    gpu.state.blend_set(mode)
    return
  if bgl is None:
    return
  try:
    if mode == 'NONE':
      bgl.glDisable(bgl.GL_BLEND)
    else:
      bgl.glEnable(bgl.GL_BLEND)
      bgl.glBlendFunc(bgl.GL_SRC_ALPHA, bgl.GL_ONE_MINUS_SRC_ALPHA)
  except Exception:
    pass


def _gpu_depth_test_get():
  if _gpu_has_state('depth_test_get'):
    return gpu.state.depth_test_get()
  if bgl is not None:
    try:
      return 'LESS_EQUAL' if bgl.glIsEnabled(bgl.GL_DEPTH_TEST) else 'NONE'
    except Exception:
      pass
  return 'NONE'


def _gpu_depth_test_set(mode):
  if _gpu_has_state('depth_test_set'):
    gpu.state.depth_test_set(mode)
    return
  if bgl is None:
    return
  try:
    if mode == 'NONE':
      bgl.glDisable(bgl.GL_DEPTH_TEST)
      return
    bgl.glEnable(bgl.GL_DEPTH_TEST)
    mapping = {
      'ALWAYS': getattr(bgl, 'GL_ALWAYS', None),
      'LESS': getattr(bgl, 'GL_LESS', None),
      'LESS_EQUAL': getattr(bgl, 'GL_LEQUAL', None),
      'EQUAL': getattr(bgl, 'GL_EQUAL', None),
      'GREATER': getattr(bgl, 'GL_GREATER', None),
      'GREATER_EQUAL': getattr(bgl, 'GL_GEQUAL', None),
    }
    func = mapping.get(mode) or getattr(bgl, 'GL_LEQUAL', None)
    if func is not None:
      bgl.glDepthFunc(func)
  except Exception:
    pass


def _gpu_depth_mask_get():
  if _gpu_has_state('depth_mask_get'):
    return bool(gpu.state.depth_mask_get())
  if _gpu_has_state('depth_mask_set_get'):
    try:
      return bool(gpu.state.depth_mask_set_get())
    except Exception:
      pass
  if bgl is not None:
    try:
      buf = bgl.Buffer(bgl.GL_INT, 1)
      bgl.glGetIntegerv(bgl.GL_DEPTH_WRITEMASK, buf)
      return bool(buf[0])
    except Exception:
      pass
  return True


def _gpu_depth_mask_set(value):
  if _gpu_has_state('depth_mask_set'):
    gpu.state.depth_mask_set(bool(value))
    return
  if bgl is not None:
    try:
      bgl.glDepthMask(bgl.GL_TRUE if value else bgl.GL_FALSE)
    except Exception:
      pass


def _gpu_viewport_get():
  if _gpu_has_state('viewport_get'):
    return tuple(gpu.state.viewport_get())
  if bgl is not None:
    try:
      buf = bgl.Buffer(bgl.GL_INT, 4)
      bgl.glGetIntegerv(bgl.GL_VIEWPORT, buf)
      return tuple(int(buf[i]) for i in range(4))
    except Exception:
      pass
  return (0, 0, 1, 1)


def _gpu_viewport_set(x, y, width, height):
  if _gpu_has_state('viewport_set'):
    gpu.state.viewport_set(int(x), int(y), int(width), int(height))
    return
  if bgl is not None:
    try:
      bgl.glViewport(int(x), int(y), int(width), int(height))
    except Exception:
      pass


def _gpu_line_width_set(width):
  if _gpu_has_state('line_width_set'):
    gpu.state.line_width_set(float(width))
    return
  if bgl is not None:
    try:
      bgl.glLineWidth(float(width))
    except Exception:
      pass


def _offscreen_draw_view3d(offscreen, context, view_matrix, projection_matrix):
  """Call GPUOffScreen.draw_view3d across Blender 2.80 -> 5.x signatures."""
  args = (
    context.scene,
    context.view_layer,
    context.space_data,
    context.region,
    view_matrix,
    projection_matrix,
  )
  try:
    offscreen.draw_view3d(
      *args,
      do_color_management=True,
      draw_background=True,
    )
    return
  except TypeError:
    pass
  try:
    offscreen.draw_view3d(*args, do_color_management=True)
    return
  except TypeError:
    pass
  try:
    offscreen.draw_view3d(*args)
    return
  except TypeError:
    pass

  # Fallback for older/alternate 2.8x signatures that omit ViewLayer.
  offscreen.draw_view3d(
    context.scene,
    context.space_data,
    context.region,
    view_matrix,
    projection_matrix,
  )


def _offscreen_color_texture(offscreen):
  """Return GPUTexture on 3.0+, OpenGL texture id on Blender 2.80-2.93."""
  texture = getattr(offscreen, 'texture_color', None)
  if texture is not None:
    return texture
  return int(getattr(offscreen, 'color_texture'))


def _bind_old_texture(shader, texture_id):
  """Bind a 2.80-2.93 offscreen texture to the old image shader."""
  if bgl is None:
    return False
  try:
    bgl.glActiveTexture(bgl.GL_TEXTURE0)
    bgl.glBindTexture(bgl.GL_TEXTURE_2D, int(texture_id))
    shader.bind()
    try:
      shader.uniform_int('image', (0,))
    except Exception:
      shader.uniform_float('image', 0)
    return True
  except Exception:
    return False


def _unbind_old_texture():
  if bgl is not None:
    try:
      bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)
    except Exception:
      pass


def _scene_ray_cast(context, origin, direction, distance):
  """Use Depsgraph ray-cast when available, otherwise the 2.80 ViewLayer API."""
  scene = context.scene
  try:
    depsgraph = context.evaluated_depsgraph_get()
    return scene.ray_cast(
      depsgraph,
      origin,
      direction,
      distance=distance,
    )
  except TypeError:
    return scene.ray_cast(
      context.view_layer,
      origin,
      direction,
      distance=distance,
    )


# -----------------------------------------------------------------------------
# Runtime state
# -----------------------------------------------------------------------------

_DRAW_HANDLE = None
_OFFSCREENS = {}
_DRAW_GUARD = False

# Vertex/edge LMB hit areas are intentionally 4x Blender-like base tolerance.
_PICK_HITBOX_MULTIPLIER = 4.0

_CV_OVERLAY_CONFIRMED = False

# Maximum near-camera depth offsets. These are NOT applied as fixed values.
# A perspective-depth falloff drives them rapidly toward zero as geometry moves
# toward the far end of the depth range. This prevents hidden edit geometry
# from punching through the surface when zoomed far out.
_CV_NDC_DEPTH_BIAS_MAX = 0.00045
_EDGE_NDC_DEPTH_BIAS_MAX = 0.00028


def _adaptive_ndc_depth_bias(ndc_z, max_bias):
  """Depth bias that becomes negligible at far depth.

  Blender's perspective depth distribution is strongly non-linear. At large
  camera distances, front/back surfaces can differ by only a tiny amount in
  NDC Z. A constant offset therefore eventually becomes larger than the
  actual occlusion gap and makes hidden CVs/edges look like X-Ray.

  Map NDC Z from [-1, 1] to near_factor [1, 0], then square it. The square
  gives us enough offset near the camera for stable surface overlays while
  collapsing extremely quickly toward zero near the far plane.

    NDC -1.0 -> 100% of max bias
    NDC 0.0 -> 25%
    NDC 0.5 ->  6.25%
    NDC 0.9 ->  0.25%
    NDC 0.99 -> 0.0025%

  Orthographic views are still safe because their depth distribution is
  linear and does not collapse with zoom in the same way.
  """
  z = max(-1.0, min(1.0, float(ndc_z)))
  near_factor = max(0.0, min(1.0, (1.0 - z) * 0.5))
  return float(max_bias) * (near_factor * near_factor)

_ADDON_KEYMAPS = []
_DIRTY_SERIAL = 1

# Performance strategy:
# - Never impose an artificial 60 Hz cap. Blender can drive this at the display rate.
# - During continuous interaction, render the custom viewport at reduced resolution.
# - After interaction stops briefly, refresh once at full native resolution.
# This preserves responsiveness while returning to a sharp full-resolution image at rest.
_FAST_RENDER_SCALE = 0.67
_SETTLE_DELAY = 0.085
_SETTLE_TIMER_ACTIVE = False
_RUNTIME_KEY = "VIEWPORT_SHRINK_RUNTIME"

# Extra pan response after shrink. This is applied adaptively according to how
# strongly the current pan direction is visually compressed by X/Y/Z shrink.
# 0.90 means a direction visually compressed to 30% receives about 1.63x the
# already projection-correct pan distance. At 1/1/1 there is no boost.
# Pan speed follows the shrink continuously and exactly:
#  effective shrink 1.0 -> 1.00x pan
#  effective shrink 0.5 -> 2.00x pan
#  effective shrink 0.3 -> 3.33x pan
# The effective value is directional when X/Y/Z use different shrink amounts.
_PAN_MIN_EFFECTIVE_SCALE = 0.05

# Global multiplier applied on top of the shrink-proportional pan speed.
# This makes panning 2x faster at every shrink amount.
_PAN_GLOBAL_MULTIPLIER = 2.0


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def _settings(context=None):
  context = context or bpy.context
  wm = getattr(context, "window_manager", None)
  return getattr(wm, "viewport_shrink_settings", None) if wm else None


def _is_shrunk(settings):
  if not settings:
    return False
  return (abs(settings.shrink_x - 1.0) > 1e-6 or
      abs(settings.shrink_y - 1.0) > 1e-6 or
      abs(settings.shrink_z - 1.0) > 1e-6)


def _mark_dirty(*_args):
  global _DIRTY_SERIAL
  _DIRTY_SERIAL += 1
  if _DIRTY_SERIAL > 2_000_000_000:
    _DIRTY_SERIAL = 1


def _tag_redraw_all_view3d():
  wm = bpy.context.window_manager
  if not wm:
    return
  for window in wm.windows:
    screen = window.screen
    if not screen:
      continue
    for area in screen.areas:
      if area.type == 'VIEW_3D':
        for region in area.regions:
          if region.type == 'WINDOW':
            region.tag_redraw()


def _on_setting_changed(self, context):
  _mark_dirty()
  _tag_redraw_all_view3d()


def _edit_mode_objects(context):
  objs = getattr(context, "objects_in_mode_unique_data", None)
  if objs:
    return list(objs)
  if context.edit_object:
    return [context.edit_object]
  return []


def _supported_edit_mode(context):
  if context.mode not in {'EDIT_CURVE', 'EDIT_SURFACE', 'EDIT_MESH'}:
    return False
  return any(obj.type in {'CURVE', 'SURFACE', 'MESH'} for obj in _edit_mode_objects(context))


# -----------------------------------------------------------------------------
# Projection math -- one source of truth for drawing AND interaction
# -----------------------------------------------------------------------------

def _axis_scale(settings):
  """Return visual world-axis scale factors. Scene data is never modified."""
  sx = max(0.05, min(1.0, float(settings.shrink_x)))
  sy = max(0.05, min(1.0, float(settings.shrink_y)))
  sz = max(0.05, min(1.0, float(settings.shrink_z)))
  return sx, sy, sz


def _world_shrink_matrix(context, settings):
  """Visual-only XYZ scaling about the current viewport pivot.

  This matrix is used only inside the custom viewport projection. It does not
  write object scale, vertex positions, CV coordinates, transforms, or scene
  data. Using the viewport's orbit/pan pivot keeps the point you are inspecting
  visually stationary while an axis is compressed.
  """
  sx, sy, sz = _axis_scale(settings)
  pivot = context.region_data.view_location.copy()
  to_pivot = Matrix.Translation(pivot)
  from_pivot = Matrix.Translation(-pivot)
  scale = Matrix.Diagonal((sx, sy, sz, 1.0))
  return to_pivot @ scale @ from_pivot


def _custom_projection_matrices(context, settings):
  """Return (projection, view_projection) matching the XYZ shrink render.

  GPUOffScreen.draw_view3d receives Blender's native view matrix separately.
  To keep that API path intact, the world-axis visual scale is conjugated into
  the projection matrix:

    P_custom @ V == P_native @ V @ S_world

  where S_world is the non-destructive XYZ scale about the viewport pivot.
  """
  rv3d = context.region_data
  native_view = rv3d.view_matrix.copy()
  native_projection = rv3d.window_matrix.copy()
  shrink_world = _world_shrink_matrix(context, settings)

  inv_view = native_view.inverted_safe()
  projection = native_projection @ native_view @ shrink_world @ inv_view
  view_projection = projection @ native_view
  return projection, view_projection



def _display_view_projection(context, settings):
  """Return the VP matrix of the frame the user is ACTUALLY looking at.

  Viewport Shrink caches an offscreen image. During/just after navigation,
  RegionView3D can already contain a newer matrix while the cached image on
  screen still belongs to the previous frame. Using rv3d directly for
  picking creates the exact symptom where double-click acts somewhere other
  than the visible cursor.

  Prefer the matrix stored with the active offscreen texture. Fall back to
  the current custom projection only when no rendered frame exists yet.
  """
  try:
    entry = _OFFSCREENS.get(_offscreen_key(context))
    if entry is not None:
      vp = entry.get("display_vp")
      if vp is not None and entry.get("active_texture") is not None:
        return vp.copy()
  except Exception:
    pass

  _projection, vp = _custom_projection_matrices(context, settings)
  return vp


def _project_world(context, world_co, settings, view_projection=None):
  """World -> displayed XYZ-shrunk pixel coordinates and NDC depth."""
  if view_projection is None:
    _projection, view_projection = _custom_projection_matrices(context, settings)

  v = view_projection @ Vector((world_co.x, world_co.y, world_co.z, 1.0))
  if abs(v.w) < 1e-12:
    return None
  if context.region_data.is_perspective and v.w <= 0.0:
    return None

  ndc = Vector((v.x / v.w, v.y / v.w, v.z / v.w))
  x = (ndc.x * 0.5 + 0.5) * context.region.width
  y = (ndc.y * 0.5 + 0.5) * context.region.height
  return x, y, ndc.z


def _unproject_display(context, x, y, ndc_z, settings, inv_view_projection=None):
  """Displayed XYZ-shrunk pixel + NDC depth -> original world coordinate."""
  if inv_view_projection is None:
    _projection, view_projection = _custom_projection_matrices(context, settings)
    inv_view_projection = view_projection.inverted_safe()

  ndc_x = (float(x) / max(1.0, context.region.width)) * 2.0 - 1.0
  ndc_y = (float(y) / max(1.0, context.region.height)) * 2.0 - 1.0
  w = inv_view_projection @ Vector((ndc_x, ndc_y, ndc_z, 1.0))
  if abs(w.w) > 1e-12:
    return Vector((w.x / w.w, w.y / w.w, w.z / w.w))
  return Vector((w.x, w.y, w.z))


# -----------------------------------------------------------------------------
# Projection-aware Object Mode picking
# -----------------------------------------------------------------------------

def _display_pick_ray(context, settings, x, y):
  """Create a world-space selection ray through the SHRUNK display pixel.

  The viewport render uses:

    custom_VP = native_P @ native_V @ shrink_world

  so inverting that exact matrix maps the visible mouse coordinate back to
  the untouched world. This is the correct ray for selecting what the user
  actually sees rather than what exists in Blender's hidden unshrunk view.
  """
  vp = _display_view_projection(context, settings)
  inv_vp = vp.inverted_safe()

  p_near = _unproject_display(
    context, x, y, -1.0, settings, inv_vp
  )
  p_far = _unproject_display(
    context, x, y, 1.0, settings, inv_vp
  )

  delta = p_far - p_near
  distance = delta.length
  if distance < 1e-10:
    return None

  return p_near, delta / distance, distance


def _object_selectable_in_view(context, obj):
  if obj is None:
    return False

  try:
    if obj.hide_select:
      return False
  except Exception:
    pass

  try:
    if obj.hide_get():
      return False
  except Exception:
    pass

  try:
    if not obj.visible_get(
      view_layer=context.view_layer,
      viewport=context.space_data,
    ):
      return False
  except TypeError:
    try:
      if not obj.visible_get(view_layer=context.view_layer):
        return False
    except Exception:
      pass
  except Exception:
    pass

  return True


def _raycast_selectable_object(context, origin, direction, distance):
  """Ray-cast through evaluated scene while respecting selection visibility.

  Scene.ray_cast() returns evaluated geometry in world space. If the closest
  hit is explicitly unselectable, advance slightly through it and continue,
  matching the expectation that a non-selectable helper should not steal the
  user's click.
  """
  current_origin = origin.copy()
  remaining = max(0.0, float(distance))
  epsilon = max(1e-5, min(1e-3, remaining * 1e-7))

  for _i in range(24):
    if remaining <= epsilon:
      break

    try:
      result, location, normal, index, obj, matrix = _scene_ray_cast(
        context,
        current_origin,
        direction,
        remaining,
      )
    except Exception:
      return None

    if not result:
      return None

    if _object_selectable_in_view(context, obj):
      return obj

    travelled = (location - current_origin).length
    step = max(epsilon, travelled + epsilon)
    current_origin = current_origin + direction * step
    remaining -= step

  return None


def _pick_object_shrunk(context, settings, mouse_x, mouse_y):
  """Pick the visible object under a shrunk-viewport mouse coordinate.

  Native Blender object selection has a small practical screen tolerance,
  especially around silhouettes. Recreate that feel by trying the exact
  pixel first, then a compact radius around it. This makes clicks on a visible
  edge reliable without turning the picker into a broad bounding-box test.
  """
  ui = float(getattr(context.preferences.system, 'ui_scale', 1.0) or 1.0)

  # Ordered nearest-first so the exact visible point always wins.
  r2 = 2.0 * ui
  r4 = 4.0 * ui
  r6 = 6.0 * ui

  offsets = (
    (0.0, 0.0),
    ( r2, 0.0), (-r2, 0.0), (0.0, r2), (0.0, -r2),
    ( r2, r2), ( r2,-r2), (-r2, r2), (-r2,-r2),
    ( r4, 0.0), (-r4, 0.0), (0.0, r4), (0.0, -r4),
    ( r4, r4), ( r4,-r4), (-r4, r4), (-r4,-r4),
    ( r6, 0.0), (-r6, 0.0), (0.0, r6), (0.0, -r6),
  )

  width = float(context.region.width)
  height = float(context.region.height)

  for ox, oy in offsets:
    x = float(mouse_x) + ox
    y = float(mouse_y) + oy

    if x < 0.0 or y < 0.0 or x >= width or y >= height:
      continue

    ray = _display_pick_ray(context, settings, x, y)
    if ray is None:
      continue

    origin, direction, distance = ray
    obj = _raycast_selectable_object(
      context,
      origin,
      direction,
      distance,
    )
    if obj is not None:
      return obj

  return None


def _apply_object_click_selection(context, obj, shift):
  """Blender-like Object Mode click semantics for the projected picker."""
  view_layer = context.view_layer

  if obj is None:
    if not shift:
      for selected in list(context.selected_objects):
        try:
          selected.select_set(False)
        except Exception:
          pass
      try:
        view_layer.objects.active = None
      except Exception:
        pass
    _force_render_current(context)
    return

  if shift:
    try:
      was_selected = bool(obj.select_get())
    except Exception:
      was_selected = obj in context.selected_objects

    if was_selected:
      try:
        obj.select_set(False)
      except Exception:
        pass
      try:
        if view_layer.objects.active == obj:
          remaining = [
            ob for ob in context.selected_objects
            if ob != obj
          ]
          view_layer.objects.active = remaining[-1] if remaining else None
      except Exception:
        pass
    else:
      try:
        obj.select_set(True)
      except Exception:
        pass
      try:
        view_layer.objects.active = obj
      except Exception:
        pass
  else:
    for selected in list(context.selected_objects):
      if selected != obj:
        try:
          selected.select_set(False)
        except Exception:
          pass
    try:
      obj.select_set(True)
    except Exception:
      pass
    try:
      view_layer.objects.active = obj
    except Exception:
      pass

  _force_render_current(context)


# -----------------------------------------------------------------------------
# Cached offscreen projection drawing
# -----------------------------------------------------------------------------

def _offscreen_key(context):
  return (
    context.window.as_pointer() if context.window else 0,
    context.area.as_pointer() if context.area else 0,
    context.region.as_pointer() if context.region else 0,
  )


def _alloc_offscreen(width, height):
  try:
    return gpu.types.GPUOffScreen(max(8, int(width)), max(8, int(height)))
  except Exception:
    return None


def _new_cache_entry(width, height):
  fast_w = max(8, int(round(width * _FAST_RENDER_SCALE)))
  fast_h = max(8, int(round(height * _FAST_RENDER_SCALE)))

  full = _alloc_offscreen(width, height)
  fast = _alloc_offscreen(fast_w, fast_h)
  if full is None or fast is None:
    for ofs in (full, fast):
      if ofs is not None:
        try:
          ofs.free()
        except Exception:
          pass
    return None

  return {
    "full": full,
    "fast": fast,
    "w": width,
    "h": height,
    "fast_w": fast_w,
    "fast_h": fast_h,
    "source_signature": None,
    "source_dirty_serial": -1,
    "last_change": 0.0,
    "needs_full": True,
    "active_texture": None,
    # Exact view-projection matrix used to generate active_texture.
    # Picking MUST use this matrix rather than a potentially newer rv3d
    # matrix, otherwise a cached frame can be visible while mouse picking
    # is already using the next orbit/zoom frame.
    "display_vp": None,
    # Discrete operations such as selection should never flash through
    # the reduced-resolution interactive buffer.
    "force_full_until": 0.0,
  }


def _free_cache_entry(entry):
  if not entry:
    return
  for key in ("full", "fast"):
    ofs = entry.get(key)
    if ofs is not None:
      try:
        ofs.free()
      except Exception:
        pass


def _get_cache_entry(context, width, height):
  key = _offscreen_key(context)
  entry = _OFFSCREENS.get(key)

  if entry and entry["w"] == width and entry["h"] == height:
    return entry

  if entry:
    _free_cache_entry(entry)
    _OFFSCREENS.pop(key, None)

  entry = _new_cache_entry(width, height)
  if entry:
    _OFFSCREENS[key] = entry
  return entry


def _force_render_current(context):
  """Force a sharp render for discrete interaction.

  Selection can trigger a second depsgraph notification a few milliseconds
  after the click. Keep a brief full-resolution protection window so that
  delayed notification cannot temporarily replace the selected CV/vertex
  with the upscaled low-resolution interactive texture.
  """
  now = time.perf_counter()
  try:
    entry = _OFFSCREENS.get(_offscreen_key(context))
    if entry:
      entry["source_dirty_serial"] = -1
      entry["last_change"] = now
      entry["needs_full"] = True
      entry["force_full_until"] = now + 0.22
  except Exception:
    pass

  _mark_dirty()
  if getattr(context, "area", None):
    context.area.tag_redraw()


def _free_offscreens():
  global _SETTLE_TIMER_ACTIVE
  for entry in list(_OFFSCREENS.values()):
    _free_cache_entry(entry)
  _OFFSCREENS.clear()
  _SETTLE_TIMER_ACTIVE = False


def _matrix_signature(matrix):
  # Ignore floating-point noise that is far below a visible viewport change.
  return tuple(round(float(v), 7) for row in matrix for v in row)


def _draw_signature(context, settings):
  shading = context.space_data.shading
  overlay = context.space_data.overlay
  return (
    _matrix_signature(context.region_data.view_matrix),
    _matrix_signature(context.region_data.window_matrix),
    round(float(settings.shrink_x), 5),
    round(float(settings.shrink_y), 5),
    round(float(settings.shrink_z), 5),
    context.region_data.view_perspective,
    shading.type,
    getattr(shading, "light", ""),
    getattr(shading, "color_type", ""),
    bool(overlay.show_overlays),
  )


def _draw_texture_rect_uv(texture, x0, y0, x1, y1, u0, v0, u1, v1):
  shader = _shader_from_builtin('IMAGE')
  batch = batch_for_shader(
    shader,
    'TRI_FAN',
    {
      "pos": ((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
      "texCoord": ((u0, v0), (u1, v0), (u1, v1), (u0, v1)),
    },
  )

  # Blender 3.0+ exposes a GPUTexture and uniform_sampler(). Blender 2.80-2.93
  # exposes only an OpenGL texture id, which must be bound through bgl.
  if not isinstance(texture, int) and hasattr(shader, 'uniform_sampler'):
    shader.bind()
    shader.uniform_sampler("image", texture)
    batch.draw(shader)
    return

  if _bind_old_texture(shader, texture):
    try:
      batch.draw(shader)
    finally:
      _unbind_old_texture()


def _draw_shrink_texture_fullscreen(texture, width, height):
  """Draw the shrunk viewport across the entire 3D window.

  The current release intentionally removes all navigation-gizmo preservation logic.
  This keeps the compositor simple and visually consistent.
  """
  _draw_texture_rect_uv(
    texture,
    0, 0, width, height,
    0.0, 0.0, 1.0, 1.0,
  )



def _viewport_xray_enabled(context):
  """Mirror Blender's viewport X-Ray intent."""
  shading = context.space_data.shading

  if bool(getattr(shading, "show_xray", False)):
    return True

  if getattr(shading, "type", "") == 'WIREFRAME':
    if bool(getattr(shading, "show_xray_wireframe", False)):
      return True

  return False


def _clip_disc_triangles(points, radius_px, width, height, depth_bias):
  """Build fixed-pixel discs directly in NDC/clip coordinates.

  `points` contains (screen_x, screen_y, ndc_z). X/Y are converted from
  pixels into [-1, 1], while Z remains the exact depth generated by the same
  shrink projection used by GPUOffScreen.draw_view3d().
  """
  verts = []
  if width <= 0 or height <= 0:
    return verts

  rx = (float(radius_px) * 2.0) / float(width)
  ry = (float(radius_px) * 2.0) / float(height)
  segments = 12

  for px, py, ndc_z in points:
    cx = (float(px) / float(width)) * 2.0 - 1.0
    cy = (float(py) / float(height)) * 2.0 - 1.0

    # In Blender's standard projection, smaller NDC Z is closer.
    # Use a depth-adaptive offset so far/zoomed-out hidden points can never
    # be pulled through a nearer surface.
    cz = float(ndc_z) - _adaptive_ndc_depth_bias(ndc_z, depth_bias)

    for i in range(segments):
      a0 = (i / segments) * 6.283185307179586
      a1 = ((i + 1) / segments) * 6.283185307179586
      verts.extend((
        (cx, cy, cz),
        (cx + math.cos(a0) * rx, cy + math.sin(a0) * ry, cz),
        (cx + math.cos(a1) * rx, cy + math.sin(a1) * ry, cz),
      ))

  return verts


def _draw_clip_disc_batch(shader, points, radius, color, width, height, depth_bias):
  if not points:
    return

  verts = _clip_disc_triangles(
    points,
    radius,
    width,
    height,
    depth_bias,
  )
  if not verts:
    return

  batch = batch_for_shader(shader, 'TRIS', {"pos": verts})
  shader.bind()
  shader.uniform_float("color", color)
  batch.draw(shader)



def _theme_edit_edge_style(context):
  """Match Blender's edit-edge colors/width as closely as possible."""
  try:
    theme = context.preferences.themes[0].view_3d
    normal = tuple(float(c) for c in theme.edge[:3]) + (1.0,)
    selected = tuple(float(c) for c in theme.edge_select[:3]) + (1.0,)
    width = float(getattr(theme, "edge_width", 1.0))
  except Exception:
    normal = (0.08, 0.08, 0.08, 1.0)
    selected = (1.0, 0.48, 0.05, 1.0)
    width = 1.0

  try:
    ui = max(0.75, float(context.preferences.system.ui_scale))
  except Exception:
    ui = 1.0

  # Blender's edit edges are compact. Keep the overlay equally compact.
  normal_width = max(1.0, width * ui)
  selected_width = max(normal_width, normal_width + 0.75 * ui)
  return normal, selected, normal_width, selected_width


def _project_world_with_vp(context, world_co, vp):
  """Project a world-space point through the exact scene frame VP matrix."""
  v = vp @ Vector((world_co.x, world_co.y, world_co.z, 1.0))
  if abs(v.w) < 1e-12:
    return None

  # Behind-camera geometry should not generate edit overlay strips.
  if context.region_data.is_perspective and v.w <= 0.0:
    return None

  ndc_x = v.x / v.w
  ndc_y = v.y / v.w
  ndc_z = v.z / v.w

  x = (ndc_x * 0.5 + 0.5) * context.region.width
  y = (ndc_y * 0.5 + 0.5) * context.region.height
  return (x, y, ndc_z)


def _iter_edit_edge_segments(context, view_projection):
  """Yield editable cage/mesh segments in the exact shrunk screen projection.

  Each result is:
    ((x0, y0, z0), (x1, y1, z1), selected)

  Mesh edges use the live Edit BMesh. Curve and Surface objects use their
  control cage so the same visibility stabilization applies to CV networks.
  """
  vp = view_projection

  for obj in _edit_mode_objects(context):
    mw = obj.matrix_world

    if obj.type == 'MESH':
      try:
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()

        for edge in bm.edges:
          if edge.hide:
            continue

          v0, v1 = edge.verts
          if v0.hide or v1.hide:
            continue

          p0 = _project_world_with_vp(context, mw @ v0.co, vp)
          p1 = _project_world_with_vp(context, mw @ v1.co, vp)
          if p0 is None or p1 is None:
            continue

          # Skip segments completely outside the clip depth range.
          if ((p0[2] < -1.01 and p1[2] < -1.01) or
              (p0[2] > 1.01 and p1[2] > 1.01)):
            continue

          yield p0, p1, bool(edge.select)

      except Exception:
        continue

    elif obj.type in {'CURVE', 'SURFACE'}:
      for spline in obj.data.splines:
        # Bezier: center-to-center cage plus handle stems.
        if spline.type == 'BEZIER':
          points = spline.bezier_points
          count = len(points)

          # Center chain.
          for i in range(max(0, count - 1)):
            a = points[i]
            b = points[i + 1]
            if getattr(a, 'hide', False) or getattr(b, 'hide', False):
              continue
            p0 = _project_world_with_vp(context, mw @ a.co, vp)
            p1 = _project_world_with_vp(context, mw @ b.co, vp)
            if p0 and p1:
              yield p0, p1, bool(
                a.select_control_point and b.select_control_point
              )

          if getattr(spline, 'use_cyclic_u', False) and count > 2:
            a = points[-1]
            b = points[0]
            if not getattr(a, 'hide', False) and not getattr(b, 'hide', False):
              p0 = _project_world_with_vp(context, mw @ a.co, vp)
              p1 = _project_world_with_vp(context, mw @ b.co, vp)
              if p0 and p1:
                yield p0, p1, bool(
                  a.select_control_point and b.select_control_point
                )

          # Handle stems, matching Blender's editable Bezier cage.
          for bp in points:
            if getattr(bp, 'hide', False):
              continue
            pc = _project_world_with_vp(context, mw @ bp.co, vp)
            pl = _project_world_with_vp(context, mw @ bp.handle_left, vp)
            pr = _project_world_with_vp(context, mw @ bp.handle_right, vp)
            if pc and pl:
              yield pc, pl, bool(
                bp.select_control_point or bp.select_left_handle
              )
            if pc and pr:
              yield pc, pr, bool(
                bp.select_control_point or bp.select_right_handle
              )
          continue

        # POLY/NURBS curve or NURBS surface cage.
        pts = spline.points
        count_u = max(1, int(getattr(spline, 'point_count_u', len(pts))))
        count_v = max(1, int(getattr(spline, 'point_count_v', 1)))

        def p_index(u, v):
          return v * count_u + u

        # U direction.
        for v in range(count_v):
          for u in range(max(0, count_u - 1)):
            ia, ib = p_index(u, v), p_index(u + 1, v)
            if ia >= len(pts) or ib >= len(pts):
              continue
            a, b = pts[ia], pts[ib]
            if getattr(a, 'hide', False) or getattr(b, 'hide', False):
              continue
            p0 = _project_world_with_vp(
              context, mw @ Vector(a.co[:3]), vp
            )
            p1 = _project_world_with_vp(
              context, mw @ Vector(b.co[:3]), vp
            )
            if p0 and p1:
              yield p0, p1, bool(a.select and b.select)

          if getattr(spline, 'use_cyclic_u', False) and count_u > 2:
            ia, ib = p_index(count_u - 1, v), p_index(0, v)
            if ia < len(pts) and ib < len(pts):
              a, b = pts[ia], pts[ib]
              if not getattr(a, 'hide', False) and not getattr(b, 'hide', False):
                p0 = _project_world_with_vp(
                  context, mw @ Vector(a.co[:3]), vp
                )
                p1 = _project_world_with_vp(
                  context, mw @ Vector(b.co[:3]), vp
                )
                if p0 and p1:
                  yield p0, p1, bool(a.select and b.select)

        # V direction for surfaces.
        if count_v > 1:
          for u in range(count_u):
            for v in range(count_v - 1):
              ia, ib = p_index(u, v), p_index(u, v + 1)
              if ia >= len(pts) or ib >= len(pts):
                continue
              a, b = pts[ia], pts[ib]
              if getattr(a, 'hide', False) or getattr(b, 'hide', False):
                continue
              p0 = _project_world_with_vp(
                context, mw @ Vector(a.co[:3]), vp
              )
              p1 = _project_world_with_vp(
                context, mw @ Vector(b.co[:3]), vp
              )
              if p0 and p1:
                yield p0, p1, bool(a.select and b.select)

            if getattr(spline, 'use_cyclic_v', False) and count_v > 2:
              ia, ib = p_index(u, count_v - 1), p_index(u, 0)
              if ia < len(pts) and ib < len(pts):
                a, b = pts[ia], pts[ib]
                if not getattr(a, 'hide', False) and not getattr(b, 'hide', False):
                  p0 = _project_world_with_vp(
                    context, mw @ Vector(a.co[:3]), vp
                  )
                  p1 = _project_world_with_vp(
                    context, mw @ Vector(b.co[:3]), vp
                  )
                  if p0 and p1:
                    yield p0, p1, bool(a.select and b.select)


def _clip_edge_line_vertices(
  segments,
  viewport_width,
  viewport_height,
  depth_bias,
):
  """Convert projected screen-space edge endpoints to clip coordinates.

  Blender's POLYLINE_UNIFORM_COLOR shader handles screen-space line width and
  antialiasing itself. We only provide the two clip-space endpoints per edge.
  """
  verts = []

  if viewport_width <= 0 or viewport_height <= 0:
    return verts

  def clip(px, py, pz):
    return (
      (float(px) / float(viewport_width)) * 2.0 - 1.0,
      (float(py) / float(viewport_height)) * 2.0 - 1.0,
      float(pz) - _adaptive_ndc_depth_bias(pz, depth_bias),
    )

  for p0, p1 in segments:
    x0, y0, z0 = p0
    x1, y1, z1 = p1

    # Skip degenerate sub-pixel segments.
    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    if (dx * dx + dy * dy) < 0.08:
      continue

    verts.append(clip(x0, y0, z0))
    verts.append(clip(x1, y1, z1))

  return verts


def _draw_clip_edge_batch(
  shader_unused,
  segments,
  width,
  color,
  viewport_width,
  viewport_height,
):
  """Draw smooth edit edges with Blender's own polyline shader.

  This deliberately avoids custom GLSL. Blender's built-in polyline shader
  is part of the normal GPU pipeline and is considerably safer on Metal.
  """
  if not segments:
    return

  verts = _clip_edge_line_vertices(
    segments,
    viewport_width,
    viewport_height,
    _EDGE_NDC_DEPTH_BIAS_MAX,
  )
  if not verts:
    return

  shader = _shader_from_builtin('POLYLINE_UNIFORM_COLOR')
  batch = batch_for_shader(
    shader,
    'LINES',
    {"pos": verts},
  )

  previous_blend = _gpu_blend_get()
  try:
    _gpu_blend_set('ALPHA')
    shader.bind()
    has_polyline_uniforms = True
    try:
      shader.uniform_float(
        "viewportSize",
        (float(viewport_width), float(viewport_height)),
      )
      shader.uniform_float("lineWidth", max(1.0, float(width)))
    except Exception:
      has_polyline_uniforms = False
      _gpu_line_width_set(max(1.0, float(width)))

    shader.uniform_float("color", color)
    batch.draw(shader)

    if not has_polyline_uniforms:
      _gpu_line_width_set(1.0)
  finally:
    try:
      _gpu_blend_set(previous_blend)
    except Exception:
      _gpu_blend_set('NONE')


def _draw_edit_points_into_offscreen(context, settings, offscreen, view_projection):
  """Draw stable Blender-sized edit edges and CVs using the shrunk depth buffer.

  This is the key difference from the previous always-on-top POST_PIXEL
  overlay:

  * X-Ray OFF:
    markers use the OFFSCREEN SHRUNK SCENE'S depth buffer. Backside /
    occluded CVs fail the depth test exactly where the shrunk surface is
    in front of them.

  * X-Ray ON:
    depth testing is disabled and all editable points are shown.

  The points are still triangle-built fixed-pixel discs, so their apparent
  diameter cannot collapse at shallow perspective angles.
  """
  if not _supported_edit_mode(context):
    return
  if not bool(getattr(context.space_data.overlay, "show_overlays", True)):
    return

  width = int(context.region.width)
  height = int(context.region.height)
  if width < 2 or height < 2:
    return

  normal_pos = []
  selected_pos = []

  # Edges use the exact same frame projection and depth buffer as CVs.
  normal_edges = []
  selected_edges = []
  for p0, p1, is_selected in _iter_edit_edge_segments(
    context,
    view_projection,
  ):
    if is_selected:
      selected_edges.append((p0, p1))
    else:
      normal_edges.append((p0, p1))

  for c in _iter_edit_candidates(context, settings, view_projection=view_projection):
    x = float(c["x"])
    y = float(c["y"])
    z = float(c["z"])

    if _point_inside_navigation_safe_area(context, x, y):
      continue

    # Ignore anything beyond the current clipping volume.
    if z < -1.01 or z > 1.01:
      continue

    item = (x, y, z)
    if c.get("selected", False):
      selected_pos.append(item)
    else:
      normal_pos.append(item)

  if not normal_pos and not selected_pos and not normal_edges and not selected_edges:
    return

  (
    normal_color,
    selected_color,
    inner_radius,
    outline_radius,
  ) = _theme_edit_point_style(context)

  (
    normal_edge_color,
    selected_edge_color,
    normal_edge_width,
    selected_edge_width,
  ) = _theme_edit_edge_style(context)

  shader = _shader_from_builtin("UNIFORM_COLOR")
  outline_color = (0.015, 0.015, 0.015, 1.0)

  # Re-bind the existing offscreen AFTER draw_view3d finished. The color and
  # depth attachments are preserved, so these discs are tested against the
  # exact shrunk scene that will later be composited into the real viewport.
  #
  # IMPORTANT:
  # GPU framebuffer rebinding does NOT restore viewport/scissor-like state
  # automatically, and model-view / projection matrices have separate stacks.
  # Leaking any of these into Blender's following draw calls causes visible
  # orbit flicker/glitches. Save/restore everything we modify.
  old_viewport = _gpu_viewport_get()
  old_blend = _gpu_blend_get()
  old_depth_test = _gpu_depth_test_get()
  old_depth_mask = _gpu_depth_mask_get()

  try:
    with offscreen.bind():
      _gpu_viewport_set(0, 0, width, height)

      # Direct clip coordinates: identity model-view AND projection.
      # Save/restore the matrices explicitly instead of relying on the newer
      # push_pop context-manager helpers. The get/load matrix API exists in
      # Blender 2.80 and remains available in current Blender versions.
      old_model_view = gpu.matrix.get_model_view_matrix().copy()
      old_projection = gpu.matrix.get_projection_matrix().copy()
      try:
        gpu.matrix.load_matrix(Matrix.Identity(4))
        gpu.matrix.load_projection_matrix(Matrix.Identity(4))

        if _viewport_xray_enabled(context):
          _gpu_depth_test_set('NONE')
        else:
          _gpu_depth_test_set('LESS_EQUAL')

        # Test against scene depth, but don't let the marker discs
        # rewrite the depth buffer themselves.
        _gpu_depth_mask_set(False)
        _gpu_blend_set('NONE')

        # Draw edit cage/mesh edges first so CV discs stay on top.
        # They share the scene depth test, so hidden/backside edges
        # remain hidden unless Blender X-Ray is enabled.
        if normal_edges:
          _draw_clip_edge_batch(
            shader,
            normal_edges,
            normal_edge_width,
            normal_edge_color,
            width,
            height,
          )

        if selected_edges:
          _draw_clip_edge_batch(
            shader,
            selected_edges,
            selected_edge_width,
            selected_edge_color,
            width,
            height,
          )

        if normal_pos:
          _draw_clip_disc_batch(
            shader, normal_pos, outline_radius, outline_color,
            width, height, _CV_NDC_DEPTH_BIAS_MAX,
          )
          _draw_clip_disc_batch(
            shader, normal_pos, inner_radius, normal_color,
            width, height, _CV_NDC_DEPTH_BIAS_MAX * 1.05,
          )

        if selected_pos:
          _draw_clip_disc_batch(
            shader, selected_pos, outline_radius, outline_color,
            width, height, _CV_NDC_DEPTH_BIAS_MAX,
          )
          _draw_clip_disc_batch(
            shader, selected_pos, inner_radius, selected_color,
            width, height, _CV_NDC_DEPTH_BIAS_MAX * 1.05,
          )
      finally:
        gpu.matrix.load_projection_matrix(old_projection)
        gpu.matrix.load_matrix(old_model_view)

  finally:
    # OffScreenStackContext restores the framebuffer, but Blender's GPU API
    # explicitly does not guarantee viewport state restoration after a
    # framebuffer rebind. Restore it ourselves.
    try:
      _gpu_viewport_set(*old_viewport)
    except Exception:
      pass

    try:
      _gpu_blend_set(old_blend)
    except Exception:
      _gpu_blend_set('NONE')

    try:
      _gpu_depth_test_set(old_depth_test)
    except Exception:
      _gpu_depth_test_set('NONE')

    try:
      _gpu_depth_mask_set(old_depth_mask)
    except Exception:
      _gpu_depth_mask_set(True)


def _render_offscreen(context, settings, offscreen):
  projection, view_projection = _custom_projection_matrices(context, settings)
  view_matrix = context.region_data.view_matrix.copy()

  _offscreen_draw_view3d(
    offscreen,
    context,
    view_matrix,
    projection,
  )

  # Use the EXACT same view-projection matrix as the scene frame above.
  # This prevents a CV/edge frame from being projected against a slightly
  # newer orbit matrix while the color/depth frame still belongs to the
  # previous one.
  #
  # IMPORTANT: edit-overlay rendering is intentionally isolated. If Blender
  # rejects an overlay GPU call on a particular driver/frame, the SHRUNK
  # scene render is still valid and must remain visible.
  try:
    _draw_edit_points_into_offscreen(
      context,
      settings,
      offscreen,
      view_projection,
    )
  except Exception as exc:
    print("Viewport Shrink edit-overlay render warning:", exc)

  return view_projection



def _safe_render_cache_target(
  context,
  settings,
  entry,
  target_key,
  signature,
):
  """Render a new shrink frame without ever dropping the previous one.

  Returns True only after a complete new shrunk scene frame exists. On any
  GPU/offscreen exception, `active_texture` and `display_vp` remain untouched,
  so the viewport continues showing the last valid SHRUNK frame rather than
  exposing Blender's normal unshrunk viewport.
  """
  try:
    offscreen = entry[target_key]
    vp = _render_offscreen(
      context,
      settings,
      offscreen,
    )

    # Commit atomically only AFTER render success.
    entry["display_vp"] = vp.copy()
    entry["active_texture"] = _offscreen_color_texture(offscreen)
    entry["source_signature"] = signature
    entry["source_dirty_serial"] = _DIRTY_SERIAL
    return True

  except Exception as exc:
    print("Viewport Shrink frame render warning:", exc)
    return False


def _schedule_settle_redraw():
  """Request one full-resolution refresh after interaction becomes idle.

  The timer only lives while a low-resolution interactive frame is waiting
  to settle. It does not continuously redraw an idle viewport.
  """
  global _SETTLE_TIMER_ACTIVE

  if _SETTLE_TIMER_ACTIVE:
    return

  _SETTLE_TIMER_ACTIVE = True

  def _settle_timer():
    global _SETTLE_TIMER_ACTIVE

    if _DRAW_HANDLE is None:
      _SETTLE_TIMER_ACTIVE = False
      return None

    now = time.perf_counter()
    pending = False
    ready = False

    for entry in list(_OFFSCREENS.values()):
      if not entry.get("needs_full", False):
        continue
      pending = True
      if (now - entry.get("last_change", 0.0)) >= _SETTLE_DELAY:
        ready = True

    if ready:
      # One redraw is enough; _draw_viewport_shrink will promote each
      # settled region from the fast texture to its native-resolution texture.
      _tag_redraw_all_view3d()

    # Continue briefly only while at least one region still needs its sharp frame.
    if pending:
      return 0.025

    _SETTLE_TIMER_ACTIVE = False
    return None

  # Application timers are available in modern 2.8x+ builds, but guard the
  # lookup so an early/limited 2.80 build still keeps the core viewport tool
  # functional instead of failing during interaction.
  timers = getattr(bpy.app, 'timers', None)
  if timers is None or not hasattr(timers, 'register'):
    _SETTLE_TIMER_ACTIVE = False
    return

  try:
    timers.register(_settle_timer, first_interval=_SETTLE_DELAY)
  except Exception:
    _SETTLE_TIMER_ACTIVE = False


def _requires_native_resolution(context):
  """Keep edit-mode control points/vertices at a constant apparent size.

  A reduced-resolution offscreen is useful for object-mode navigation, but
  upscaling it also upscales Blender's edit-point pixels. That makes CVs and
  mesh vertices briefly look larger while panning/zooming.

  In Curve/Surface/Mesh Edit Mode we therefore render directly at the real
  viewport resolution. Static frames are still cached, so the full render is
  only repeated while the view or geometry is actually changing.
  """
  return (
    context.mode in {'EDIT_CURVE', 'EDIT_SURFACE', 'EDIT_MESH'}
    and bool(getattr(context.space_data.overlay, "show_overlays", True))
  )


def _draw_viewport_shrink():
  global _DRAW_GUARD

  if _DRAW_GUARD:
    return

  context = bpy.context
  if not context.area or context.area.type != 'VIEW_3D':
    return
  if not context.region or context.region.type != 'WINDOW' or not context.region_data:
    return

  settings = _settings(context)
  if not _is_shrunk(settings):
    return

  width, height = int(context.region.width), int(context.region.height)
  if width < 8 or height < 8:
    return

  entry = _get_cache_entry(context, width, height)
  if not entry:
    return

  _DRAW_GUARD = True
  try:
    now = time.perf_counter()
    signature = _draw_signature(context, settings)

    source_changed = (
      signature != entry["source_signature"]
      or entry["source_dirty_serial"] != _DIRTY_SERIAL
      or entry["active_texture"] is None
    )

    if source_changed:
      # IMPORTANT:
      # Do not mark the cache signature as updated until the replacement
      # SHRUNK frame has actually rendered successfully.
      use_native = (
        _requires_native_resolution(context)
        or now < entry.get("force_full_until", 0.0)
      )

      if use_native:
        rendered = _safe_render_cache_target(
          context,
          settings,
          entry,
          "full",
          signature,
        )
        if rendered:
          entry["last_change"] = now
          entry["needs_full"] = False
      else:
        # Object-mode/navigation optimization only.
        rendered = _safe_render_cache_target(
          context,
          settings,
          entry,
          "fast",
          signature,
        )
        if rendered:
          entry["last_change"] = now
          entry["needs_full"] = True
          _schedule_settle_redraw()

    elif entry["needs_full"] and (now - entry["last_change"]) >= _SETTLE_DELAY:
      # Idle path: one sharp native-resolution render, then reuse it.
      rendered = _safe_render_cache_target(
        context,
        settings,
        entry,
        "full",
        signature,
      )
      if rendered:
        entry["needs_full"] = False

    _gpu_depth_test_set('NONE')
    _gpu_depth_mask_set(False)
    _gpu_blend_set('NONE')

    if entry["active_texture"] is not None:
      _draw_shrink_texture_fullscreen(entry["active_texture"], width, height)

    # CVs are already embedded into the offscreen render with correct
    # shrink-aware depth occlusion. Keep only the 2D selection box here.
    _draw_box_overlay()

  except Exception as exc:
    print(f"Viewport Shrink draw error: {exc}")
  finally:
    try:
      _gpu_blend_set('NONE')
      _gpu_depth_mask_set(True)
    except Exception:
      pass
    _DRAW_GUARD = False


def _depsgraph_dirty_handler(scene, depsgraph):
  """Invalidate only for updates that can alter the visible 3D result.

  The previous build dirtied the offscreen for every depsgraph notification,
  which could cause an unnecessary second full viewport render over and over.
  """
  relevant_types = (
    bpy.types.Object,
    bpy.types.Mesh,
    bpy.types.Curve,
    bpy.types.MetaBall,
    bpy.types.Material,
    bpy.types.Image,
    bpy.types.Light,
    bpy.types.World,
  )

  try:
    for update in depsgraph.updates:
      id_data = getattr(update, "id", None)
      if id_data is None or not isinstance(id_data, relevant_types):
        continue

      # Object/geometry updates: only redraw for actual transform/geometry changes.
      if isinstance(id_data, (bpy.types.Object, bpy.types.Mesh, bpy.types.Curve, bpy.types.MetaBall)):
        if (getattr(update, "is_updated_geometry", False)
            or getattr(update, "is_updated_transform", False)):
          _mark_dirty()
          return
      else:
        # Materials, images, lights and world updates can directly change pixels.
        _mark_dirty()
        return
  except Exception:
    # Fallback is intentionally conservative but still avoids a permanent redraw loop.
    pass


def _frame_dirty_handler(*_args):
  _mark_dirty()


# -----------------------------------------------------------------------------
# Properties
# -----------------------------------------------------------------------------


def _find_custom_preset_kmi(context, slot_uid):
  """Find the user keymap item belonging to one custom preset row."""
  if not slot_uid:
    return None

  km = _user_3d_view_keymap(context, create=False)
  if km is None:
    return None

  for kmi in km.keymap_items:
    if kmi.idname != VIEWPORTSHRINK_OT_AxisPreset.bl_idname:
      continue

    try:
      if str(kmi.properties.slot_uid) == str(slot_uid):
        return kmi
    except Exception:
      continue

  return None


def _on_custom_preset_changed(self, context):
  """Keep the assigned keyboard shortcut live-linked to this slider."""
  uid = str(getattr(self, "uid", "") or "")
  if not uid:
    return

  kmi = _find_custom_preset_kmi(context, uid)
  if kmi is None:
    return

  try:
    kmi.properties.axis = str(self.axis).upper()
    kmi.properties.value = max(
      0.10,
      min(1.0, float(self.value)),
    )
  except Exception:
    pass


class VIEWPORTSHRINK_PG_CustomPresetShortcut(bpy.types.PropertyGroup):
  uid: StringProperty(
    name="ID",
    default="",
    options={'HIDDEN'},
  )

  axis: StringProperty(
    name="Axis",
    default="X",
    options={'HIDDEN'},
    update=_on_custom_preset_changed,
  )

  value: FloatProperty(
    name="Shrink",
    description="Custom viewport shrink value used by this shortcut",
    default=0.50,
    min=0.10,
    max=1.0,
    soft_min=0.10,
    soft_max=1.0,
    precision=2,
    subtype='FACTOR',
    update=_on_custom_preset_changed,
  )


class VIEWPORTSHRINK_PG_Settings(bpy.types.PropertyGroup):
  shrink_x: FloatProperty(
    name="X",
    description="Visually compress the viewport along world X without changing scene geometry",
    default=1.0, min=0.05, max=1.0, soft_min=0.10, soft_max=1.0,
    precision=2, subtype='FACTOR', update=_on_setting_changed,
  )
  shrink_y: FloatProperty(
    name="Y",
    description="Visually compress the viewport along world Y without changing scene geometry",
    default=1.0, min=0.05, max=1.0, soft_min=0.10, soft_max=1.0,
    precision=2, subtype='FACTOR', update=_on_setting_changed,
  )
  shrink_z: FloatProperty(
    name="Z",
    description="Visually compress the viewport along world Z without changing scene geometry",
    default=1.0, min=0.05, max=1.0, soft_min=0.10, soft_max=1.0,
    precision=2, subtype='FACTOR', update=_on_setting_changed,
  )
  lock_x: BoolProperty(
    name="Lock X", description="Keep X unchanged when using presets",
    default=False, update=_on_setting_changed,
  )
  lock_y: BoolProperty(
    name="Lock Y", description="Keep Y unchanged when using presets",
    default=True, update=_on_setting_changed,
  )
  lock_z: BoolProperty(
    name="Lock Z", description="Keep Z unchanged when using presets",
    default=True, update=_on_setting_changed,
  )

  custom_preset_shortcuts: CollectionProperty(
    type=VIEWPORTSHRINK_PG_CustomPresetShortcut,
  )


# -----------------------------------------------------------------------------
# Presets
# -----------------------------------------------------------------------------

class _VIEWPORTSHRINK_OT_PresetBase:
  preset_value = 1.0

  def execute(self, context):
    settings = _settings(context)
    if not settings:
      return {'CANCELLED'}
    if not settings.lock_x:
      settings.shrink_x = self.preset_value
    if not settings.lock_y:
      settings.shrink_y = self.preset_value
    if not settings.lock_z:
      settings.shrink_z = self.preset_value
    _mark_dirty()
    _tag_redraw_all_view3d()
    return {'FINISHED'}


class VIEWPORTSHRINK_OT_Preset03(_VIEWPORTSHRINK_OT_PresetBase, bpy.types.Operator):
  bl_idname = "viewport_shrink.preset_03"
  bl_label = "Viewport Shrink .3"
  bl_description = "Set each unlocked world axis to 0.3"
  bl_options = {'REGISTER'}
  preset_value = 0.3


class VIEWPORTSHRINK_OT_Preset05(_VIEWPORTSHRINK_OT_PresetBase, bpy.types.Operator):
  bl_idname = "viewport_shrink.preset_05"
  bl_label = "Viewport Shrink .5"
  bl_description = "Set each unlocked world axis to 0.5"
  bl_options = {'REGISTER'}
  preset_value = 0.5


class VIEWPORTSHRINK_OT_Preset10(_VIEWPORTSHRINK_OT_PresetBase, bpy.types.Operator):
  bl_idname = "viewport_shrink.preset_10"
  bl_label = "Viewport Shrink 1"
  bl_description = "Set each unlocked world axis to 1.0"
  bl_options = {'REGISTER'}
  preset_value = 1.0


class VIEWPORTSHRINK_OT_AxisPreset(bpy.types.Operator):
  bl_idname = "viewport_shrink.axis_preset"
  bl_label = "Viewport Shrink Axis Preset"
  bl_description = "Set a preset value on one viewport shrink axis"
  bl_options = {'REGISTER'}

  axis: StringProperty(options={'HIDDEN'})
  value: FloatProperty(options={'HIDDEN'})
  slot_uid: StringProperty(default="", options={'HIDDEN'})

  def execute(self, context):
    s = _settings(context)
    if not s:
      return {'CANCELLED'}

    axis = self.axis.upper()
    mapping = {
      'X': ('shrink_x', 'lock_x'),
      'Y': ('shrink_y', 'lock_y'),
      'Z': ('shrink_z', 'lock_z'),
    }
    item = mapping.get(axis)
    if item is None:
      return {'CANCELLED'}

    value_prop, lock_prop = item
    if getattr(s, lock_prop):
      self.report({'INFO'}, f"{axis} is locked")
      return {'CANCELLED'}

    setattr(s, value_prop, max(0.05, min(1.0, float(self.value))))
    _mark_dirty()
    _tag_redraw_all_view3d()
    return {'FINISHED'}



# -----------------------------------------------------------------------------
# User shortcut assignment
# -----------------------------------------------------------------------------

def _shortcut_value_key(value):
  value = float(value)
  if abs(value - 0.3) < 1e-4:
    return 0.3
  if abs(value - 0.5) < 1e-4:
    return 0.5
  return 1.0


def _shortcut_target_matches(kmi, axis, value):
  """True when a user keymap item belongs to one of our 9 axis presets."""
  if kmi.idname != VIEWPORTSHRINK_OT_AxisPreset.bl_idname:
    return False

  try:
    item_axis = str(kmi.properties.axis).upper()
    item_value = float(kmi.properties.value)
  except Exception:
    return False

  return (
    item_axis == str(axis).upper()
    and abs(item_value - _shortcut_value_key(value)) < 1e-4
  )


def _user_3d_view_keymap(context, create=False):
  wm = context.window_manager
  if not wm or not wm.keyconfigs or not wm.keyconfigs.user:
    return None

  kc = wm.keyconfigs.user
  km = kc.keymaps.get('3D View')

  if km is None and create:
    km = kc.keymaps.new(
      name='3D View',
      space_type='VIEW_3D',
      region_type='WINDOW',
    )

  return km


def _remove_axis_shortcut(context, axis, value):
  """Remove the currently assigned shortcut for one axis/value target."""
  km = _user_3d_view_keymap(context, create=False)
  if km is None:
    return False

  removed = False

  for kmi in list(km.keymap_items):
    if _shortcut_target_matches(kmi, axis, value):
      try:
        km.keymap_items.remove(kmi)
        removed = True
      except Exception:
        pass

  return removed


def _find_axis_shortcut(context, axis, value):
  """Return the user keymap item currently assigned to this preset."""
  km = _user_3d_view_keymap(context, create=False)
  if km is None:
    return None

  for kmi in km.keymap_items:
    if _shortcut_target_matches(kmi, axis, value):
      return kmi

  return None


def _shortcut_display_from_kmi(kmi):
  if kmi is None:
    return "Not set"

  parts = []

  try:
    if kmi.ctrl:
      parts.append("Ctrl")
    if kmi.shift:
      parts.append("Shift")
    if kmi.alt:
      parts.append("Alt")
    if kmi.oskey:
      parts.append("Cmd" if _is_macos() else "OS")
  except Exception:
    pass

  try:
    key = str(kmi.type).replace('_', ' ').title()
  except Exception:
    key = "Key"

  parts.append(key)
  return " + ".join(parts)


def _shortcut_display_from_event(event):
  parts = []

  if event.ctrl:
    parts.append("Ctrl")
  if event.shift:
    parts.append("Shift")
  if event.alt:
    parts.append("Alt")
  if event.oskey:
    parts.append("Cmd" if _is_macos() else "OS")

  key = str(event.type).replace('_', ' ').title()
  parts.append(key)

  return " + ".join(parts)


def _shortcut_event_is_assignable(event):
  """Accept keyboard-like PRESS events; reject navigation/mouse noise."""
  if event.value != 'PRESS':
    return False

  rejected = {
    'NONE',
    'MOUSEMOVE',
    'INBETWEEN_MOUSEMOVE',
    'LEFTMOUSE',
    'MIDDLEMOUSE',
    'RIGHTMOUSE',
    'BUTTON4MOUSE',
    'BUTTON5MOUSE',
    'BUTTON6MOUSE',
    'BUTTON7MOUSE',
    'WHEELUPMOUSE',
    'WHEELDOWNMOUSE',
    'WHEELINMOUSE',
    'WHEELOUTMOUSE',
    'TRACKPADPAN',
    'TRACKPADZOOM',
    'MOUSEROTATE',
    'MOUSESMARTZOOM',
    'TIMER',
    'TIMER0',
    'TIMER1',
    'TIMER2',
    'TIMER_JOBS',
    'TIMER_AUTOSAVE',
    'WINDOW_DEACTIVATE',
    'LEFT_CTRL',
    'RIGHT_CTRL',
    'LEFT_SHIFT',
    'RIGHT_SHIFT',
    'LEFT_ALT',
    'RIGHT_ALT',
    'OSKEY',
  }

  event_type = str(event.type)

  if event_type in rejected:
    return False

  if event_type.startswith('NDOF_'):
    return False

  return True



def _custom_preset_slot_by_uid(settings, uid):
  if settings is None or not uid:
    return None

  for item in settings.custom_preset_shortcuts:
    if str(item.uid) == str(uid):
      return item

  return None


def _custom_preset_kmi_uid(kmi):
  try:
    return str(kmi.properties.slot_uid or "")
  except Exception:
    return ""


def _ensure_custom_preset_rows(context):
  """Synchronize dynamic UI rows from persistent user keymap entries.

  Older v22-v30 fixed .3/.5/1 axis shortcuts are migrated automatically:
  they receive a slot UID and appear as editable custom rows.
  """
  settings = _settings(context)
  if settings is None:
    return

  km = _user_3d_view_keymap(context, create=False)
  if km is None:
    return

  existing = {
    str(item.uid): item
    for item in settings.custom_preset_shortcuts
    if str(item.uid)
  }

  for kmi in km.keymap_items:
    if kmi.idname != VIEWPORTSHRINK_OT_AxisPreset.bl_idname:
      continue

    try:
      axis = str(kmi.properties.axis).upper()
      value = float(kmi.properties.value)
    except Exception:
      continue

    if axis not in {'X', 'Y', 'Z'}:
      continue

    uid = _custom_preset_kmi_uid(kmi)

    # Migrate old fixed preset shortcut items into the new custom system.
    if not uid:
      uid = uuid.uuid4().hex
      try:
        kmi.properties.slot_uid = uid
      except Exception:
        continue

    item = existing.get(uid)
    if item is None:
      item = settings.custom_preset_shortcuts.add()
      item.uid = uid
      item.axis = axis
      item.value = max(0.10, min(1.0, value))
      existing[uid] = item


def _remove_custom_preset_kmi(context, uid):
  km = _user_3d_view_keymap(context, create=False)
  if km is None:
    return False

  removed = False

  for kmi in list(km.keymap_items):
    if (
      kmi.idname == VIEWPORTSHRINK_OT_AxisPreset.bl_idname
      and _custom_preset_kmi_uid(kmi) == str(uid)
    ):
      try:
        km.keymap_items.remove(kmi)
        removed = True
      except Exception:
        pass

  return removed


def _custom_preset_current_shortcut(context, uid):
  return _shortcut_compact_display(
    _find_custom_preset_kmi(context, uid)
  )


class VIEWPORTSHRINK_OT_AddCustomPresetShortcut(bpy.types.Operator):
  bl_idname = "viewport_shrink.add_custom_preset_shortcut"
  bl_label = "Add Custom Shortcut"
  bl_description = "Add a custom shrink value and shortcut for this axis"
  bl_options = {'INTERNAL'}

  axis: StringProperty(options={'HIDDEN'})

  def execute(self, context):
    settings = _settings(context)
    if settings is None:
      return {'CANCELLED'}

    axis = str(self.axis).upper()
    if axis not in {'X', 'Y', 'Z'}:
      return {'CANCELLED'}

    item = settings.custom_preset_shortcuts.add()
    item.uid = uuid.uuid4().hex
    item.axis = axis

    # Start from the axis's current viewport shrink value.
    value_prop = {
      'X': 'shrink_x',
      'Y': 'shrink_y',
      'Z': 'shrink_z',
    }[axis]
    item.value = max(
      0.10,
      min(1.0, float(getattr(settings, value_prop))),
    )

    try:
      context.area.tag_redraw()
    except Exception:
      pass

    return {'FINISHED'}


class VIEWPORTSHRINK_OT_RemoveCustomPresetShortcut(bpy.types.Operator):
  bl_idname = "viewport_shrink.remove_custom_preset_shortcut"
  bl_label = "Remove Custom Shortcut"
  bl_description = "Remove this custom shrink shortcut"
  bl_options = {'INTERNAL'}

  slot_uid: StringProperty(options={'HIDDEN'})

  def execute(self, context):
    settings = _settings(context)
    if settings is None:
      return {'CANCELLED'}

    uid = str(self.slot_uid)
    _remove_custom_preset_kmi(context, uid)

    for index, item in enumerate(settings.custom_preset_shortcuts):
      if str(item.uid) == uid:
        settings.custom_preset_shortcuts.remove(index)
        break

    try:
      context.area.tag_redraw()
    except Exception:
      pass

    return {'FINISHED'}


class VIEWPORTSHRINK_OT_CaptureCustomPresetShortcut(bpy.types.Operator):
  bl_idname = "viewport_shrink.capture_custom_preset_shortcut"
  bl_label = "Set Shortcut"
  bl_description = "Assign a keyboard shortcut to this custom shrink value"
  bl_options = {'INTERNAL'}

  slot_uid: StringProperty(options={'HIDDEN'})

  @classmethod
  def description(cls, context, properties):
    try:
      settings = _settings(context)
      item = _custom_preset_slot_by_uid(
        settings,
        properties.slot_uid,
      )
      if item is None:
        return "Set keyboard shortcut"

      current = _custom_preset_current_shortcut(
        context,
        item.uid,
      )
      return (
        f"Set shortcut for {item.axis} = {item.value:.2f}\n"
        f"Current: {current}\n"
        "Esc: Cancel  Backspace/Delete: Clear"
      )
    except Exception:
      return "Set keyboard shortcut"

  def _clear_header(self, context):
    try:
      if context.area:
        context.area.header_text_set(None)
    except Exception:
      pass

  def invoke(self, context, event):
    settings = _settings(context)
    item = _custom_preset_slot_by_uid(
      settings,
      self.slot_uid,
    )
    if item is None:
      return {'CANCELLED'}

    try:
      if context.area:
        context.area.header_text_set(
          f"Set shortcut: {item.axis} = {item.value:.2f}  "
          "Press keys • Backspace/Delete clears • Esc cancels"
        )
    except Exception:
      pass

    context.window_manager.modal_handler_add(self)
    return {'RUNNING_MODAL'}

  def modal(self, context, event):
    settings = _settings(context)
    item = _custom_preset_slot_by_uid(
      settings,
      self.slot_uid,
    )
    if item is None:
      self._clear_header(context)
      return {'CANCELLED'}

    if event.type == 'ESC' and event.value == 'PRESS':
      self._clear_header(context)
      self.report({'INFO'}, "Shortcut assignment cancelled")
      return {'CANCELLED'}

    if event.type in {'BACK_SPACE', 'DEL'} and event.value == 'PRESS':
      removed = _remove_custom_preset_kmi(
        context,
        item.uid,
      )
      self._clear_header(context)

      if removed:
        self.report(
          {'INFO'},
          f"Cleared shortcut for {item.axis} = {item.value:.2f}",
        )
      else:
        self.report(
          {'INFO'},
          "No shortcut was assigned",
        )

      try:
        _tag_redraw_all_view3d()
      except Exception:
        pass

      return {'FINISHED'}

    if not _shortcut_event_is_assignable(event):
      return {'RUNNING_MODAL'}

    km = _user_3d_view_keymap(context, create=True)
    if km is None:
      self._clear_header(context)
      self.report({'ERROR'}, "User keymap is unavailable")
      return {'CANCELLED'}

    _remove_custom_preset_kmi(
      context,
      item.uid,
    )

    try:
      kmi = km.keymap_items.new(
        VIEWPORTSHRINK_OT_AxisPreset.bl_idname,
        type=event.type,
        value='PRESS',
        ctrl=bool(event.ctrl),
        shift=bool(event.shift),
        alt=bool(event.alt),
        oskey=bool(event.oskey),
      )

      kmi.properties.axis = str(item.axis).upper()
      kmi.properties.value = float(item.value)
      kmi.properties.slot_uid = str(item.uid)

    except Exception as exc:
      self._clear_header(context)
      self.report(
        {'ERROR'},
        f"Could not assign shortcut: {exc}",
      )
      return {'CANCELLED'}

    shortcut_name = _shortcut_display_from_event(event)
    self._clear_header(context)
    self.report(
      {'INFO'},
      f"{item.axis} = {item.value:.2f} → {shortcut_name}",
    )

    try:
      _tag_redraw_all_view3d()
    except Exception:
      pass

    return {'FINISHED'}


class VIEWPORTSHRINK_OT_CaptureShortcut(bpy.types.Operator):
  bl_idname = "viewport_shrink.capture_shortcut"
  bl_label = "Set Viewport Shrink Shortcut"
  bl_description = (
    "Choose this preset, then press the keyboard shortcut you want"
  )
  bl_options = {'INTERNAL'}

  axis: StringProperty(options={'HIDDEN'})
  value: FloatProperty(options={'HIDDEN'})

  @classmethod
  def description(cls, context, properties):
    try:
      axis = str(properties.axis).upper()
      value = _shortcut_value_key(properties.value)
      current = _shortcut_display_from_kmi(
        _find_axis_shortcut(context, axis, value)
      )
      return (
        f"Set shortcut for {axis} = {value:g}\n"
        f"Current: {current}\n"
        "Esc: Cancel  Backspace/Delete: Clear"
      )
    except Exception:
      return "Set keyboard shortcut for this viewport shrink preset"

  def _clear_header(self, context):
    try:
      if context.area:
        context.area.header_text_set(None)
    except Exception:
      pass

  def invoke(self, context, event):
    axis = str(self.axis).upper()
    value = _shortcut_value_key(self.value)

    if axis not in {'X', 'Y', 'Z'}:
      return {'CANCELLED'}

    self.axis = axis
    self.value = value

    try:
      if context.area:
        context.area.header_text_set(
          f"Set shortcut: {axis} = {value:g}  "
          "Press keys • Backspace/Delete clears • Esc cancels"
        )
    except Exception:
      pass

    context.window_manager.modal_handler_add(self)
    return {'RUNNING_MODAL'}

  def modal(self, context, event):
    if event.type == 'ESC' and event.value == 'PRESS':
      self._clear_header(context)
      self.report({'INFO'}, "Shortcut assignment cancelled")
      return {'CANCELLED'}

    if event.type in {'BACK_SPACE', 'DEL'} and event.value == 'PRESS':
      removed = _remove_axis_shortcut(
        context,
        self.axis,
        self.value,
      )
      self._clear_header(context)

      if removed:
        self.report(
          {'INFO'},
          f"Cleared shortcut for {self.axis} = {self.value:g}",
        )
      else:
        self.report(
          {'INFO'},
          f"No shortcut was set for {self.axis} = {self.value:g}",
        )

      return {'FINISHED'}

    if not _shortcut_event_is_assignable(event):
      return {'RUNNING_MODAL'}

    km = _user_3d_view_keymap(context, create=True)
    if km is None:
      self._clear_header(context)
      self.report({'ERROR'}, "User keymap is unavailable")
      return {'CANCELLED'}

    # One shortcut per target. Reassigning is intentionally effortless.
    _remove_axis_shortcut(
      context,
      self.axis,
      self.value,
    )

    try:
      kmi = km.keymap_items.new(
        VIEWPORTSHRINK_OT_AxisPreset.bl_idname,
        type=event.type,
        value='PRESS',
        ctrl=bool(event.ctrl),
        shift=bool(event.shift),
        alt=bool(event.alt),
        oskey=bool(event.oskey),
      )

      kmi.properties.axis = self.axis
      kmi.properties.value = float(self.value)

    except Exception as exc:
      self._clear_header(context)
      self.report({'ERROR'}, f"Could not assign shortcut: {exc}")
      return {'CANCELLED'}

    shortcut_name = _shortcut_display_from_event(event)
    self._clear_header(context)

    self.report(
      {'INFO'},
      f"{self.axis} = {self.value:g} → {shortcut_name}",
    )

    return {'FINISHED'}



def _slider_shortcut_target_matches(kmi, axis):
  if kmi.idname != VIEWPORTSHRINK_OT_SliderGesture.bl_idname:
    return False

  try:
    return str(kmi.properties.axis).upper() == str(axis).upper()
  except Exception:
    return False


def _find_slider_shortcut(context, axis):
  km = _user_3d_view_keymap(context, create=False)
  if km is None:
    return None

  for kmi in km.keymap_items:
    if _slider_shortcut_target_matches(kmi, axis):
      return kmi

  return None


def _remove_slider_shortcut(context, axis):
  km = _user_3d_view_keymap(context, create=False)
  if km is None:
    return False

  removed = False

  for kmi in list(km.keymap_items):
    if _slider_shortcut_target_matches(kmi, axis):
      try:
        km.keymap_items.remove(kmi)
        removed = True
      except Exception:
        pass

  return removed


def _cleanup_legacy_slider_mouse_shortcuts(context):
  """Remove old slider shortcuts that were bound directly to mouse buttons."""
  km = _user_3d_view_keymap(context, create=False)
  if km is None:
    return

  mouse_types = {
    'LEFTMOUSE',
    'MIDDLEMOUSE',
    'RIGHTMOUSE',
    'BUTTON4MOUSE',
    'BUTTON5MOUSE',
    'BUTTON6MOUSE',
    'BUTTON7MOUSE',
  }

  for kmi in list(km.keymap_items):
    if (
      kmi.idname == VIEWPORTSHRINK_OT_SliderGesture.bl_idname
      and str(kmi.type) in mouse_types
    ):
      try:
        km.keymap_items.remove(kmi)
      except Exception:
        pass


def _slider_axis_props(axis):
  axis = str(axis).upper()
  mapping = {
    'X': ('shrink_x', 'lock_x'),
    'Y': ('shrink_y', 'lock_y'),
    'Z': ('shrink_z', 'lock_z'),
  }
  return mapping.get(axis)


# -----------------------------------------------------------------------------
# Slider drag shortcuts: 1–4 keyboard buttons + LMB
# -----------------------------------------------------------------------------

_SLIDER_CHORD_MAX_KEYS = 4

_MODIFIER_GROUPS = {
  'CTRL': {'LEFT_CTRL', 'RIGHT_CTRL'},
  'SHIFT': {'LEFT_SHIFT', 'RIGHT_SHIFT'},
  'ALT': {'LEFT_ALT', 'RIGHT_ALT'},
  'OSKEY': {'OSKEY', 'LEFT_OSKEY', 'RIGHT_OSKEY'},
}


def _encode_slider_chord(keys):
  clean = []
  for key in keys:
    key = str(key)
    if key and key not in clean:
      clean.append(key)
    if len(clean) >= _SLIDER_CHORD_MAX_KEYS:
      break
  return "|".join(clean)


def _decode_slider_chord(value):
  if not value:
    return []

  result = []
  for key in str(value).split("|"):
    key = key.strip()
    if key and key not in result:
      result.append(key)
    if len(result) >= _SLIDER_CHORD_MAX_KEYS:
      break
  return result


def _pretty_chord_key(key):
  key = str(key)

  if key in _MODIFIER_GROUPS['CTRL']:
    return "Ctrl"
  if key in _MODIFIER_GROUPS['SHIFT']:
    return "Shift"
  if key in _MODIFIER_GROUPS['ALT']:
    return "Alt"
  if key in _MODIFIER_GROUPS['OSKEY']:
    return "Cmd" if _is_macos() else "OS"

  aliases = {
    'LEFTMOUSE': 'LMB',
    'MIDDLEMOUSE': 'MMB',
    'RIGHTMOUSE': 'RMB',
    'BACK_SPACE': 'Backspace',
    'RET': 'Enter',
    'NUMPAD_ENTER': 'Numpad Enter',
    'SPACE': 'Space',
    'PAGE_UP': 'Page Up',
    'PAGE_DOWN': 'Page Down',
    'UP_ARROW': 'Up',
    'DOWN_ARROW': 'Down',
    'LEFT_ARROW': 'Left',
    'RIGHT_ARROW': 'Right',
    'ACCENT_GRAVE': '`',
    'SEMI_COLON': ';',
  }

  if key in aliases:
    return aliases[key]

  if key.startswith('NUMPAD_'):
    return "Numpad " + key[len('NUMPAD_'):].replace('_', ' ').title()

  return key.replace('_', ' ').title()


def _pretty_slider_chord(keys, include_lmb=True):
  keys = _decode_slider_chord(_encode_slider_chord(keys))
  if not keys:
    return "Not set"

  parts = [_pretty_chord_key(key) for key in keys]
  if include_lmb:
    parts.append("LMB")
  return " + ".join(parts)


def _event_is_keyboard_button(event):
  if event.value != 'PRESS':
    return False

  event_type = str(event.type)

  rejected = {
    'NONE',
    'MOUSEMOVE',
    'INBETWEEN_MOUSEMOVE',
    'LEFTMOUSE',
    'MIDDLEMOUSE',
    'RIGHTMOUSE',
    'BUTTON4MOUSE',
    'BUTTON5MOUSE',
    'BUTTON6MOUSE',
    'BUTTON7MOUSE',
    'WHEELUPMOUSE',
    'WHEELDOWNMOUSE',
    'WHEELINMOUSE',
    'WHEELOUTMOUSE',
    'TRACKPADPAN',
    'TRACKPADZOOM',
    'MOUSEROTATE',
    'MOUSESMARTZOOM',
    'TIMER',
    'TIMER0',
    'TIMER1',
    'TIMER2',
    'TIMER_JOBS',
    'TIMER_AUTOSAVE',
    'WINDOW_DEACTIVATE',
  }

  if event_type in rejected:
    return False

  if event_type.startswith('NDOF_'):
    return False

  return True


def _modifier_group_for_key(key):
  for group_name, members in _MODIFIER_GROUPS.items():
    if key in members:
      return group_name
  return None


def _chord_key_is_held(key, held_keys, event=None):
  """Test one required key, treating left/right modifiers as equivalent."""
  group = _modifier_group_for_key(key)

  if group is not None:
    members = _MODIFIER_GROUPS[group]
    if any(member in held_keys for member in members):
      return True

    # Blender also supplies generic modifier flags on unrelated events.
    if event is not None:
      if group == 'CTRL' and bool(getattr(event, 'ctrl', False)):
        return True
      if group == 'SHIFT' and bool(getattr(event, 'shift', False)):
        return True
      if group == 'ALT' and bool(getattr(event, 'alt', False)):
        return True
      if group == 'OSKEY' and bool(getattr(event, 'oskey', False)):
        return True

    return False

  return key in held_keys


def _slider_chord_is_held(chord, held_keys, event=None):
  if not chord:
    return False

  return all(
    _chord_key_is_held(key, held_keys, event)
    for key in chord
  )


def _seed_modifier_keys_from_event(chord, held_keys, event):
  """Recognize modifiers that were already held before the trigger key."""
  for key in chord:
    group = _modifier_group_for_key(key)

    if group == 'CTRL' and bool(getattr(event, 'ctrl', False)):
      held_keys.add(key)
    elif group == 'SHIFT' and bool(getattr(event, 'shift', False)):
      held_keys.add(key)
    elif group == 'ALT' and bool(getattr(event, 'alt', False)):
      held_keys.add(key)
    elif group == 'OSKEY' and bool(getattr(event, 'oskey', False)):
      held_keys.add(key)


class VIEWPORTSHRINK_OT_SliderGesture(bpy.types.Operator):
  """Hold a 1–4 key chord, then LMB-drag to change one shrink axis."""

  bl_idname = "viewport_shrink.slider_gesture"
  bl_label = "Viewport Shrink Slider Drag"
  bl_description = (
    "Hold the assigned keyboard chord and drag LMB toward the viewport "
    "center to shrink, or outward to expand"
  )
  bl_options = {'INTERNAL', 'BLOCKING'}

  axis: StringProperty(options={'HIDDEN'})
  chord: StringProperty(default="", options={'HIDDEN'})

  _chord_keys = None
  _held_keys = None
  _trigger_key = None

  _waiting_for_lmb = True
  _dragging = False

  _start_mouse = None
  _start_value = 1.0
  _radial_direction = None
  _value_prop = None
  _lock_prop = None

  _PIXELS_PER_FULL_RANGE = 320.0

  @classmethod
  def poll(cls, context):
    return (
      context.area is not None
      and context.area.type == 'VIEW_3D'
      and context.region is not None
      and context.region.type == 'WINDOW'
    )

  def _clear_header(self, context):
    try:
      if context.area:
        context.area.header_text_set(None)
    except Exception:
      pass

  def _begin_drag(self, context, event, settings):
    self._dragging = True
    self._waiting_for_lmb = False

    self._start_mouse = Vector((
      float(event.mouse_region_x),
      float(event.mouse_region_y),
    ))
    self._start_value = float(getattr(settings, self._value_prop))

    center = Vector((
      float(context.region.width) * 0.5,
      float(context.region.height) * 0.5,
    ))

    radial = self._start_mouse - center

    if radial.length >= 18.0:
      radial.normalize()
      self._radial_direction = radial
    else:
      self._radial_direction = Vector((1.0, 0.0))

    try:
      if context.area:
        context.area.header_text_set(
          f"{self.axis} Shrink  "
          "LMB inward = shrink  •  outward = expand  •  Esc = cancel"
        )
    except Exception:
      pass

  def invoke(self, context, event):
    axis = str(self.axis).upper()
    props = _slider_axis_props(axis)

    if props is None:
      return {'CANCELLED'}

    settings = _settings(context)
    if not settings:
      return {'CANCELLED'}

    value_prop, lock_prop = props

    if bool(getattr(settings, lock_prop)):
      self.report({'INFO'}, f"{axis} shrink is locked")
      return {'CANCELLED'}

    self.axis = axis
    self._value_prop = value_prop
    self._lock_prop = lock_prop

    # New v29 keymap items store the whole chord. Old v28 one-key items
    # remain usable as a fallback until reassigned.
    self._chord_keys = _decode_slider_chord(self.chord)
    if not self._chord_keys:
      self._chord_keys = [str(event.type)]

    self._trigger_key = str(event.type)
    self._held_keys = {self._trigger_key}

    # If Ctrl/Shift/Alt/Cmd were pressed before the trigger key, Blender
    # exposes them through event modifier flags. Seed them here so modifier
    # order does not matter.
    _seed_modifier_keys_from_event(
      self._chord_keys,
      self._held_keys,
      event,
    )

    self._waiting_for_lmb = True
    self._dragging = False
    self._start_mouse = None
    self._start_value = float(getattr(settings, value_prop))

    chord_text = _pretty_slider_chord(
      self._chord_keys,
      include_lmb=False,
    )

    try:
      if context.area:
        context.area.header_text_set(
          f"{axis} Shrink  Hold {chord_text} + drag LMB"
        )
    except Exception:
      pass

    context.window_manager.modal_handler_add(self)
    return {'RUNNING_MODAL'}

  def modal(self, context, event):
    settings = _settings(context)
    if not settings:
      self._clear_header(context)
      return {'CANCELLED'}

    event_type = str(event.type)

    if event.type == 'ESC' and event.value == 'PRESS':
      if self._dragging:
        setattr(settings, self._value_prop, self._start_value)
        _mark_dirty()
        _tag_redraw_all_view3d()

      self._clear_header(context)
      return {'CANCELLED'}

    if event.type == 'WINDOW_DEACTIVATE':
      self._clear_header(context)
      return {'FINISHED'}

    # Track keyboard state after the first trigger key started this modal.
    if event_type not in {
      'LEFTMOUSE',
      'MIDDLEMOUSE',
      'RIGHTMOUSE',
    }:
      if event.value == 'PRESS':
        self._held_keys.add(event_type)
      elif event.value == 'RELEASE':
        self._held_keys.discard(event_type)

        # Releasing any required chord key ends an active drag.
        if self._dragging and any(
          (
            required == event_type
            or (
              _modifier_group_for_key(required) is not None
              and _modifier_group_for_key(required)
              == _modifier_group_for_key(event_type)
            )
          )
          for required in self._chord_keys
        ):
          self._clear_header(context)
          return {'FINISHED'}

        # Releasing the key that originally invoked the shortcut ends
        # the waiting state too.
        if (
          not self._dragging
          and event_type == self._trigger_key
        ):
          self._clear_header(context)
          return {'FINISHED'}

    if self._waiting_for_lmb:
      if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
        if _slider_chord_is_held(
          self._chord_keys,
          self._held_keys,
          event,
        ):
          self._begin_drag(context, event, settings)
          return {'RUNNING_MODAL'}

        # Do not start until the entire configured chord is held.
        missing = [
          key for key in self._chord_keys
          if not _chord_key_is_held(
            key,
            self._held_keys,
            event,
          )
        ]

        if missing:
          try:
            missing_text = " + ".join(
              _pretty_chord_key(key)
              for key in missing
            )
            if context.area:
              context.area.header_text_set(
                f"Hold remaining key(s): {missing_text}"
              )
          except Exception:
            pass

        return {'RUNNING_MODAL'}

      return {'RUNNING_MODAL'}

    if self._dragging:
      if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
        self._clear_header(context)
        return {'FINISHED'}

      if event.type == 'MOUSEMOVE':
        current = Vector((
          float(event.mouse_region_x),
          float(event.mouse_region_y),
        ))

        mouse_delta = current - self._start_mouse
        radial_pixels = mouse_delta.dot(self._radial_direction)
        value_delta = radial_pixels / self._PIXELS_PER_FULL_RANGE

        new_value = max(
          0.05,
          min(
            1.0,
            self._start_value + value_delta,
          ),
        )

        setattr(settings, self._value_prop, new_value)

        _mark_dirty()
        if context.area:
          context.area.tag_redraw()

        return {'RUNNING_MODAL'}

    return {'RUNNING_MODAL'}


def _slider_shortcut_display(kmi):
  if kmi is None:
    return "Not set"

  try:
    chord = _decode_slider_chord(kmi.properties.chord)
  except Exception:
    chord = []

  # Legacy v28 one-key assignment.
  if not chord:
    try:
      chord = [str(kmi.type)]
    except Exception:
      return "Not set"

  return _pretty_slider_chord(chord, include_lmb=True)




class VIEWPORTSHRINK_OT_ClearSliderShortcut(bpy.types.Operator):
  bl_idname = "viewport_shrink.clear_slider_shortcut"
  bl_label = "Clear Slider Shortcut"
  bl_description = "Clear this slider drag shortcut"
  bl_options = {'INTERNAL'}

  axis: StringProperty(options={'HIDDEN'})

  def execute(self, context):
    _remove_slider_shortcut(context, str(self.axis).upper())
    try:
      _tag_redraw_all_view3d()
    except Exception:
      pass
    return {'FINISHED'}


class VIEWPORTSHRINK_OT_CaptureSliderShortcut(bpy.types.Operator):
  bl_idname = "viewport_shrink.capture_slider_shortcut"
  bl_label = "Set Slider Drag Shortcut"
  bl_description = (
    "Choose 1 to 4 keyboard buttons, then click LMB to save the chord"
  )
  bl_options = {'INTERNAL'}

  axis: StringProperty(options={'HIDDEN'})

  _captured_keys = None
  _ignore_launch_mouse = True

  @classmethod
  def description(cls, context, properties):
    try:
      axis = str(properties.axis).upper()
      current = _slider_shortcut_display(
        _find_slider_shortcut(context, axis)
      )
      return (
        f"Set live drag chord for {axis} slider\n"
        f"Current: {current}\n"
        "Press 1–4 keyboard buttons, then click LMB to save\n"
        "Esc: Cancel  Backspace/Delete: Clear"
      )
    except Exception:
      return "Set 1–4 keyboard buttons + LMB drag"

  def _clear_header(self, context):
    try:
      if context.area:
        context.area.header_text_set(None)
    except Exception:
      pass

  def _update_header(self, context):
    try:
      if not context.area:
        return

      if self._captured_keys:
        chord_text = _pretty_slider_chord(
          self._captured_keys,
          include_lmb=False,
        )
        context.area.header_text_set(
          f"{self.axis} Drag: {chord_text}  "
          f"({len(self._captured_keys)}/4)  "
          "Click LMB to save  •  Esc cancels"
        )
      else:
        context.area.header_text_set(
          f"Set {self.axis} Drag  "
          "Press 1–4 keys • LMB = Save"
        )
    except Exception:
      pass

  def invoke(self, context, event):
    axis = str(self.axis).upper()
    if axis not in {'X', 'Y', 'Z'}:
      return {'CANCELLED'}

    self.axis = axis
    self._captured_keys = []

    # The operator can be invoked by Blender on either the UI mouse PRESS
    # or RELEASE depending on context/version.
    #
    # Only wait for a release if invoke() actually arrived on PRESS.
    # v29 set this True unconditionally, which swallowed every keyboard
    # event until the user clicked LMB again.
    self._ignore_launch_mouse = (
      event.type == 'LEFTMOUSE'
      and event.value == 'PRESS'
    )

    self._update_header(context)

    context.window_manager.modal_handler_add(self)
    return {'RUNNING_MODAL'}

  def _save_chord(self, context):
    if not self._captured_keys:
      self.report(
        {'WARNING'},
        "Press at least one keyboard button first",
      )
      return {'RUNNING_MODAL'}

    km = _user_3d_view_keymap(context, create=True)
    if km is None:
      self._clear_header(context)
      self.report({'ERROR'}, "User keymap is unavailable")
      return {'CANCELLED'}

    _remove_slider_shortcut(context, self.axis)

    chord = self._captured_keys[:_SLIDER_CHORD_MAX_KEYS]
    trigger_key = chord[0]
    encoded = _encode_slider_chord(chord)

    try:
      try:
        kmi = km.keymap_items.new(
          VIEWPORTSHRINK_OT_SliderGesture.bl_idname,
          type=trigger_key,
          value='PRESS',
          head=True,
        )
      except TypeError:
        kmi = km.keymap_items.new(
          VIEWPORTSHRINK_OT_SliderGesture.bl_idname,
          type=trigger_key,
          value='PRESS',
        )

      kmi.properties.axis = self.axis
      kmi.properties.chord = encoded

    except Exception as exc:
      self._clear_header(context)
      self.report(
        {'ERROR'},
        f"Could not assign slider shortcut: {exc}",
      )
      return {'CANCELLED'}

    current = _pretty_slider_chord(chord, include_lmb=True)

    self._clear_header(context)
    self.report(
      {'INFO'},
      f"{self.axis} Drag → {current}",
    )

    try:
      _tag_redraw_all_view3d()
    except Exception:
      pass

    return {'FINISHED'}

  def modal(self, context, event):
    # Consume only the leftover mouse event from the UI click that opened
    # this capture operator. Never swallow keyboard events.
    if self._ignore_launch_mouse:
      if event.type == 'LEFTMOUSE':
        if event.value == 'RELEASE':
          self._ignore_launch_mouse = False
        return {'RUNNING_MODAL'}

      # A keyboard event means the user has already started entering the
      # chord. Stop caring about the old launch click immediately.
      self._ignore_launch_mouse = False

    if event.type == 'ESC' and event.value == 'PRESS':
      self._clear_header(context)
      self.report(
        {'INFO'},
        "Slider shortcut assignment cancelled",
      )
      return {'CANCELLED'}

    # Clear assignment. Reserved during capture rather than usable as a
    # chord key because it provides an easy, discoverable clear action.
    if event.type in {'BACK_SPACE', 'DEL'} and event.value == 'PRESS':
      removed = _remove_slider_shortcut(
        context,
        self.axis,
      )

      self._clear_header(context)

      if removed:
        self.report(
          {'INFO'},
          f"Cleared {self.axis} slider drag shortcut",
        )
      else:
        self.report(
          {'INFO'},
          f"No {self.axis} slider drag shortcut was set",
        )

      try:
        _tag_redraw_all_view3d()
      except Exception:
        pass

      return {'FINISHED'}

    # LMB confirms the completed 1–4 key chord.
    if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
      if not self._captured_keys:
        self.report(
          {'WARNING'},
          "Press 1 to 4 keyboard buttons, then click LMB to save",
        )
        self._update_header(context)
        return {'RUNNING_MODAL'}

      return self._save_chord(context)

    if not _event_is_keyboard_button(event):
      return {'RUNNING_MODAL'}

    key = str(event.type)

    if key in self._captured_keys:
      return {'RUNNING_MODAL'}

    if len(self._captured_keys) >= _SLIDER_CHORD_MAX_KEYS:
      self.report(
        {'WARNING'},
        "Maximum is 4 keyboard buttons",
      )
      return {'RUNNING_MODAL'}

    self._captured_keys.append(key)
    self._update_header(context)

    # Redraw immediately so the header reflects the chord as each key is
    # captured.
    try:
      if context.area:
        context.area.tag_redraw()
    except Exception:
      pass

    return {'RUNNING_MODAL'}



def _reset_shortcut_target_matches(kmi):
  return kmi.idname == VIEWPORTSHRINK_OT_Reset.bl_idname


def _find_reset_shortcut(context):
  km = _user_3d_view_keymap(context, create=False)
  if km is None:
    return None

  for kmi in km.keymap_items:
    if _reset_shortcut_target_matches(kmi):
      return kmi

  return None


def _remove_reset_shortcut(context):
  km = _user_3d_view_keymap(context, create=False)
  if km is None:
    return False

  removed = False

  for kmi in list(km.keymap_items):
    if _reset_shortcut_target_matches(kmi):
      try:
        km.keymap_items.remove(kmi)
        removed = True
      except Exception:
        pass

  return removed



class VIEWPORTSHRINK_OT_ClearResetShortcut(bpy.types.Operator):
  bl_idname = "viewport_shrink.clear_reset_shortcut"
  bl_label = "Clear Reset Shortcut"
  bl_description = "Clear the Reset All shortcut"
  bl_options = {'INTERNAL'}

  def execute(self, context):
    _remove_reset_shortcut(context)
    try:
      _tag_redraw_all_view3d()
    except Exception:
      pass
    return {'FINISHED'}


class VIEWPORTSHRINK_OT_CaptureResetShortcut(bpy.types.Operator):
  bl_idname = "viewport_shrink.capture_reset_shortcut"
  bl_label = "Set Reset All Shortcut"
  bl_description = "Assign a keyboard shortcut that resets X, Y, and Z to 1"
  bl_options = {'INTERNAL'}

  @classmethod
  def description(cls, context, properties):
    try:
      current = _shortcut_display_from_kmi(
        _find_reset_shortcut(context)
      )
      return (
        "Set shortcut for Reset All\n"
        f"Current: {current}\n"
        "Esc: Cancel  Backspace/Delete: Clear"
      )
    except Exception:
      return "Set keyboard shortcut for Reset All"

  def _clear_header(self, context):
    try:
      if context.area:
        context.area.header_text_set(None)
    except Exception:
      pass

  def invoke(self, context, event):
    try:
      if context.area:
        context.area.header_text_set(
          "Set Reset All Shortcut  "
          "Press keys • Backspace/Delete clears • Esc cancels"
        )
    except Exception:
      pass

    context.window_manager.modal_handler_add(self)
    return {'RUNNING_MODAL'}

  def modal(self, context, event):
    if event.type == 'ESC' and event.value == 'PRESS':
      self._clear_header(context)
      self.report({'INFO'}, "Reset shortcut assignment cancelled")
      return {'CANCELLED'}

    if event.type in {'BACK_SPACE', 'DEL'} and event.value == 'PRESS':
      removed = _remove_reset_shortcut(context)
      self._clear_header(context)

      if removed:
        self.report({'INFO'}, "Cleared Reset All shortcut")
      else:
        self.report({'INFO'}, "No Reset All shortcut was set")

      return {'FINISHED'}

    if not _shortcut_event_is_assignable(event):
      return {'RUNNING_MODAL'}

    km = _user_3d_view_keymap(context, create=True)
    if km is None:
      self._clear_header(context)
      self.report({'ERROR'}, "User keymap is unavailable")
      return {'CANCELLED'}

    _remove_reset_shortcut(context)

    try:
      kmi = km.keymap_items.new(
        VIEWPORTSHRINK_OT_Reset.bl_idname,
        type=event.type,
        value='PRESS',
        ctrl=bool(event.ctrl),
        shift=bool(event.shift),
        alt=bool(event.alt),
        oskey=bool(event.oskey),
      )

    except Exception as exc:
      self._clear_header(context)
      self.report(
        {'ERROR'},
        f"Could not assign Reset All shortcut: {exc}",
      )
      return {'CANCELLED'}

    shortcut_name = _shortcut_display_from_event(event)
    self._clear_header(context)

    self.report(
      {'INFO'},
      f"Reset All → {shortcut_name}",
    )

    return {'FINISHED'}


def _shortcut_compact_display(kmi):
  """Friendly, compact text used in the shortcut overview popup."""
  if kmi is None:
    return "Not set"

  text_value = _shortcut_display_from_kmi(kmi)

  replacements = {
    "Leftmouse": "LMB",
    "Middlemouse": "MMB",
    "Rightmouse": "RMB",
    "Button4Mouse": "Mouse 4",
    "Button5Mouse": "Mouse 5",
    "Back Space": "Backspace",
    "Del": "Delete",
    "Numpad ": "Num ",
  }

  for old, new in replacements.items():
    text_value = text_value.replace(old, new)

  return text_value


def _draw_shortcut_value(row, text_value):
  """Right-aligned muted shortcut readout."""
  value = row.row()
  value.alignment = 'RIGHT'
  value.active = text_value != "Not set"
  value.label(text=text_value)


class VIEWPORTSHRINK_OT_ShortcutPopup(bpy.types.Operator):
  bl_idname = "viewport_shrink.shortcut_popup"
  bl_label = "Set Shortcut"
  bl_description = "View and edit all Viewport Shrink shortcuts"
  bl_options = {'INTERNAL'}

  def invoke(self, context, event):
    # Unlike bpy.types.Menu, invoke_popup does not auto-columnize a tall
    # layout. The requested width keeps this as one clean vertical panel.
    return context.window_manager.invoke_popup(self, width=470)

  def execute(self, context):
    return {'FINISHED'}

  def draw(self, context):
    layout = self.layout
    root = layout.column(align=False)

    _ensure_custom_preset_rows(context)

    settings = _settings(context)

    # --------------------------------------------------------------
    # User-defined preset shortcuts
    # --------------------------------------------------------------
    box = root.box()
    box.label(text="Custom Shrink Shortcuts", icon='KEYINGSET')
    col = box.column(align=False)

    if settings is not None:
      for axis in ('X', 'Y', 'Z'):
        header = col.row(align=True)

        # Mirror the actual custom shortcut row proportions:
        # [ slider/value area ][ Set Shortcut area ][ + ]
        #
        # This keeps "X Axis / Y Axis / Z Axis" centered directly
        # over the numeric slider value such as 1.00.
        header_main = header.split(factor=0.475, align=True)

        axis_label = header_main.row(align=True)
        axis_label.alignment = 'CENTER'
        axis_label.label(text=f"{axis} Axis")

        header_right = header_main.row(align=True)
        header_right_split = header_right.split(factor=0.887, align=True)

        # Empty middle area directly above Set Shortcut.
        header_right_split.row(align=True)

        # + remains in the same compact far-right column as delete X.
        add_side = header_right_split.row(align=True)
        add_side.alignment = 'RIGHT'

        add = add_side.operator(
          VIEWPORTSHRINK_OT_AddCustomPresetShortcut.bl_idname,
          text="",
          icon='ADD',
        )
        add.axis = axis

        axis_items = [
          item
          for item in settings.custom_preset_shortcuts
          if str(item.axis).upper() == axis
        ]

        if not axis_items:
          # Place the empty-state text beneath the + side instead of
          # spanning the whole custom-shortcut section.
          empty = col.row(align=True)
          empty.active = False
          empty.alignment = 'RIGHT'
          empty.label(text="No custom shortcuts")

        for item in axis_items:
          row = col.row(align=True)

          # Full freedom: drag the slider or click/type an exact
          # number from 0.10 to 1.00.
          row.prop(
            item,
            "value",
            text="",
            slider=True,
          )

          set_op = row.operator(
            VIEWPORTSHRINK_OT_CaptureCustomPresetShortcut.bl_idname,
            text="Set Shortcut",
            icon='KEYINGSET',
          )
          set_op.slot_uid = item.uid

          remove = row.operator(
            VIEWPORTSHRINK_OT_RemoveCustomPresetShortcut.bl_idname,
            text="",
            icon='X',
          )
          remove.slot_uid = item.uid

          current = col.row()
          current.active = False
          current.alignment = 'RIGHT'
          current.label(
            text=(
              "Shortcut: "
              + _custom_preset_current_shortcut(
                context,
                item.uid,
              )
            )
          )

        if axis != 'Z':
          col.separator(factor=0.65)

    root.separator(factor=0.7)

    # --------------------------------------------------------------
    # Live slider drag
    # --------------------------------------------------------------
    box = root.box()
    box.label(text="Slider Drag", icon='MOUSE_LMB')
    col = box.column(align=False)

    for axis in ('X', 'Y', 'Z'):
      row = col.row(align=True)

      # Layout proportions:
      # ~47% disabled label | ~47% Set Shortcut | ~6% delete
      main_split = row.split(factor=0.475, align=True)

      # Text-only label, centered in the same left column as X/Y/Z Axis.
      label_side = main_split.row(align=True)
      label_side.alignment = 'CENTER'
      label_side.label(text=f"{axis} Drag")

      action_side = main_split.row(align=True)

      set_op = action_side.operator(
        VIEWPORTSHRINK_OT_CaptureSliderShortcut.bl_idname,
        text="Set Shortcut",
        icon='KEYINGSET',
      )
      set_op.axis = axis

      clear = action_side.operator(
        VIEWPORTSHRINK_OT_ClearSliderShortcut.bl_idname,
        text="",
        icon='X',
      )
      clear.axis = axis

      current = col.row()
      current.active = False
      current.alignment = 'RIGHT'
      current.label(
        text=(
          "Shortcut: "
          + _slider_shortcut_display(
            _find_slider_shortcut(context, axis)
          )
        )
      )

    hint = box.row()
    hint.active = False
    hint.label(text="Hold Key + LMB drag • inward shrink • outward expand")

    root.separator(factor=0.7)

    # --------------------------------------------------------------
    # Reset
    # --------------------------------------------------------------
    box = root.box()
    box.label(text="Reset", icon='LOOP_BACK')
    col = box.column(align=False)

    row = col.row(align=True)

    # Same proportions and disabled visual language as Slider Drag.
    main_split = row.split(factor=0.475, align=True)

    # Text-only label aligned with the X/Y/Z Drag column.
    label_side = main_split.row(align=True)
    label_side.alignment = 'CENTER'
    label_side.label(text="Reset All")

    action_side = main_split.row(align=True)

    action_side.operator(
      VIEWPORTSHRINK_OT_CaptureResetShortcut.bl_idname,
      text="Set Shortcut",
      icon='KEYINGSET',
    )

    action_side.operator(
      VIEWPORTSHRINK_OT_ClearResetShortcut.bl_idname,
      text="",
      icon='X',
    )

    current = col.row()
    current.active = False
    current.alignment = 'RIGHT'
    current.label(
      text=(
        "Shortcut: "
        + _shortcut_compact_display(
          _find_reset_shortcut(context)
        )
      )
    )

    # --------------------------------------------------------------
    # Help
    # --------------------------------------------------------------
    root.separator(factor=0.65)
    help_col = root.column(align=True)
    help_col.active = False
    help_col.label(text="Esc = Cancel")


class VIEWPORTSHRINK_OT_Reset(bpy.types.Operator):
  bl_idname = "viewport_shrink.reset"
  bl_label = "Reset Viewport"
  bl_description = "Restore X, Y and Z viewport shrink to 1.0"
  bl_options = {'REGISTER'}

  def execute(self, context):
    s = _settings(context)
    if not s:
      return {'CANCELLED'}
    s.shrink_x = 1.0
    s.shrink_y = 1.0
    s.shrink_z = 1.0
    _mark_dirty()
    _tag_redraw_all_view3d()
    return {'FINISHED'}


# -----------------------------------------------------------------------------
# Projection-aware editable point selection
# -----------------------------------------------------------------------------

def _candidate_object(candidate):
  """Resolve the candidate object fresh every time.

  Never keep BMesh/BMVert or curve-point RNA references inside a mouse modal.
  Blender is free to rebuild Edit Mode data after selection flushes/redraws;
  old element wrappers then raise "... has been removed". Stable names and
  indices survive those rebuilds and are resolved on demand.
  """
  return bpy.data.objects.get(candidate.get('object_name', ''))


def _resolve_curve_element(candidate):
  obj = _candidate_object(candidate)
  if not obj or obj.type not in {'CURVE', 'SURFACE'}:
    return None
  si = int(candidate.get('si', -1))
  pi = int(candidate.get('pi', -1))
  if si < 0 or si >= len(obj.data.splines):
    return None
  spline = obj.data.splines[si]
  try:
    if candidate.get('part') in {'BEZIER_CENTER', 'BEZIER_LEFT', 'BEZIER_RIGHT'}:
      if spline.type != 'BEZIER' or pi >= len(spline.bezier_points):
        return None
      return spline.bezier_points[pi]
    if pi >= len(spline.points):
      return None
    return spline.points[pi]
  except (ReferenceError, IndexError, RuntimeError):
    return None


def _resolve_mesh_vert(candidate):
  obj = _candidate_object(candidate)
  if not obj or obj.type != 'MESH' or obj.mode != 'EDIT':
    return None
  idx = int(candidate.get('pi', -1))
  if idx < 0:
    return None
  try:
    bm = bmesh.from_edit_mesh(obj.data)
    # ensure_lookup_table() makes bm.verts[i] valid, but it does NOT
    # guarantee that BMVert.index fields are current. Keep both in sync.
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    if idx >= len(bm.verts):
      return None
    return bm.verts[idx]
  except (ReferenceError, RuntimeError):
    return None



def _resolve_mesh_edge(candidate):
  obj = _candidate_object(candidate)
  if not obj or obj.type != 'MESH' or obj.mode != 'EDIT':
    return None

  idx = int(candidate.get('ei', -1))
  if idx < 0:
    return None

  try:
    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    bm.edges.index_update()
    if idx >= len(bm.edges):
      return None
    return bm.edges[idx]
  except (ReferenceError, RuntimeError, IndexError):
    return None


def _set_mesh_edge_candidate(candidate, value):
  obj = _candidate_object(candidate)
  if not obj or obj.type != 'MESH' or obj.mode != 'EDIT':
    return False

  idx = int(candidate.get('ei', -1))
  if idx < 0:
    return False

  try:
    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    bm.edges.index_update()
    if idx >= len(bm.edges):
      return False

    edge = bm.edges[idx]
    value = bool(value)
    edge.select_set(value)

    if value:
      bm.select_history.discard(edge)
      bm.select_history.add(edge)
    else:
      bm.select_history.discard(edge)

    bm.select_history.validate()
    bm.select_flush_mode()
    return True
  except (ReferenceError, RuntimeError, IndexError):
    return False


def _set_curve_candidate(candidate, value):
  elem = _resolve_curve_element(candidate)
  if elem is None:
    return False
  part = candidate['part']
  value = bool(value)
  try:
    if part == 'POINT':
      elem.select = value
    elif part == 'BEZIER_CENTER':
      # Native curve behavior: center selection carries both handles.
      elem.select_control_point = value
      elem.select_left_handle = value
      elem.select_right_handle = value
    elif part == 'BEZIER_LEFT':
      elem.select_left_handle = value
    elif part == 'BEZIER_RIGHT':
      elem.select_right_handle = value
    return True
  except ReferenceError:
    return False


def _candidate_selected(candidate):
  try:
    if candidate['kind'] == 'MESH_VERT':
      elem = _resolve_mesh_vert(candidate)
      return bool(elem.select) if elem is not None else False
    if candidate['kind'] == 'MESH_EDGE':
      elem = _resolve_mesh_edge(candidate)
      return bool(elem.select) if elem is not None else False
    elem = _resolve_curve_element(candidate)
    if elem is None:
      return False
    part = candidate['part']
    if part == 'POINT': return bool(elem.select)
    if part == 'BEZIER_CENTER': return bool(elem.select_control_point)
    if part == 'BEZIER_LEFT': return bool(elem.select_left_handle)
    if part == 'BEZIER_RIGHT': return bool(elem.select_right_handle)
  except ReferenceError:
    return False
  return False


def _set_candidate(candidate, value):
  if candidate['kind'] == 'MESH_EDGE':
    return _set_mesh_edge_candidate(candidate, value)

  if candidate['kind'] == 'MESH_VERT':
    obj = _candidate_object(candidate)
    if not obj or obj.type != 'MESH' or obj.mode != 'EDIT':
      return False
    idx = int(candidate.get('pi', -1))
    if idx < 0:
      return False
    try:
      bm = bmesh.from_edit_mesh(obj.data)
      bm.verts.ensure_lookup_table()
      bm.verts.index_update()
      if idx >= len(bm.verts):
        return False
      elem = bm.verts[idx]
      value = bool(value)
      elem.select_set(value)
      # Match Blender's active-element behavior as closely as possible.
      # Selection history matters to many Edit Mode operations even when
      # the visible orange highlight itself only needs elem.select.
      if value:
        bm.select_history.discard(elem)
        bm.select_history.add(elem)
      else:
        bm.select_history.discard(elem)
      bm.select_history.validate()
      bm.select_flush_mode()
      return True
    except (ReferenceError, RuntimeError, IndexError):
      return False
  return _set_curve_candidate(candidate, value)


def _deselect_supported_edit(context):
  touched_meshes = []
  for obj in _edit_mode_objects(context):
    if obj.type == 'MESH':
      try:
        bm = bmesh.from_edit_mesh(obj.data)
        bm.select_history.clear()
        for v in bm.verts:
          v.select_set(False)
        for e in bm.edges:
          e.select_set(False)
        for f in bm.faces:
          f.select_set(False)
        bm.select_flush_mode()
        touched_meshes.append(obj.data)
      except Exception:
        pass
    elif obj.type in {'CURVE', 'SURFACE'}:
      for spline in obj.data.splines:
        if spline.type == 'BEZIER':
          for bp in spline.bezier_points:
            bp.select_control_point = False
            bp.select_left_handle = False
            bp.select_right_handle = False
        else:
          for pt in spline.points:
            pt.select = False
  for data in touched_meshes:
    try: bmesh.update_edit_mesh(data, loop_triangles=False, destructive=False)
    except Exception: pass


def _iter_edit_candidates(context, settings, include_offscreen=False, view_projection=None):
  """Yield editable CV/vertex candidates in the *displayed* XYZ-shrink projection."""
  if view_projection is None:
    vp = _display_view_projection(context, settings)
  else:
    vp = view_projection
  w, h = context.region.width, context.region.height
  margin = 24.0

  for obj in _edit_mode_objects(context):
    mw = obj.matrix_world

    if obj.type in {'CURVE', 'SURFACE'}:
      for si, spline in enumerate(obj.data.splines):
        if spline.type == 'BEZIER':
          for pi, bp in enumerate(spline.bezier_points):
            if getattr(bp, 'hide', False):
              continue
            for part, local in (
              ('BEZIER_CENTER', bp.co),
              ('BEZIER_LEFT', bp.handle_left),
              ('BEZIER_RIGHT', bp.handle_right),
            ):
              proj = _project_world(context, mw @ local, settings, vp)
              if not proj: continue
              x, y, z = proj
              if include_offscreen or (-margin <= x <= w+margin and -margin <= y <= h+margin):
                yield {'kind':'CURVE', 'object_name':obj.name, 'part':part, 'x':x, 'y':y, 'z':z, 'si':si, 'pi':pi, 'selected': (bool(bp.select_control_point) if part=='BEZIER_CENTER' else bool(bp.select_left_handle) if part=='BEZIER_LEFT' else bool(bp.select_right_handle))}
        else:
          for pi, pt in enumerate(spline.points):
            if getattr(pt, 'hide', False):
              continue
            proj = _project_world(context, mw @ Vector(pt.co[:3]), settings, vp)
            if not proj: continue
            x, y, z = proj
            if include_offscreen or (-margin <= x <= w+margin and -margin <= y <= h+margin):
              yield {'kind':'CURVE', 'object_name':obj.name, 'part':'POINT', 'x':x, 'y':y, 'z':z, 'si':si, 'pi':pi, 'selected': bool(pt.select)}

    elif obj.type == 'MESH':
      # CV-style interaction for meshes means Vertex Select mode. Edge and
      # face selection remain native rather than pretending vertex picks
      # are equivalent to those modes.
      if not context.tool_settings.mesh_select_mode[0]:
        continue
      try:
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.verts.index_update()
        for v in bm.verts:
          if v.hide: continue
          proj = _project_world(context, mw @ v.co, settings, vp)
          if not proj: continue
          x, y, z = proj
          if include_offscreen or (-margin <= x <= w+margin and -margin <= y <= h+margin):
            yield {'kind':'MESH_VERT', 'object_name':obj.name, 'part':'VERT', 'x':x, 'y':y, 'z':z, 'pi':v.index, 'selected': bool(v.select)}
      except Exception:
        continue



def _point_segment_distance_sq(px, py, x0, y0, x1, y1):
  dx = float(x1) - float(x0)
  dy = float(y1) - float(y0)
  denom = dx * dx + dy * dy

  if denom <= 1e-12:
    ex = float(px) - float(x0)
    ey = float(py) - float(y0)
    return ex * ex + ey * ey, 0.0

  t = (
    (float(px) - float(x0)) * dx
    + (float(py) - float(y0)) * dy
  ) / denom
  t = max(0.0, min(1.0, t))

  qx = float(x0) + dx * t
  qy = float(y0) + dy * t

  ex = float(px) - qx
  ey = float(py) - qy
  return ex * ex + ey * ey, t


def _iter_mesh_edge_candidates(context, settings, view_projection=None, include_offscreen=False):
  """Yield stable mesh-edge IDs in the exact displayed shrink projection."""
  if view_projection is None:
    vp = _display_view_projection(context, settings)
  else:
    vp = view_projection

  w = float(context.region.width)
  h = float(context.region.height)
  margin = 30.0

  for obj in _edit_mode_objects(context):
    if obj.type != 'MESH':
      continue

    try:
      bm = bmesh.from_edit_mesh(obj.data)
      bm.verts.ensure_lookup_table()
      bm.edges.ensure_lookup_table()
      bm.verts.index_update()
      bm.edges.index_update()
    except Exception:
      continue

    mw = obj.matrix_world

    for edge in bm.edges:
      if edge.hide:
        continue

      v0, v1 = edge.verts
      if v0.hide or v1.hide:
        continue

      w0 = mw @ v0.co
      w1 = mw @ v1.co

      p0 = _project_world(context, w0, settings, vp)
      p1 = _project_world(context, w1, settings, vp)
      if p0 is None or p1 is None:
        continue

      x0, y0, z0 = p0
      x1, y1, z1 = p1

      if not include_offscreen:
        if (
          max(x0, x1) < -margin
          or min(x0, x1) > w + margin
          or max(y0, y1) < -margin
          or min(y0, y1) > h + margin
        ):
          continue

      yield {
        'kind': 'MESH_EDGE',
        'object_name': obj.name,
        'ei': int(edge.index),
        'x0': float(x0),
        'y0': float(y0),
        'z0': float(z0),
        'x1': float(x1),
        'y1': float(y1),
        'z1': float(z1),
        'v0': int(v0.index),
        'v1': int(v1.index),
        'selected': bool(edge.select),
      }



def _visible_surface_ndc_depth(context, settings, mx, my, view_projection=None):
  """NDC depth of the visible surface under the SHRUNK display pixel.

  Returns None over background. X-Ray intentionally returns None because
  hidden edit geometry is allowed to be selectable there.
  """
  if _viewport_xray_enabled(context):
    return None

  vp = (
    view_projection.copy()
    if view_projection is not None
    else _display_view_projection(context, settings)
  )
  inv_vp = vp.inverted_safe()

  p_near = _unproject_display(
    context, mx, my, -1.0, settings, inv_vp
  )
  p_far = _unproject_display(
    context, mx, my, 1.0, settings, inv_vp
  )

  delta = p_far - p_near
  distance = delta.length
  if distance <= 1e-10:
    return None

  direction = delta / distance

  try:
    hit, location, normal, index, obj, matrix = _scene_ray_cast(
      context,
      p_near,
      direction,
      distance,
    )
  except Exception:
    return None

  if not hit:
    return None

  projected = _project_world(
    context,
    location,
    settings,
    view_projection=vp,
  )
  if projected is None:
    return None

  return float(projected[2])


def _candidate_depth_visible(candidate_z, surface_z):
  if surface_z is None:
    return True

  z = float(candidate_z)

  # Match the tiny adaptive depth offset used by the visible edit overlay,
  # with a small numeric allowance for ray/edge interpolation.
  allowance = max(
    0.00012,
    _adaptive_ndc_depth_bias(z, _EDGE_NDC_DEPTH_BIAS_MAX) * 2.25,
  )
  return z <= float(surface_z) + allowance


def _pick_mesh_edge_candidate(context, settings, mx, my, radius=None):
  """Pick the edge that is visibly under the cursor in the displayed frame."""
  ui = float(getattr(context.preferences.system, 'ui_scale', 1.0) or 1.0)

  if radius is None:
    # Large convenience hit area, but the nearest VISIBLE projected edge
    # still wins. This changes tolerance only, never the selected topology.
    radius = max(6.0, 8.0 * ui) * _PICK_HITBOX_MULTIPLIER

  radius2 = radius * radius
  vp = _display_view_projection(context, settings)
  surface_z = _visible_surface_ndc_depth(
    context,
    settings,
    mx,
    my,
    view_projection=vp,
  )

  hits = []

  for c in _iter_mesh_edge_candidates(
    context,
    settings,
    view_projection=vp,
  ):
    d2, t = _point_segment_distance_sq(
      mx, my,
      c['x0'], c['y0'],
      c['x1'], c['y1'],
    )
    if d2 > radius2:
      continue

    z = c['z0'] + (c['z1'] - c['z0']) * t

    if not _candidate_depth_visible(z, surface_z):
      continue

    hits.append((d2, z, c))

  if not hits:
    return None

  # What is visually closest to the cursor wins. Depth only breaks a screen
  # distance tie after hidden candidates have already been rejected.
  hits.sort(key=lambda item: (round(item[0], 7), item[1]))
  return hits[0][2]


def _pick_edit_candidate(context, settings, mx, my):
  """Use Blender's active mesh selection mode to choose point vs edge picker."""
  if context.mode == 'EDIT_MESH':
    vert_mode, edge_mode, face_mode = context.tool_settings.mesh_select_mode

    if edge_mode and not vert_mode:
      return _pick_mesh_edge_candidate(context, settings, mx, my)

    if vert_mode and not edge_mode:
      return _pick_candidate(context, settings, mx, my)

    if vert_mode and edge_mode:
      vert_hit = _pick_candidate(context, settings, mx, my)
      edge_hit = _pick_mesh_edge_candidate(context, settings, mx, my)

      if vert_hit is None:
        return edge_hit
      if edge_hit is None:
        return vert_hit

      vd2 = (
        (float(vert_hit['x']) - float(mx)) ** 2
        + (float(vert_hit['y']) - float(my)) ** 2
      )
      ed2, _t = _point_segment_distance_sq(
        mx, my,
        edge_hit['x0'], edge_hit['y0'],
        edge_hit['x1'], edge_hit['y1'],
      )
      return vert_hit if vd2 <= ed2 else edge_hit

    # Face mode remains Blender-native for now.
    return None

  return _pick_candidate(context, settings, mx, my)


def _edge_opposite_score(current_edge, at_vert, candidate):
  """How straight/topologically-opposite is candidate through at_vert?"""
  try:
    incoming_other = current_edge.other_vert(at_vert)
    outgoing_other = candidate.other_vert(at_vert)

    incoming = at_vert.co - incoming_other.co
    outgoing = outgoing_other.co - at_vert.co

    if incoming.length_squared <= 1e-16 or outgoing.length_squared <= 1e-16:
      return -10.0

    incoming.normalize()
    outgoing.normalize()
    straight = incoming.dot(outgoing)

    current_faces = set(current_edge.link_faces)
    candidate_faces = set(candidate.link_faces)

    # A true quad-grid continuation normally does NOT share a face with
    # the incoming edge at a valence-four vertex.
    no_shared_face = 1.0 if current_faces.isdisjoint(candidate_faces) else 0.0

    # Boundary edge loops should follow the boundary.
    boundary_bonus = 0.0
    if len(current_edge.link_faces) <= 1 and len(candidate.link_faces) <= 1:
      boundary_bonus = 2.0

    return boundary_bonus + no_shared_face * 1.5 + straight
  except Exception:
    return -10.0


def _edge_loop_continue(current_edge, at_vert):
  """Choose Blender-like topological continuation through one vertex."""
  candidates = [
    e for e in at_vert.link_edges
    if e is not current_edge and not e.hide
  ]
  if not candidates:
    return None

  # Boundary loop: strongly prefer another boundary edge.
  if len(current_edge.link_faces) <= 1:
    boundary = [e for e in candidates if len(e.link_faces) <= 1]
    if boundary:
      candidates = boundary
  else:
    # Interior quad loop: prefer an edge that shares no face with the
    # incoming edge. This is the topologically opposite continuation.
    cur_faces = set(current_edge.link_faces)
    opposite = [
      e for e in candidates
      if cur_faces.isdisjoint(set(e.link_faces))
    ]
    if opposite:
      candidates = opposite

  if not candidates:
    return None

  return max(
    candidates,
    key=lambda e: _edge_opposite_score(current_edge, at_vert, e),
  )


def _collect_edge_loop(start_edge):
  """Follow an edge loop in both directions from start_edge."""
  result = {start_edge}

  for start_vert in start_edge.verts:
    current_edge = start_edge
    at_vert = start_vert

    # Mesh size is a natural hard safety bound for malformed topology.
    max_steps = 100000
    steps = 0

    while steps < max_steps:
      steps += 1
      next_edge = _edge_loop_continue(current_edge, at_vert)
      if next_edge is None or next_edge in result:
        break

      result.add(next_edge)

      try:
        next_vert = next_edge.other_vert(at_vert)
      except Exception:
        break

      current_edge = next_edge
      at_vert = next_vert

  return result


def _select_mesh_loop_from_edge(context, edge_candidate, shift=False):
  """Select the edge loop containing the projected edge candidate.

  Edge select mode -> selects loop edges.
  Vertex select mode -> selects vertices belonging to the same loop.
  """
  obj = _candidate_object(edge_candidate)
  if not obj or obj.type != 'MESH' or obj.mode != 'EDIT':
    return False

  edge_index = int(edge_candidate.get('ei', -1))
  if edge_index < 0:
    return False

  try:
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.index_update()
    bm.edges.index_update()

    if edge_index >= len(bm.edges):
      return False

    start_edge = bm.edges[edge_index]
    loop_edges = _collect_edge_loop(start_edge)
    if not loop_edges:
      return False

    vert_mode, edge_mode, face_mode = context.tool_settings.mesh_select_mode

    # Normal double-click replaces. Shift-double-click toggles/adds.
    if not shift:
      _deselect_supported_edit(context)

    if edge_mode:
      target_select = True
      if shift and all(e.select for e in loop_edges):
        target_select = False

      for edge in loop_edges:
        edge.select_set(target_select)

      if target_select:
        bm.select_history.discard(start_edge)
        bm.select_history.add(start_edge)

    else:
      # Vertex mode: convert the topological edge loop into its vertices.
      loop_verts = set()
      for edge in loop_edges:
        loop_verts.update(edge.verts)

      target_select = True
      if shift and loop_verts and all(v.select for v in loop_verts):
        target_select = False

      for vert in loop_verts:
        vert.select_set(target_select)

      # Make the vertex nearest the clicked edge's first endpoint active.
      if target_select and loop_verts:
        active_vert = start_edge.verts[0]
        bm.select_history.discard(active_vert)
        bm.select_history.add(active_vert)

    bm.select_history.validate()
    bm.select_flush_mode()
    bmesh.update_edit_mesh(
      obj.data,
      loop_triangles=False,
      destructive=False,
    )
    _force_render_current(context)
    return True

  except Exception as exc:
    print("Viewport Shrink loop select error:", exc)
    return False


def _double_click_loop_select(context, settings, mx, my, shift=False, edge_hit=None):
  """Projection-aware double-click loop selection for Mesh Edit Mode."""
  if context.mode != 'EDIT_MESH':
    return False

  vert_mode, edge_mode, face_mode = context.tool_settings.mesh_select_mode
  if not (vert_mode or edge_mode):
    return False

  # Even in Vertex mode Blender's edge-loop gesture is defined by the edge
  # under/nearest the click. This also resolves which of the possible loops
  # through a vertex the user meant.
  if edge_hit is None:
    edge_hit = _pick_mesh_edge_candidate(
      context,
      settings,
      mx,
      my,
    )

  if edge_hit is None:
    return False

  return _select_mesh_loop_from_edge(
    context,
    edge_hit,
    shift=bool(shift),
  )


def _flush_selection(context):
  for obj in _edit_mode_objects(context):
    if obj.type == 'MESH':
      try:
        bm = bmesh.from_edit_mesh(obj.data)
        bm.select_history.validate()
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
      except Exception:
        pass
    elif obj.type in {'CURVE', 'SURFACE'}:
      try:
        obj.data.update_tag()
      except Exception:
        pass
  # A click must never be allowed to display the pre-click cached texture.
  _force_render_current(context)


def _pick_candidate(context, settings, mx, my):
  ui_scale = float(getattr(context.preferences.system, 'ui_scale', 1.0) or 1.0)
  radius = 10.0 * ui_scale * _PICK_HITBOX_MULTIPLIER
  radius2 = radius * radius

  vp = _display_view_projection(context, settings)
  surface_z = _visible_surface_ndc_depth(
    context,
    settings,
    mx,
    my,
    view_projection=vp,
  )

  best = None
  best_key = None

  for c in _iter_edit_candidates(
    context,
    settings,
    view_projection=vp,
  ):
    dx, dy = c['x'] - mx, c['y'] - my
    d2 = dx*dx + dy*dy
    if d2 > radius2:
      continue

    # Same visible-depth rule as the edge overlay/picker.
    if not _candidate_depth_visible(c['z'], surface_z):
      continue

    key = (round(d2, 7), c['z'])
    if best_key is None or key < best_key:
      best, best_key = c, key

  return best


def _capture_selection_state(context):
  """Capture selection so cancelled tweak drags can restore it exactly."""
  state=[]
  for obj in _edit_mode_objects(context):
    if obj.type == 'MESH':
      try:
        bm=bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.verts.index_update()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        state.append((
          'MESH',
          obj.name,
          (
            tuple(v.select for v in bm.verts),
            tuple(e.select for e in bm.edges),
            tuple(f.select for f in bm.faces),
          ),
        ))
      except Exception:
        pass
    elif obj.type in {'CURVE','SURFACE'}:
      spl=[]
      for sp in obj.data.splines:
        if sp.type == 'BEZIER':
          spl.append(('BEZIER', tuple((bp.select_control_point, bp.select_left_handle, bp.select_right_handle) for bp in sp.bezier_points)))
        else:
          spl.append(('POINTS', tuple(pt.select for pt in sp.points)))
      state.append(('CURVE', obj.name, tuple(spl)))
  return state


def _restore_selection_state(context, state):
  for kind,name,payload in state:
    obj=bpy.data.objects.get(name)
    if not obj: continue
    if kind=='MESH' and obj.type=='MESH':
      try:
        bm=bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        if (
          isinstance(payload, tuple)
          and len(payload) == 3
          and isinstance(payload[0], tuple)
        ):
          vert_state, edge_state, face_state = payload
        else:
          # Backward-safe fallback for an older captured state.
          vert_state = payload
          edge_state = ()
          face_state = ()

        for v,val in zip(bm.verts,vert_state):
          v.select_set(bool(val))
        for e,val in zip(bm.edges,edge_state):
          e.select_set(bool(val))
        for f,val in zip(bm.faces,face_state):
          f.select_set(bool(val))

        bm.select_flush_mode()
        bmesh.update_edit_mesh(
          obj.data,
          loop_triangles=False,
          destructive=False,
        )
      except Exception:
        pass
    elif kind=='CURVE' and obj.type in {'CURVE','SURFACE'}:
      for sp, saved in zip(obj.data.splines,payload):
        stype, vals=saved
        if stype=='BEZIER':
          for bp,(c,l,r) in zip(sp.bezier_points,vals):
            bp.select_control_point=bool(c); bp.select_left_handle=bool(l); bp.select_right_handle=bool(r)
        else:
          for pt,val in zip(sp.points,vals): pt.select=bool(val)
      try: obj.data.update_tag()
      except Exception: pass
  _mark_dirty(); context.area.tag_redraw()




def _active_mesh_edge(obj):
  """Return Blender's active/history edge for projected Ctrl-path selection."""
  if not obj or obj.type != 'MESH' or obj.mode != 'EDIT':
    return None

  try:
    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    bm.edges.index_update()

    active = bm.select_history.active
    if (
      isinstance(active, bmesh.types.BMEdge)
      and active.is_valid
      and not active.hide
    ):
      return active

    # Newest selected edge in history.
    for elem in reversed(list(bm.select_history)):
      if (
        isinstance(elem, bmesh.types.BMEdge)
        and elem.is_valid
        and not elem.hide
        and elem.select
      ):
        return elem

    # Fallback when exactly one edge is selected.
    selected = [e for e in bm.edges if e.select and not e.hide]
    if len(selected) == 1:
      return selected[0]

  except Exception:
    pass

  return None


def _edge_midpoint(edge):
  return (edge.verts[0].co + edge.verts[1].co) * 0.5


def _edge_transition_cost(current_edge, next_edge, shared_vert):
  """Cost for moving from one edge to another through a shared vertex.

  Primary cost is geometric distance between edge centers. A modest turn
  penalty prefers a straight continuation when several paths have nearly
  equal length, which matches how users expect Ctrl-path selection to travel
  along rows/columns of a regular quad grid.
  """
  try:
    c0 = _edge_midpoint(current_edge)
    c1 = _edge_midpoint(next_edge)
    distance = max(1e-9, float((c1 - c0).length))

    a = current_edge.other_vert(shared_vert).co
    b = next_edge.other_vert(shared_vert).co

    incoming = shared_vert.co - a
    outgoing = b - shared_vert.co

    if incoming.length_squared <= 1e-16 or outgoing.length_squared <= 1e-16:
      return distance

    incoming.normalize()
    outgoing.normalize()

    # straight => dot ~ +1 => almost no penalty
    # 90 degree => dot ~ 0 => moderate penalty
    # reverse => dot ~ -1 => larger penalty
    turn_penalty = (1.0 - incoming.dot(outgoing)) * distance * 0.35
    return distance + max(0.0, turn_penalty)

  except Exception:
    return 1.0


def _mesh_edge_shortest_path(bm, start_edge, end_edge):
  """Dijkstra over mesh EDGES, with adjacency through shared vertices."""
  if start_edge is end_edge:
    return [start_edge]

  bm.edges.ensure_lookup_table()
  bm.edges.index_update()

  start_i = int(start_edge.index)
  end_i = int(end_edge.index)

  dist = {start_i: 0.0}
  previous = {}
  heap = [(0.0, start_i)]

  while heap:
    current_dist, current_i = heapq.heappop(heap)

    if current_dist != dist.get(current_i):
      continue

    if current_i == end_i:
      break

    current = bm.edges[current_i]

    for shared_vert in current.verts:
      if shared_vert.hide:
        continue

      for next_edge in shared_vert.link_edges:
        if next_edge is current or next_edge.hide:
          continue

        ni = int(next_edge.index)
        step_cost = _edge_transition_cost(
          current,
          next_edge,
          shared_vert,
        )
        nd = current_dist + step_cost

        if nd < dist.get(ni, float('inf')) - 1e-12:
          dist[ni] = nd
          previous[ni] = current_i
          heapq.heappush(heap, (nd, ni))

  if end_i not in dist:
    return None

  indices = [end_i]
  cursor = end_i

  while cursor != start_i:
    cursor = previous.get(cursor)
    if cursor is None:
      return None
    indices.append(cursor)

  indices.reverse()
  return [bm.edges[i] for i in indices]


def _ctrl_select_projected_edge_path(context, settings, mx, my):
  """Ctrl+LMB shortest path using the visible projected destination edge."""
  if context.mode != 'EDIT_MESH':
    return False

  vert_mode, edge_mode, face_mode = context.tool_settings.mesh_select_mode
  if not edge_mode:
    return False

  # Same visible-frame, depth-aware projected edge picker as normal LMB.
  target = _pick_mesh_edge_candidate(
    context,
    settings,
    mx,
    my,
  )
  if target is None or target.get('kind') != 'MESH_EDGE':
    return False

  obj = _candidate_object(target)
  if not obj or obj.type != 'MESH' or obj.mode != 'EDIT':
    return False

  target_index = int(target.get('ei', -1))
  if target_index < 0:
    return False

  try:
    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    bm.edges.index_update()

    if target_index >= len(bm.edges):
      return False

    target_edge = bm.edges[target_index]
    start_edge = _active_mesh_edge(obj)

    # No start edge yet: establish the precisely clicked edge as start.
    if start_edge is None:
      _deselect_supported_edit(context)

      bm = bmesh.from_edit_mesh(obj.data)
      bm.edges.ensure_lookup_table()
      bm.edges.index_update()

      if target_index >= len(bm.edges):
        return False

      target_edge = bm.edges[target_index]
      target_edge.select_set(True)

      bm.select_history.clear()
      bm.select_history.add(target_edge)
      bm.select_history.validate()
      bm.select_flush_mode()

      bmesh.update_edit_mesh(
        obj.data,
        loop_triangles=False,
        destructive=False,
      )
      _force_render_current(context)
      return True

    path = _mesh_edge_shortest_path(
      bm,
      start_edge,
      target_edge,
    )
    if not path:
      return False

    path_indices = [int(e.index) for e in path]

    # Match the vertex Ctrl-path behavior: result is ONLY the path.
    _deselect_supported_edit(context)

    # Deselect can invalidate/rebuild edit selection internals, so resolve
    # fresh BMesh edge references from stable indices.
    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    bm.edges.index_update()

    for i in path_indices:
      if 0 <= i < len(bm.edges):
        bm.edges[i].select_set(True)

    # Destination edge becomes the active edge for the next Ctrl-click.
    if 0 <= target_index < len(bm.edges):
      bm.select_history.clear()
      bm.select_history.add(bm.edges[target_index])

    bm.select_history.validate()
    bm.select_flush_mode()

    bmesh.update_edit_mesh(
      obj.data,
      loop_triangles=False,
      destructive=False,
    )
    _force_render_current(context)
    return True

  except Exception as exc:
    print("Viewport Shrink Ctrl-edge-path select error:", exc)
    return False


def _active_mesh_vertex(obj):
  """Return Blender's current active/history vertex without changing selection."""
  if not obj or obj.type != 'MESH' or obj.mode != 'EDIT':
    return None

  try:
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()

    active = bm.select_history.active
    if isinstance(active, bmesh.types.BMVert) and active.is_valid and not active.hide:
      return active

    # Fallback: newest selected vertex in selection history.
    history = list(bm.select_history)
    for elem in reversed(history):
      if (
        isinstance(elem, bmesh.types.BMVert)
        and elem.is_valid
        and not elem.hide
        and elem.select
      ):
        return elem

    # Final fallback: a single selected vertex.
    selected = [v for v in bm.verts if v.select and not v.hide]
    if len(selected) == 1:
      return selected[0]
  except Exception:
    pass

  return None


def _mesh_vertex_shortest_path(bm, start_vert, end_vert):
  """Dijkstra path over mesh edges, using actual edge length as cost.

  This gives a precise source-to-destination path and never adds a neighbor
  simply because it happened to be closer to the mouse in the hidden,
  unshrunk viewport.
  """
  if start_vert is end_vert:
    return [start_vert]

  # BMVert indices are stable for this BMesh snapshot.
  bm.verts.ensure_lookup_table()
  bm.verts.index_update()

  start_i = int(start_vert.index)
  end_i = int(end_vert.index)

  dist = {start_i: 0.0}
  previous = {}
  heap = [(0.0, start_i)]

  while heap:
    current_dist, current_i = heapq.heappop(heap)
    if current_dist != dist.get(current_i):
      continue

    if current_i == end_i:
      break

    current = bm.verts[current_i]

    for edge in current.link_edges:
      if edge.hide:
        continue

      other = edge.other_vert(current)
      if other.hide:
        continue

      oi = int(other.index)

      # Geometric edge length chooses the natural direct strip on the
      # mesh when multiple topological routes exist.
      weight = max(1e-9, float(edge.calc_length()))
      nd = current_dist + weight

      if nd < dist.get(oi, float('inf')) - 1e-12:
        dist[oi] = nd
        previous[oi] = current_i
        heapq.heappush(heap, (nd, oi))

  if end_i not in dist:
    return None

  indices = [end_i]
  cursor = end_i
  while cursor != start_i:
    cursor = previous.get(cursor)
    if cursor is None:
      return None
    indices.append(cursor)

  indices.reverse()
  return [bm.verts[i] for i in indices]


def _ctrl_select_projected_vertex_path(context, settings, mx, my):
  """Ctrl+LMB shortest path using the vertex actually visible under the cursor.

  Returns True when handled. This is intentionally limited to Mesh Vertex
  Select mode, matching the workflow requested here.
  """
  if context.mode != 'EDIT_MESH':
    return False

  vert_mode, edge_mode, face_mode = context.tool_settings.mesh_select_mode
  if not vert_mode:
    return False

  # Use the same displayed-frame, depth-aware projected vertex picker as
  # ordinary LMB selection, including the enlarged convenience hitbox.
  target = _pick_candidate(context, settings, mx, my)
  if target is None or target.get('kind') != 'MESH_VERT':
    return False

  obj = _candidate_object(target)
  if not obj or obj.type != 'MESH' or obj.mode != 'EDIT':
    return False

  target_index = int(target.get('pi', -1))
  if target_index < 0:
    return False

  try:
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()

    if target_index >= len(bm.verts):
      return False

    target_vert = bm.verts[target_index]
    start_vert = _active_mesh_vertex(obj)

    # No existing active start vertex: Ctrl-click behaves like a precise
    # projected ordinary click and establishes this vertex as the start.
    if start_vert is None:
      _deselect_supported_edit(context)
      target_vert.select_set(True)
      bm.select_history.clear()
      bm.select_history.add(target_vert)
      bm.select_history.validate()
      bm.select_flush_mode()
      bmesh.update_edit_mesh(
        obj.data,
        loop_triangles=False,
        destructive=False,
      )
      _force_render_current(context)
      return True

    path = _mesh_vertex_shortest_path(bm, start_vert, target_vert)
    if not path:
      return False

    # IMPORTANT: select ONLY this source-to-target path.
    # Do not use native mesh.shortest_path_pick because its destination is
    # resolved against Blender's hidden unshrunk viewport.
    _deselect_supported_edit(context)

    # _deselect_supported_edit can refresh selection state, so resolve the
    # current BMesh again before applying the saved path indices.
    path_indices = [int(v.index) for v in path]

    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()

    for i in path_indices:
      if 0 <= i < len(bm.verts):
        bm.verts[i].select_set(True)

    # Target becomes active, exactly like the end of the Ctrl-click path.
    if 0 <= target_index < len(bm.verts):
      bm.select_history.clear()
      bm.select_history.add(bm.verts[target_index])

    bm.select_history.validate()

    # In vertex-select mode this can highlight connecting edges where both
    # endpoints are selected, but it does NOT add extra vertices.
    bm.select_flush_mode()

    bmesh.update_edit_mesh(
      obj.data,
      loop_triangles=False,
      destructive=False,
    )
    _force_render_current(context)
    return True

  except Exception as exc:
    print("Viewport Shrink Ctrl-path select error:", exc)
    return False


def _apply_click_selection(context, hit, shift):
  if hit is None:
    if not shift:
      _deselect_supported_edit(context)
      _flush_selection(context)
    return
  if shift:
    _set_candidate(hit, not _candidate_selected(hit))
  else:
    _deselect_supported_edit(context)
    _set_candidate(hit, True)
  _flush_selection(context)


def _segments_intersect_2d(a, b, c, d):
  def orient(p, q, r):
    return (
      (q[0] - p[0]) * (r[1] - p[1])
      - (q[1] - p[1]) * (r[0] - p[0])
    )

  o1 = orient(a, b, c)
  o2 = orient(a, b, d)
  o3 = orient(c, d, a)
  o4 = orient(c, d, b)

  return (
    (o1 == 0.0 or o2 == 0.0 or (o1 < 0) != (o2 < 0))
    and
    (o3 == 0.0 or o4 == 0.0 or (o3 < 0) != (o4 < 0))
  )


def _edge_intersects_box(c, xmin, xmax, ymin, ymax):
  a = (c['x0'], c['y0'])
  b = (c['x1'], c['y1'])

  def inside(p):
    return xmin <= p[0] <= xmax and ymin <= p[1] <= ymax

  if inside(a) or inside(b):
    return True

  corners = (
    (xmin, ymin),
    (xmax, ymin),
    (xmax, ymax),
    (xmin, ymax),
  )
  sides = (
    (corners[0], corners[1]),
    (corners[1], corners[2]),
    (corners[2], corners[3]),
    (corners[3], corners[0]),
  )
  return any(_segments_intersect_2d(a, b, s0, s1) for s0, s1 in sides)


def _box_apply(context, settings, start, end, mode='SET'):
  x0,y0=start; x1,y1=end
  xmin,xmax=sorted((x0,x1)); ymin,ymax=sorted((y0,y1))

  if mode=='SET':
    _deselect_supported_edit(context)

  if context.mode == 'EDIT_MESH':
    vert_mode, edge_mode, face_mode = context.tool_settings.mesh_select_mode

    if edge_mode and not vert_mode:
      for c in _iter_mesh_edge_candidates(context, settings):
        if _edge_intersects_box(c, xmin, xmax, ymin, ymax):
          if mode == 'SUB':
            _set_candidate(c, False)
          else:
            _set_candidate(c, True)
      _flush_selection(context)
      return

  for c in _iter_edit_candidates(context, settings):
    if xmin <= c['x'] <= xmax and ymin <= c['y'] <= ymax:
      if mode=='SUB':
        _set_candidate(c, False)
      else:
        _set_candidate(c, True)

  _flush_selection(context)


# -----------------------------------------------------------------------------
# Always-visible edit CV / vertex overlay
# -----------------------------------------------------------------------------
def _theme_edit_point_style(context):
  """Match Blender's normal edit-point appearance as closely as possible.

  The visibility fix is ONLY draw-order related: markers are still rendered
  after the shrunk scene so geometry cannot occlude them. Their diameter and
  colors come from Blender's own 3D View theme instead of using oversized
  custom markers.
  """
  try:
    theme = context.preferences.themes[0].view_3d

    normal = tuple(float(c) for c in theme.vertex[:3]) + (1.0,)
    selected = tuple(float(c) for c in theme.vertex_select[:3]) + (1.0,)

    # ThemeView3D.vertex_size is Blender's configured vertex diameter in
    # screen pixels. Use it directly instead of inventing our own CV size.
    diameter = float(max(1, int(theme.vertex_size)))
  except Exception:
    normal = (0.0, 0.0, 0.0, 1.0)
    selected = (1.0, 0.48, 0.05, 1.0)
    diameter = 3.0

  try:
    ui = max(0.75, float(context.preferences.system.ui_scale))
  except Exception:
    ui = 1.0

  diameter *= ui

  # Blender's default point is compact. The colored disc uses the exact theme
  # diameter; a sub-pixel/one-pixel dark rim prevents selected orange points
  # disappearing against bright geometry without visibly enlarging them.
  inner_radius = max(0.75, diameter * 0.5)
  outline_radius = inner_radius + max(0.55, 0.65 * ui)

  return normal, selected, inner_radius, outline_radius


def _point_inside_navigation_safe_area(context, x, y):
  """No reserved navigation-gizmo area in the current release."""
  return False


def _disc_triangles(points, radius, segments=12):
  """Return 2D triangles for opaque fixed-pixel discs."""
  verts = []
  for cx, cy in points:
    for i in range(segments):
      a0 = (i / segments) * 6.283185307179586
      a1 = ((i + 1) / segments) * 6.283185307179586
      verts.extend((
        (cx, cy),
        (cx + math.cos(a0) * radius, cy + math.sin(a0) * radius),
        (cx + math.cos(a1) * radius, cy + math.sin(a1) * radius),
      ))
  return verts


def _draw_disc_batch(shader, points, radius, color):
  if not points:
    return
  verts = _disc_triangles(points, radius)
  if not verts:
    return
  batch = batch_for_shader(shader, 'TRIS', {'pos': verts})
  shader.bind()
  shader.uniform_float('color', color)
  batch.draw(shader)


def _draw_projected_edit_points(context, settings):
  global _CV_OVERLAY_CONFIRMED

  """Legacy POST_PIXEL CV overlay (kept unused; depth-aware path is active).

  The prior implementation used GL/Metal point primitives. Those can become
  sub-pixel-looking pinpricks in shallow perspective views on macOS even when
  Blender is asked for a larger point size.

  This implementation draws each marker as actual 2D triangle geometry in
  POST_PIXEL space. Therefore:
   - perspective depth cannot shrink it,
   - zoom cannot shrink it,
   - surface depth cannot occlude it,
   - GPU point-size implementation cannot alter it.

  The visible marker position is still generated by the exact same custom
  XYZ-shrink projection used for picking and box selection.
  """
  if not _supported_edit_mode(context):
    return
  if not bool(getattr(context.space_data.overlay, 'show_overlays', True)):
    return

  normal_pos = []
  selected_pos = []

  for c in _iter_edit_candidates(context, settings):
    x = float(c['x'])
    y = float(c['y'])

    if _point_inside_navigation_safe_area(context, x, y):
      continue

    if c.get('selected', False):
      selected_pos.append((x, y))
    else:
      normal_pos.append((x, y))

  if not normal_pos and not selected_pos:
    return

  if not _CV_OVERLAY_CONFIRMED:
    print("Viewport Shrink: legacy CV overlay active")
    _CV_OVERLAY_CONFIRMED = True

  (
    normal_color,
    selected_color,
    inner_radius,
    outline_radius,
  ) = _theme_edit_point_style(context)

  shader = _shader_from_builtin('UNIFORM_COLOR')

  # Completely independent of the scene depth buffer.
  _gpu_depth_test_set('NONE')
  _gpu_depth_mask_set(False)

  # Use opaque passes rather than alpha-soft points. This prevents a CV from
  # appearing to "fade" as the camera approaches a grazing angle.
  _gpu_blend_set('NONE')

  # Draw compact Blender-style points.
  #
  # The important difference from native offscreen points is not their size:
  # it is that these are painted AFTER the final shrunk scene, with depth
  # testing disabled. They therefore cannot fade behind the surface.
  #
  # A tiny dark rim mirrors the crisp edge Blender normally gives edit points
  # without turning them into the large rings used by the previous build.
  outline_color = (0.015, 0.015, 0.015, 1.0)

  if normal_pos:
    _draw_disc_batch(
      shader,
      normal_pos,
      outline_radius,
      outline_color,
    )
    _draw_disc_batch(
      shader,
      normal_pos,
      inner_radius,
      normal_color,
    )

  if selected_pos:
    _draw_disc_batch(
      shader,
      selected_pos,
      outline_radius,
      outline_color,
    )
    _draw_disc_batch(
      shader,
      selected_pos,
      inner_radius,
      selected_color,
    )

  _gpu_blend_set('NONE')
  _gpu_depth_mask_set(True)

# Global overlay state is rendered by the same POST_PIXEL draw handler as the
# shrunk viewport, avoiding a second draw handler and extra redraw overhead.
_BOX_DRAW_STATE = None


class VIEWPORTSHRINK_OT_SeamlessMouse(bpy.types.Operator):
  """Projection-aware LMB selection while shrink is active.

  Object Mode:
  - LMB selects the object actually visible under the SHRUNK mouse position
  - Shift + LMB toggles/adds object selection
  - silhouette-edge clicks get a small native-style pixel tolerance

  Edit Mode:
  - Vertex mode LMB: select projected vertex/CV
  - Edge mode LMB: select projected edge
  - Shift + LMB: toggle/add selection
  - Double-click has no special loop behavior; it acts like projected LMB
  - Ctrl + LMB in Mesh Vertex/Edge mode: exact projected shortest path
  - LMB drag from empty space: box select
  - LMB drag starting on geometry: NEVER moves geometry

  Movement is handled only by the projection-aware G operator.
  """
  bl_idname='viewport_shrink.seamless_mouse'
  bl_label='Viewport Shrink Select'
  bl_options={'INTERNAL','BLOCKING'}

  _start=(0.0,0.0)
  _hit=None
  _shift=False
  _mode='PENDING'
  _selection_before=None

  @classmethod
  def poll(cls,context):
    return context.area is not None and context.area.type=='VIEW_3D'

  def invoke(self,context,event):
    global _BOX_DRAW_STATE
    s=_settings(context)
    if not s or not _is_shrunk(s):
      return {'PASS_THROUGH'}

    # Alt/Cmd keep Blender's special modifier behavior. Ctrl is handled
    # projection-aware below so shortest-path selection cannot mis-pick in
    # the hidden unshrunk viewport.
    if event.alt or event.oskey:
      return {'PASS_THROUGH'}

    mouse_x = float(event.mouse_region_x)
    mouse_y = float(event.mouse_region_y)

    # OBJECT MODE:
    # Blender's native picker still sees the hidden unshrunk viewport.
    # Replace it with a ray through the exact custom XYZ-shrink projection.
    if context.mode == 'OBJECT':
      hit_obj = _pick_object_shrunk(
        context,
        s,
        mouse_x,
        mouse_y,
      )
      _apply_object_click_selection(
        context,
        hit_obj,
        bool(event.shift),
      )
      return {'FINISHED'}

    if not _supported_edit_mode(context):
      return {'PASS_THROUGH'}

    # Ctrl + LMB: projection-aware shortest path in the ACTIVE mesh
    # selection mode. No native unshrunk picker is involved.
    if event.ctrl and context.mode == 'EDIT_MESH':
      vert_mode, edge_mode, face_mode = (
        context.tool_settings.mesh_select_mode
      )

      # Edge-only mode.
      if edge_mode and not vert_mode:
        if _ctrl_select_projected_edge_path(
          context,
          s,
          mouse_x,
          mouse_y,
        ):
          return {'FINISHED'}

      # Vertex-only mode.
      elif vert_mode and not edge_mode:
        if _ctrl_select_projected_vertex_path(
          context,
          s,
          mouse_x,
          mouse_y,
        ):
          return {'FINISHED'}

      # If both Vertex + Edge modes are enabled, use whichever projected
      # element is actually nearest the cursor.
      elif vert_mode and edge_mode:
        hit = _pick_edit_candidate(
          context,
          s,
          mouse_x,
          mouse_y,
        )

        if hit is not None and hit.get('kind') == 'MESH_EDGE':
          if _ctrl_select_projected_edge_path(
            context,
            s,
            mouse_x,
            mouse_y,
          ):
            return {'FINISHED'}
        else:
          if _ctrl_select_projected_vertex_path(
            context,
            s,
            mouse_x,
            mouse_y,
          ):
            return {'FINISHED'}

    self._start=(mouse_x,mouse_y)
    self._hit=_pick_edit_candidate(context,s,*self._start)

    self._shift=bool(event.shift)
    self._mode='PENDING'
    self._selection_before=_capture_selection_state(context)
    _BOX_DRAW_STATE=None

    # Selection happens immediately on PRESS. There is no deferred
    # tweak-drag state anymore, because LMB is never allowed to transform
    # geometry in Viewport Shrink.
    _apply_click_selection(context,self._hit,self._shift)

    context.window_manager.modal_handler_add(self)
    return {'RUNNING_MODAL'}

  def _drag_threshold(self,context):
    ui=float(getattr(context.preferences.system,'ui_scale',1.0) or 1.0)
    return max(4.0, 5.0*ui)

  def _begin_box(self,context,event):
    global _BOX_DRAW_STATE
    self._mode='BOX'
    p=self._start
    _BOX_DRAW_STATE={
      'dragging':True,
      'start':p,
      'end':(float(event.mouse_region_x),float(event.mouse_region_y)),
    }
    context.area.tag_redraw()

  def _finish(self,context):
    global _BOX_DRAW_STATE
    _BOX_DRAW_STATE=None
    context.area.tag_redraw()

  def modal(self,context,event):
    global _BOX_DRAW_STATE
    s=_settings(context)
    if not s or not _is_shrunk(s):
      self._finish(context)
      return {'CANCELLED'}

    if event.type in {'ESC','RIGHTMOUSE'} and event.value=='PRESS':
      _restore_selection_state(context,self._selection_before)
      self._finish(context)
      return {'CANCELLED'}

    if event.type=='MOUSEMOVE':
      mx,my=float(event.mouse_region_x),float(event.mouse_region_y)
      if self._mode=='PENDING':
        dx,dy=mx-self._start[0],my-self._start[1]
        if dx*dx+dy*dy >= self._drag_threshold(context)**2:
          if self._hit is None:
            # Empty-space LMB drag remains the convenient box-select
            # gesture from the previous build.
            self._begin_box(context,event)
          else:
            # A drag that starts on a CV/vertex is deliberately
            # consumed without changing coordinates. Use G to move.
            self._mode='POINT_DRAG'
      elif self._mode=='BOX':
        _BOX_DRAW_STATE['end']=(mx,my)
        context.area.tag_redraw()
      return {'RUNNING_MODAL'}

    # Blender may finish a mouse gesture as RELEASE or CLICK depending on
    # the active keymap/tool. Handle both, but do not transform anything.
    if event.type=='LEFTMOUSE' and event.value in {'RELEASE','CLICK'}:
      if self._mode=='BOX':
        end=(float(event.mouse_region_x),float(event.mouse_region_y))
        _box_apply(context,s,self._start,end,'ADD' if self._shift else 'SET')
      self._finish(context)
      return {'FINISHED'}

    return {'RUNNING_MODAL'}


# -----------------------------------------------------------------------------
# Projection-aware Box Select (B)
# -----------------------------------------------------------------------------


def _draw_box_overlay():
  state = _BOX_DRAW_STATE
  if not state or not state.get('dragging'):
    return
  x0, y0 = state['start']; x1, y1 = state['end']
  xmin, xmax = sorted((x0, x1)); ymin, ymax = sorted((y0, y1))

  _gpu_blend_set('ALPHA')
  fill_shader = _shader_from_builtin('UNIFORM_COLOR')
  fill = batch_for_shader(fill_shader, 'TRIS',
    {'pos':((xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax))},
    indices=((0,1,2),(0,2,3)))
  fill_shader.bind(); fill_shader.uniform_float('color',(0.35,0.65,1.0,0.10)); fill.draw(fill_shader)

  line_shader = _shader_from_builtin('UNIFORM_COLOR')
  line = batch_for_shader(line_shader, 'LINE_STRIP',
    {'pos':((xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax),(xmin,ymin))})
  line_shader.bind(); line_shader.uniform_float('color',(0.55,0.80,1.0,0.95))
  _gpu_line_width_set(1.0); line.draw(line_shader)
  _gpu_blend_set('NONE')


class VIEWPORTSHRINK_OT_ProjectedBoxSelect(bpy.types.Operator):
  bl_idname = "viewport_shrink.projected_box_select"
  bl_label = "Viewport Shrink Box Select"
  bl_description = "Box-select CVs/vertices in the displayed XYZ-XYZ-shrink projection"
  bl_options = {'INTERNAL', 'BLOCKING'}

  _dragging = False
  _button = None
  _subtract = False
  _extend = False

  @classmethod
  def poll(cls, context):
    return context.area is not None and context.area.type == 'VIEW_3D'

  def _cleanup(self, context):
    global _BOX_DRAW_STATE
    _BOX_DRAW_STATE = None
    try: context.area.header_text_set(None)
    except Exception: pass
    context.area.tag_redraw()

  def invoke(self, context, event):
    global _BOX_DRAW_STATE
    settings = _settings(context)
    if not settings or not _is_shrunk(settings) or not _supported_edit_mode(context):
      return {'PASS_THROUGH'}
    self._dragging = False
    self._button = None
    self._subtract = False
    self._extend = False
    _BOX_DRAW_STATE = {'dragging':False, 'start':(0,0), 'end':(0,0)}
    context.area.header_text_set("Box Select — drag: set | Shift: add | Ctrl/MMB: subtract | Esc/RMB: cancel")
    context.window_manager.modal_handler_add(self)
    return {'RUNNING_MODAL'}

  def _apply(self, context):
    settings=_settings(context)
    mode='SUB' if self._subtract else ('ADD' if self._extend else 'SET')
    _box_apply(context,settings,_BOX_DRAW_STATE['start'],_BOX_DRAW_STATE['end'],mode)

  def modal(self, context, event):
    global _BOX_DRAW_STATE
    if event.type == 'ESC' or (event.type == 'RIGHTMOUSE' and event.value == 'PRESS'):
      self._cleanup(context); return {'CANCELLED'}

    if event.type in {'LEFTMOUSE','MIDDLEMOUSE'} and event.value == 'PRESS' and not self._dragging:
      self._dragging = True
      self._button = event.type
      self._subtract = (event.type == 'MIDDLEMOUSE') or bool(event.ctrl)
      self._extend = bool(event.shift) and not self._subtract
      p=(float(event.mouse_region_x),float(event.mouse_region_y))
      _BOX_DRAW_STATE.update({'dragging':True,'start':p,'end':p})
      context.area.tag_redraw(); return {'RUNNING_MODAL'}

    if event.type == 'MOUSEMOVE' and self._dragging:
      _BOX_DRAW_STATE['end']=(float(event.mouse_region_x),float(event.mouse_region_y))
      context.area.tag_redraw(); return {'RUNNING_MODAL'}

    if self._dragging and event.type == self._button and event.value == 'RELEASE':
      _BOX_DRAW_STATE['end']=(float(event.mouse_region_x),float(event.mouse_region_y))
      self._apply(context)
      self._cleanup(context); return {'FINISHED'}

    return {'RUNNING_MODAL'}


# -----------------------------------------------------------------------------
# Snapshot helpers for G translation
# -----------------------------------------------------------------------------

def _collect_transform_snapshot(context):
  records=[]; world_points=[]
  if context.mode == 'OBJECT':
    for obj in context.selected_objects:
      mw=obj.matrix_world.copy(); records.append({'kind':'OBJECT','object_name':obj.name,'matrix_world':mw}); world_points.append(mw.translation.copy())
    return records,world_points

  for obj in _edit_mode_objects(context):
    if obj.type == 'MESH':
      bm=bmesh.from_edit_mesh(obj.data); bm.verts.ensure_lookup_table(); bm.verts.index_update(); entries=[]
      for v in bm.verts:
        if v.select and not v.hide:
          entries.append((v.index,v.co.copy())); world_points.append(obj.matrix_world @ v.co)
      if entries: records.append({'kind':'MESH','object_name':obj.name,'verts':entries})
    elif obj.type in {'CURVE','SURFACE'}:
      entries=[]
      for si,spline in enumerate(obj.data.splines):
        if spline.type == 'BEZIER':
          for pi,bp in enumerate(spline.bezier_points):
            if bp.select_control_point:
              entries.append(('BEZIER_CO',si,pi,bp.co.copy())); world_points.append(obj.matrix_world @ bp.co)
            # Handles are only independent transform records when the center is not selected.
            if bp.select_left_handle and not bp.select_control_point:
              entries.append(('BEZIER_LEFT',si,pi,bp.handle_left.copy())); world_points.append(obj.matrix_world @ bp.handle_left)
            if bp.select_right_handle and not bp.select_control_point:
              entries.append(('BEZIER_RIGHT',si,pi,bp.handle_right.copy())); world_points.append(obj.matrix_world @ bp.handle_right)
        else:
          for pi,pt in enumerate(spline.points):
            if pt.select:
              entries.append(('POINT',si,pi,pt.co.copy())); world_points.append(obj.matrix_world @ Vector(pt.co[:3]))
      if entries: records.append({'kind':'CURVE','object_name':obj.name,'points':entries})
  return records,world_points


def _apply_snapshot_delta(records, world_delta, context=None):
  touched=[]
  for r in records:
    obj=bpy.data.objects.get(r['object_name'])
    if not obj: continue
    if r['kind']=='OBJECT':
      m=r['matrix_world'].copy(); m.translation=r['matrix_world'].translation+world_delta; obj.matrix_world=m
    elif r['kind']=='MESH' and obj.type=='MESH':
      bm=bmesh.from_edit_mesh(obj.data); bm.verts.ensure_lookup_table(); local=obj.matrix_world.inverted_safe().to_3x3() @ world_delta
      for idx,orig in r['verts']:
        if idx < len(bm.verts): bm.verts[idx].co=orig+local
      touched.append(obj.data)
    elif r['kind']=='CURVE' and obj.type in {'CURVE','SURFACE'}:
      local=obj.matrix_world.inverted_safe().to_3x3() @ world_delta
      for kind,si,pi,orig in r['points']:
        if si>=len(obj.data.splines): continue
        sp=obj.data.splines[si]
        try:
          if kind=='BEZIER_CO':
            bp=sp.bezier_points[pi]; bp.co=orig+local
            # Blender moves the handles with a selected center.
            # Since bp.co assignment alone may not preserve original handle offset for all handle types,
            # native handle behavior is left to Blender's curve update where applicable.
          elif kind=='BEZIER_LEFT': sp.bezier_points[pi].handle_left=orig+local
          elif kind=='BEZIER_RIGHT': sp.bezier_points[pi].handle_right=orig+local
          elif kind=='POINT':
            p=sp.points[pi]; xyz=Vector(orig[:3])+local; p.co=(xyz.x,xyz.y,xyz.z,orig.w)
        except Exception: pass
      try: obj.data.update_tag()
      except Exception: pass
  for data in touched:
    try: bmesh.update_edit_mesh(data, loop_triangles=False, destructive=False)
    except Exception: pass
  _mark_dirty()
  if context and context.area:
    context.area.tag_redraw()
  else:
    _tag_redraw_all_view3d()


def _restore_snapshot(records, context=None):
  _apply_snapshot_delta(records, Vector((0,0,0)), context)


def _average_world(points):
  if not points: return Vector((0,0,0))
  s=Vector((0,0,0))
  for p in points: s+=p
  return s/len(points)



class VIEWPORTSHRINK_OT_ProjectedPan(bpy.types.Operator):
  """Pan at native Blender screen speed while XYZ shrink is active.

  Native Blender pan changes RegionView3D.view_location in world space.
  With a visual XYZ scale applied about that location, part of that movement
  is visually compressed as well (for example X=.3 can make a pan component
  along X appear ~3.3x slower).

  This operator measures the mouse movement through the SAME custom
  projection used to draw Viewport Shrink, then solves the required
  view_location delta in world space. The result is one screen pixel of
  mouse movement producing one matching screen pixel of pan, regardless of
  X/Y/Z shrink values.
  """
  bl_idname = "viewport_shrink.projected_pan"
  bl_label = "Viewport Shrink Pan"
  bl_description = "Pan speed automatically follows XYZ shrink and is globally boosted 2x"
  bl_options = {'INTERNAL', 'BLOCKING'}

  _start_mouse = None
  _start_view_location = None
  _start_world = None
  _inv_vp = None
  _depth_ndc = 0.0

  @classmethod
  def poll(cls, context):
    return (
      context.area is not None
      and context.area.type == 'VIEW_3D'
      and context.region is not None
      and context.region.type == 'WINDOW'
      and context.region_data is not None
    )

  def invoke(self, context, event):
    settings = _settings(context)
    if not settings or not _is_shrunk(settings):
      return {'PASS_THROUGH'}

    # Camera-view panning has camera-specific semantics. Keep Blender native
    # there rather than overriding camera framing.
    if context.region_data.view_perspective == 'CAMERA':
      return {'PASS_THROUGH'}

    self._start_mouse = Vector((
      float(event.mouse_region_x),
      float(event.mouse_region_y),
    ))
    self._start_view_location = context.region_data.view_location.copy()

    _projection, vp = _custom_projection_matrices(context, settings)
    self._inv_vp = vp.inverted_safe()

    pivot_screen = _project_world(
      context,
      self._start_view_location,
      settings,
      vp,
    )
    if pivot_screen is None:
      return {'PASS_THROUGH'}

    self._depth_ndc = pivot_screen[2]
    self._start_world = _unproject_display(
      context,
      self._start_mouse.x,
      self._start_mouse.y,
      self._depth_ndc,
      settings,
      self._inv_vp,
    )

    context.window_manager.modal_handler_add(self)
    return {'RUNNING_MODAL'}

  def modal(self, context, event):
    settings = _settings(context)
    if not settings or not _is_shrunk(settings):
      return {'CANCELLED'}

    if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
      context.region_data.view_location = self._start_view_location
      context.area.tag_redraw()
      return {'CANCELLED'}

    if event.type == 'MIDDLEMOUSE' and event.value == 'RELEASE':
      return {'FINISHED'}

    if event.type == 'MOUSEMOVE':
      current_world = _unproject_display(
        context,
        float(event.mouse_region_x),
        float(event.mouse_region_y),
        self._depth_ndc,
        settings,
        self._inv_vp,
      )

      # Dragging right/up should move the displayed scene right/up, just
      # like Blender's regular pan. Moving the view pivot in the opposite
      # world direction produces that visual motion.
      mouse_world_delta = current_world - self._start_world

      # Pan response follows the shrink continuously.
      #
      # For a movement direction d, measure how much the current XYZ
      # shrink visually compresses that direction:
      #
      #   effective_scale = |S*d| / |d|
      #
      # Then use the exact inverse as the pan-speed multiplier:
      #
      #   pan_multiplier = 1 / effective_scale
      #
      # Therefore:
      #  shrink 1.0 -> 1.00x
      #  shrink 0.5 -> 2.00x
      #  shrink 0.3 -> 3.33x
      #
      # With different X/Y/Z values this remains smooth because the
      # effective scale is calculated from the actual world-space pan
      # direction rather than choosing a single axis abruptly.
      delta_len = mouse_world_delta.length
      pan_multiplier = 1.0

      if delta_len > 1e-12:
        sx, sy, sz = _axis_scale(settings)
        visually_scaled = Vector((
          mouse_world_delta.x * sx,
          mouse_world_delta.y * sy,
          mouse_world_delta.z * sz,
        ))

        effective_scale = visually_scaled.length / delta_len
        effective_scale = max(
          _PAN_MIN_EFFECTIVE_SCALE,
          min(1.0, effective_scale),
        )
        pan_multiplier = 1.0 / effective_scale

      context.region_data.view_location = (
        self._start_view_location
        - mouse_world_delta * pan_multiplier * _PAN_GLOBAL_MULTIPLIER
      )
      context.area.tag_redraw()
      return {'RUNNING_MODAL'}

    # Keep wheel and unrelated events from unexpectedly starting native
    # navigation while this Shift+MMB gesture is active.
    return {'RUNNING_MODAL'}


class VIEWPORTSHRINK_OT_ProjectedTranslate(bpy.types.Operator):
  bl_idname="viewport_shrink.projected_translate"
  bl_label="Viewport Shrink Move"
  bl_description="Move selected CVs/vertices with cursor motion matched to the XYZ-shrink projection"
  bl_options={'REGISTER','UNDO','BLOCKING'}

  _records=None; _depth_ndc=0.0; _start_world=None; _constraint=None; _inv_vp=None

  @classmethod
  def poll(cls,context): return context.area is not None and context.area.type=='VIEW_3D'

  def _header(self,context):
    c=self._constraint or 'View Plane'
    context.area.header_text_set(f"Move — {c} | X/Y/Z constrain | Shift+X/Y/Z plane | LMB/Enter confirm | RMB/Esc cancel")

  def invoke(self,context,event):
    s=_settings(context)
    if not s or not _is_shrunk(s) or not _supported_edit_mode(context): return {'PASS_THROUGH'}
    records,points=_collect_transform_snapshot(context)
    if not records or not points: return {'PASS_THROUGH'}
    self._records=records; self._constraint=None
    center=_average_world(points)
    _proj,vp=_custom_projection_matrices(context,s); self._inv_vp=vp.inverted_safe()
    center_screen=_project_world(context,center,s,vp)
    if not center_screen: return {'PASS_THROUGH'}
    self._depth_ndc=center_screen[2]
    self._start_world=_unproject_display(context,event.mouse_region_x,event.mouse_region_y,self._depth_ndc,s,self._inv_vp)
    self._header(context); context.window_manager.modal_handler_add(self); return {'RUNNING_MODAL'}

  def _constrain(self,d):
    c=self._constraint
    if c=='X': return Vector((d.x,0,0))
    if c=='Y': return Vector((0,d.y,0))
    if c=='Z': return Vector((0,0,d.z))
    if c=='YZ': return Vector((0,d.y,d.z))
    if c=='XZ': return Vector((d.x,0,d.z))
    if c=='XY': return Vector((d.x,d.y,0))
    return d

  def modal(self,context,event):
    s=_settings(context)
    if not s: context.area.header_text_set(None); return {'CANCELLED'}
    if event.type in {'ESC','RIGHTMOUSE'} and event.value=='PRESS':
      _restore_snapshot(self._records,context); context.area.header_text_set(None); return {'CANCELLED'}
    if event.type in {'LEFTMOUSE','RET','NUMPAD_ENTER'} and event.value=='PRESS':
      context.area.header_text_set(None); return {'FINISHED'}
    if event.value=='PRESS' and event.type in {'X','Y','Z'}:
      self._constraint = ({'X':'YZ','Y':'XZ','Z':'XY'}[event.type] if event.shift else event.type); self._header(context); return {'RUNNING_MODAL'}
    if event.type=='MOUSEMOVE':
      current=_unproject_display(context,event.mouse_region_x,event.mouse_region_y,self._depth_ndc,s,self._inv_vp)
      delta=self._constrain(current-self._start_world)
      if event.shift and not self._constraint: delta*=0.1
      _apply_snapshot_delta(self._records,delta,context)
      return {'RUNNING_MODAL'}
    return {'RUNNING_MODAL'}



_AXIS_GIZMO_ICON = {
  'X': 'EVENT_X',
  'Y': 'EVENT_Y',
  'Z': 'EVENT_Z',
}


def _axis_badge_text(axis, value):
  return f"{str(axis).upper()} Axis"


class VIEWPORTSHRINK_OT_ToggleAxisLock(bpy.types.Operator):
  bl_idname = "viewport_shrink.toggle_axis_lock"
  bl_label = "Toggle Axis Lock"
  bl_description = "Toggle lock for this axis"
  bl_options = {'INTERNAL'}

  axis: StringProperty(options={'HIDDEN'})

  def execute(self, context):
    s = _settings(context)
    if not s:
      return {'CANCELLED'}

    axis = str(self.axis).upper()
    prop = {
      'X': 'lock_x',
      'Y': 'lock_y',
      'Z': 'lock_z',
    }.get(axis)

    if not prop:
      return {'CANCELLED'}

    setattr(s, prop, not bool(getattr(s, prop)))
    try:
      _tag_redraw_all_view3d()
    except Exception:
      pass
    return {'FINISHED'}


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

class VIEWPORTSHRINK_PT_Main(bpy.types.Panel):
  bl_label = "Viewport Shrink"
  bl_idname = "VIEWPORTSHRINK_PT_main"
  bl_space_type = 'VIEW_3D'
  bl_region_type = 'UI'
  bl_category = "Viewport Shrink"

  def draw(self, context):
    layout = self.layout
    s = _settings(context)
    if not s:
      layout.label(text="Settings unavailable", icon='ERROR')
      return

    box = layout.box()
    for label, value_prop, lock_prop in (
      ("X", "shrink_x", "lock_x"),
      ("Y", "shrink_y", "lock_y"),
      ("Z", "shrink_z", "lock_z"),
    ):
      locked = getattr(s, lock_prop)
      value = getattr(s, value_prop)

      # Header row:
      # left = colored axis badge using the same axis icon language
      # right = axis-colored lock toggle
      head = box.row(align=True)

      badge = head.row(align=True)
      badge.alignment = 'LEFT'
      badge.label(
        text=_axis_badge_text(label, value),
      )

      lock_row = head.row(align=True)
      lock_row.alignment = 'RIGHT'
      lock_op = lock_row.operator(
        VIEWPORTSHRINK_OT_ToggleAxisLock.bl_idname,
        text="",
        icon='LOCKED' if locked else 'UNLOCKED',
      )
      lock_op.axis = label

      # Native slider stays intact for direct interaction.
      slider = box.row()
      slider.enabled = not locked
      slider.prop(s, value_prop, text="", slider=True)

      presets = box.row(align=True)
      presets.enabled = not locked

      op = presets.operator("viewport_shrink.axis_preset", text=".3")
      op.axis = label
      op.value = 0.3

      op = presets.operator("viewport_shrink.axis_preset", text="1")
      op.axis = label
      op.value = 1.0

      if label != "Z":
        box.separator(factor=0.7)


    layout.separator()

    shortcut_row = layout.row()
    shortcut_row.scale_y = 1.12
    shortcut_row.operator(
      VIEWPORTSHRINK_OT_ShortcutPopup.bl_idname,
      text="Set Shortcut",
      icon='KEYINGSET',
    )

    layout.operator(
      "viewport_shrink.reset",
      text="Reset Viewport",
      icon='LOOP_BACK',
    )


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

_classes=(VIEWPORTSHRINK_PG_CustomPresetShortcut, VIEWPORTSHRINK_PG_Settings, VIEWPORTSHRINK_OT_Preset03, VIEWPORTSHRINK_OT_Preset05, VIEWPORTSHRINK_OT_Preset10, VIEWPORTSHRINK_OT_AxisPreset, VIEWPORTSHRINK_OT_AddCustomPresetShortcut, VIEWPORTSHRINK_OT_RemoveCustomPresetShortcut, VIEWPORTSHRINK_OT_CaptureCustomPresetShortcut, VIEWPORTSHRINK_OT_CaptureShortcut, VIEWPORTSHRINK_OT_SliderGesture, VIEWPORTSHRINK_OT_ClearSliderShortcut, VIEWPORTSHRINK_OT_CaptureSliderShortcut, VIEWPORTSHRINK_OT_ClearResetShortcut, VIEWPORTSHRINK_OT_CaptureResetShortcut, VIEWPORTSHRINK_OT_ShortcutPopup, VIEWPORTSHRINK_OT_Reset, VIEWPORTSHRINK_OT_SeamlessMouse, VIEWPORTSHRINK_OT_ProjectedBoxSelect, VIEWPORTSHRINK_OT_ProjectedPan, VIEWPORTSHRINK_OT_ProjectedTranslate, VIEWPORTSHRINK_OT_ToggleAxisLock, VIEWPORTSHRINK_PT_Main)


def register():
  global _DRAW_HANDLE
  for cls in _classes: bpy.utils.register_class(cls)
  bpy.types.WindowManager.viewport_shrink_settings=bpy.props.PointerProperty(type=VIEWPORTSHRINK_PG_Settings)
  _DRAW_HANDLE=bpy.types.SpaceView3D.draw_handler_add(_draw_viewport_shrink,(), 'WINDOW','POST_PIXEL')

  if _depsgraph_dirty_handler not in bpy.app.handlers.depsgraph_update_post:
    bpy.app.handlers.depsgraph_update_post.append(_depsgraph_dirty_handler)
  if _frame_dirty_handler not in bpy.app.handlers.frame_change_post:
    bpy.app.handlers.frame_change_post.append(_frame_dirty_handler)

  wm=bpy.context.window_manager
  if wm and wm.keyconfigs and wm.keyconfigs.addon:
    km=wm.keyconfigs.addon.keymaps.new(name='3D View',space_type='VIEW_3D',region_type='WINDOW')
    # Consume DOUBLE_CLICK through the same projected selection operator.
    # There is intentionally NO loop-select action attached to double click.
    kmi=km.keymap_items.new(
      VIEWPORTSHRINK_OT_SeamlessMouse.bl_idname,
      type='LEFTMOUSE',
      value='DOUBLE_CLICK',
      any=True,
      head=True,
    )
    _ADDON_KEYMAPS.append((km,kmi))

    # PRESS feels like native Blender selection and consumes the event before
    # the hidden, unshrunk native picker can act on a different element.
    kmi=km.keymap_items.new(
      VIEWPORTSHRINK_OT_SeamlessMouse.bl_idname,
      type='LEFTMOUSE',
      value='PRESS',
      any=True,
      head=True,
    )
    _ADDON_KEYMAPS.append((km,kmi))
    kmi=km.keymap_items.new(VIEWPORTSHRINK_OT_ProjectedBoxSelect.bl_idname,type='B',value='PRESS',head=True); _ADDON_KEYMAPS.append((km,kmi))
    # Replace only Blender's normal Shift+MMB pan while shrink is active.
    # Orbit (plain MMB) remains entirely native.
    kmi=km.keymap_items.new(VIEWPORTSHRINK_OT_ProjectedPan.bl_idname,type='MIDDLEMOUSE',value='PRESS',shift=True,head=True); _ADDON_KEYMAPS.append((km,kmi))
    kmi=km.keymap_items.new(VIEWPORTSHRINK_OT_ProjectedTranslate.bl_idname,type='G',value='PRESS',head=True); _ADDON_KEYMAPS.append((km,kmi))
  try:
    _cleanup_legacy_slider_mouse_shortcuts(bpy.context)
  except Exception:
    pass

  _mark_dirty(); _tag_redraw_all_view3d()


def unregister():
  global _DRAW_HANDLE
  for km,kmi in reversed(_ADDON_KEYMAPS):
    try: km.keymap_items.remove(kmi)
    except Exception: pass
  _ADDON_KEYMAPS.clear()
  if _DRAW_HANDLE is not None:
    try: bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLE,'WINDOW')
    except Exception: pass
    _DRAW_HANDLE=None
  try:
    while _depsgraph_dirty_handler in bpy.app.handlers.depsgraph_update_post:
      bpy.app.handlers.depsgraph_update_post.remove(_depsgraph_dirty_handler)
  except Exception:
    pass
  try:
    while _frame_dirty_handler in bpy.app.handlers.frame_change_post:
      bpy.app.handlers.frame_change_post.remove(_frame_dirty_handler)
  except Exception:
    pass
  _free_offscreens()
  if hasattr(bpy.types.WindowManager,"viewport_shrink_settings"): del bpy.types.WindowManager.viewport_shrink_settings
  for cls in reversed(_classes):
    try: bpy.utils.unregister_class(cls)
    except Exception: pass
  _tag_redraw_all_view3d()


if __name__ == "__main__":
  try: unregister()
  except Exception: pass
  register()
