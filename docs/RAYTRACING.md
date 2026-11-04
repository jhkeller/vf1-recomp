# Ray-traced lighting (fork work in progress)

Goal: an opt-in "Enhanced" 3D renderer with ray-traced shadows, floor reflections and one-bounce global
illumination, beside the arcade-exact CPU and GPU renderers (which stay as they are).

Plan: https://claude.ai/artifact/CxsJ2TfuvCrxLyN3LPSvU7

| Phase | What | State |
| --- | --- | --- |
| 1 | Scene capture: every quad in view space, before culling/clipping, with its unlit colour, normal and light mode | Done |
| 2 | Depth-buffered raster of the capture, in the game's own colours (CPU) | Done |
| 3 | Ray-traced shadows through a BVH, the game's MOIRE blob shadows off (CPU; GPU compute port to come) | Done (CPU) |
| 4 | Floor reflections, the sky panorama for misses | |
| 5 | One-bounce GI, temporal accumulation, denoiser | |
| 6 | Metal, menu settings, tuning | |

## Phase 1: checking the capture

```
VF_RT_DUMP=rt_dump build/bin/vf-native
```

writes `rt_dump/rt_<frame>.obj` (+ `.mtl`) every 60th picture, 10 of them (`VF_RT_DUMP_EVERY`,
`VF_RT_DUMP_COUNT`). Open one in Blender (File > Import > Wavefront): the camera is at the origin
looking down -Z, each viewport is a group (`view0`, ...), and the polygons the game culled are in
`view*_culled`. Each view's light vector is in the file's header comments.

`VF_RT_CAPTURE=1` gathers the scene (`model1::m_rt_scene`) without writing it. With neither set,
nothing changes.

Code: `src/machine/model1_rt.cpp`, the `rt_*` types in `src/machine/model1.h`, and the hooks in
`push_object` / `tgp_render` (`src/machine/model1_video.cpp`).

## Phases 2 and 3: the ray-traced renderer

```
VF_RT=shadows build/bin/vf-native   # ray-traced shadows instead of the game's blob shadows
VF_RT=raster  build/bin/vf-native   # the same renderer without them: should look like the arcade's
```

Code: `src/raytrace/rt_render.cpp` (`rt::render`), called from `model1::rt_end`. Each viewport's
polygons are drawn with a depth buffer in the game's paint order; each lit pixel then sends a ray
toward the game's light through a BVH of every polygon in its viewport, culled ones included. A
pixel in shadow takes the colour the game would give it with only the ambient light (captured per
polygon as `rt::quad::shade`). It runs on the CPU, threaded by rows, so keep the render scale
modest (2x or 3x) for now; the GPU compute port is the next step. `VF_RT` turns the GPU 3D renderer
off for that run.

`build/bin/rttest [dir]` draws a made-up scene (no ROM) both ways, checks pixels in and out of
shadow, and writes `rt_raster.ppm` / `rt_shadows.ppm`.

Known limits: culled polygons take their palette colour, without the random fighters' recolouring.
