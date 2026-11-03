# Ray-traced lighting (fork work in progress)

Goal: an opt-in "Enhanced" 3D renderer with ray-traced shadows, floor reflections and one-bounce global
illumination, beside the arcade-exact CPU and GPU renderers (which stay as they are).

Plan: https://claude.ai/artifact/CxsJ2TfuvCrxLyN3LPSvU7

| Phase | What | State |
| --- | --- | --- |
| 1 | Scene capture: every quad in view space, before culling/clipping, with its unlit colour, normal and light mode | Done |
| 2 | Depth-buffered G-buffer raster, re-lit from the capture to match the game's own shading | Next |
| 3 | Ray-traced shadows (compute BVH), the game's MOIRE blob shadows off | |
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

Known limits: culled polygons take their palette colour, without the random fighters' recolouring.
