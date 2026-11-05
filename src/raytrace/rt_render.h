// Ray-traced lighting's renderer (on the CPU for now): draws an rt::scene into a picture.
//
// Each view's polygons are drawn with a depth buffer (1/z, perspective-correct), in the order the
// game would have painted them, so where two lie at the same depth (a mark on the floor) the one
// painted later still wins, as on the arcade. Each pixel keeps which polygon it shows (a G-buffer
// of polygon ids and depths), and a pass over the pixels then colours them:
//  - raster: the game's own colour (the picture as the arcade's, depth-buffered);
//  - shadows: a ray from each lit pixel toward the light, through a BVH of every polygon in its
//    view, the ones the game culled too (they cast shadows); a pixel the ray finds blocked takes
//    its colour in shadow (the light's direct part gone). The game's own shadow polygons are left out.
//  - gi (global illumination): from points on a coarse grid over the picture (a few render pixels
//    apart), rays over the hemisphere above the surface; what they hit sends back its colour as
//    lit there (the game's ambient and diffuse, in shadow or not), and the rays that hit something
//    near darken the ambient light (ambient occlusion). Kept from picture to picture where the same
//    polygon is under a grid point (less noise), and spread over the pixels between, polygon by polygon.
// The game's shadow polygons, when drawn, shade a pixel once, as model1::draw_hline_moired does.
#pragma once
#include "rt_scene.h"
#include <cstddef>
#include <cstdint>

namespace rt {

struct target
{
	uint32_t *px = nullptr;   // 0x00rrggbb pixels (the top byte: the machine's marks)
	int w = 0, h = 0;
	size_t stride = 0;        // pixels a row
	int cx0 = 0, cy0 = 0, cx1 = -1, cy1 = -1;   // the clip (inclusive)
};

struct options
{
	bool shadows = false;        // ray-traced shadows
	bool gi = false;             // ray-traced global illumination (one bounce) and ambient occlusion
	int gi_rays = 6;             // rays a grid point a picture
	int gi_step = 0;             // render pixels between grid points (0: twice the render scale)
	float gi_strength = 1.0f;    // how much bounced light is added
	float ao_strength = 0.9f;    // how far ambient occlusion darkens the ambient light (0 none, 1 fully)
	float ao_direct = 0.5f;      // and the direct light, as a share of that (flat-shaded models' creases show it)
	float ao_radius = 0.3f;      // how near a hit darkens, as a share of the point's depth
	int gi_view = 0;             // 0 the picture; debugging: 1 ambient occlusion alone (grey), 2 bounced light alone
	bool game_shadows = true;    // the game's shadow polygons
	bool blend_shadows = true;   // they halve the pixel beneath (above 1x); false: the arcade's checkerboard
	uint32_t key = 0xff000000;          // a pixel with nothing under it (the GPU's parts: the 2D goes there later)
	uint32_t shadow_key = 0xfd000000;   // a shadow over that: | its half colour
	uint32_t mark = 0x01000000;         // a pixel already shaded by a shadow polygon
};

struct result
{
	int mark_y1 = 1 << 30, mark_y2 = -1;   // rows holding marks (the caller clears them)
	size_t tris = 0, rays = 0, shadowed = 0, gi_rays = 0;
};

result render(const scene &s, const target &t, const options &o);

} // namespace rt
