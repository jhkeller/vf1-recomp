// Ray-traced lighting: the 3D scene as the game built it (see docs/RAYTRACING.md).
//
// model1::push_object hands every quad here (model1_rt.cpp) in view space, before culling,
// clipping and projection, with its colour unlit, its colour as the game lit it, and its colour
// with the light's direct part taken away (what it looks like in shadow). View space is the
// game's: x right, y up, z forward (into the screen). Nothing here depends on the machine, so the
// renderer (rt_render.h) can be tried on scenes made up by hand (tools/rt_test.cpp).
#pragma once
#include <cstdint>
#include <vector>

namespace rt {

enum flag : uint8_t
{
	MOIRE = 1,        // one of the game's shadow polygons (a ray-traced picture leaves them out)
	STAGE = 2,        // from a stage's model
	CULLED = 4,       // facing away: not drawn by the game, but it casts shadows and shows in reflections
	UNLIT = 8,        // a flat UI element the game draws unlit (casts and takes no shadow)
	TWO_SIDED = 16,   // never culled (the polygon's flags bit 14)
	RECOLOURED = 32,  // albedo is the game's final colour (random fighter, build effect); culled ones keep their pen's
};

struct quad
{
	float v[4][3];          // view space, in the game's winding (old_p1, old_p0, p0, p1)
	float n[3];             // unit normal, view space (the game's, turned by the object's matrix)
	uint32_t albedo = 0;    // 0x00rrggbb: unlit (fully lit, as the game's colour table has it)
	uint32_t lit = 0;       // 0x00rrggbb: the colour the game drew it (0 for culled ones)
	uint32_t shade = 0;     // 0x00rrggbb: its colour with the light's direct part gone (in shadow); = lit when the light can't reach it anyway
	uint32_t obj = 0;       // its model's address (the polygons' list in ROM / RAM)
	float sortz = 0;        // the depth the game sorted it by (far to near; ties in list order)
	uint16_t view = 0;      // which view it belongs to
	uint8_t lightmode = 0;  // index into the light parameters (0x80 | n: the second bank)
	uint8_t flags = 0;      // flag
};

struct view
{
	// a point's place on screen, in render pixels (model1::view_t::project_point):
	//   x = (xc + (vx / vz * zoomx + viewx) + ext) * scale,  y = (yc - (vy / vz * zoomy + viewy)) * scale
	float xc = 0, yc = 0, zoomx = 0, zoomy = 0, viewx = 0, viewy = 0, ext = 0, scale = 1;
	int x1 = 0, y1 = 0, x2 = 0, y2 = 0;    // its clip, render pixels (inclusive)
	float light[3] = {};                   // toward the light (unit, but Sarah's stage keeps its length)
	bool spec_enable = false;
	uint32_t first = 0, count = 0;         // its quads in scene::quads
};

struct light_param { float a = 0, d = 0, s = 0; int p = 0; };   // ambient, diffuse, specular, power

struct scene
{
	std::vector<quad> quads;
	std::vector<view> views;
	std::vector<light_param> lightparams;   // 256, as the list left them
	uint64_t frame = 0;
	double moment = 0;                      // smooth motion's moment drawn
	void clear() { quads.clear(); views.clear(); }
};

} // namespace rt
