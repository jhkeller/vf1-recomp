// The ray-traced renderer (src/raytrace) on a scene made up by hand, no ROM needed: a floor, a
// box standing on it and one floating over it, lit from above and to the left. Writes the
// picture raster-only and with shadows as PPMs, and checks a few pixels: the floor in the boxes'
// shadows takes its shadow colour, the floor away from them stays lit, and the near box hides the far one.
//   build/bin/rttest [out_dir]
#include "../src/raytrace/rt_render.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace rt;

static float g_normal_sign = 1;   // (-1: the normals into the models, the light vector away from the light)

static void add_quad(scene &s, const float (&p)[4][3], uint32_t lit, uint32_t shade, uint8_t flags, float sortz)
{
	quad q;
	for (int i = 0; i < 4; i++) for (int k = 0; k < 3; k++) q.v[i][k] = p[i][k];
	// its normal: out of the model (this winding's cross product) or, as a game might have them, in
	const float e1[3] = { p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2] }, e2[3] = { p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2] };
	float n[3] = { e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
	const float l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
	for (int k = 0; k < 3; k++) q.n[k] = l > 0 ? n[k] / l * g_normal_sign : 0;
	q.lit = lit; q.shade = shade; q.albedo = lit; q.flags = flags; q.sortz = sortz;
	s.quads.push_back(q);
	s.views.back().count++;
}

// an axis-aligned box: its six faces, wound to face out (the game's back faces culled)
static void add_box(scene &s, float x0, float y0, float z0, float x1, float y1, float z1, uint32_t lit, uint32_t shade)
{
	const float f[6][4][3] = {
		{ { x0, y0, z0 }, { x0, y1, z0 }, { x1, y1, z0 }, { x1, y0, z0 } },   // front (-z, toward the camera)
		{ { x1, y0, z1 }, { x1, y1, z1 }, { x0, y1, z1 }, { x0, y0, z1 } },   // back
		{ { x0, y0, z1 }, { x0, y1, z1 }, { x0, y1, z0 }, { x0, y0, z0 } },   // left
		{ { x1, y0, z0 }, { x1, y1, z0 }, { x1, y1, z1 }, { x1, y0, z1 } },   // right
		{ { x0, y1, z0 }, { x0, y1, z1 }, { x1, y1, z1 }, { x1, y1, z0 } },   // top
		{ { x0, y0, z1 }, { x0, y0, z0 }, { x1, y0, z0 }, { x1, y0, z1 } },   // bottom
	};
	for (int i = 0; i < 6; i++) {
		// culled when facing away from the camera at the origin (as the game's determinant test)
		const float *a = f[i][0], *b = f[i][1], *c = f[i][2];
		const float e1[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] }, e2[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
		const float n[3] = { e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
		const bool away = n[0] * a[0] + n[1] * a[1] + n[2] * a[2] > 0;
		add_quad(s, f[i], lit, shade, away ? CULLED : 0, std::max({ a[2], b[2], c[2] }));
	}
}

static void write_ppm(const std::string &path, const std::vector<uint32_t> &px, int w, int h)
{
	FILE *f = fopen(path.c_str(), "wb");
	if (!f) { fprintf(stderr, "can't write %s\n", path.c_str()); return; }
	fprintf(f, "P6\n%d %d\n255\n", w, h);
	for (uint32_t p : px) { const unsigned char c[3] = { uint8_t(p >> 16), uint8_t(p >> 8), uint8_t(p) }; fwrite(c, 1, 3, f); }
	fclose(f);
}

static int run(const std::string &dir, float sign)
{
	g_normal_sign = sign;
	printf("normals %s the models:\n", sign > 0 ? "out of" : "into");
	const int scale = 2, W = 496 * scale, H = 384 * scale;
	scene s;
	view v;
	v.xc = 248; v.yc = 192; v.zoomx = v.zoomy = 300; v.scale = scale;
	v.x1 = 0; v.y1 = 0; v.x2 = W - 1; v.y2 = H - 1;
	v.light[0] = -0.5f * sign; v.light[1] = 0.8f * sign; v.light[2] = -0.3f * sign;   // toward the light (sign 1): up, left, toward the camera
	s.views.push_back(v);

	const uint32_t floor_lit = 0x60a060, floor_shade = 0x284828;
	const float floor[4][3] = { { -400, -100, 150 }, { -400, -100, 1400 }, { 400, -100, 1400 }, { 400, -100, 150 } };
	add_quad(s, floor, floor_lit, floor_shade, STAGE, 1400);
	add_box(s, -60, -100, 450, 40, 20, 550, 0xc04040, 0x501818);     // standing on the floor
	add_box(s, 120, 40, 650, 200, 100, 730, 0x4060d0, 0x182858);    // floating
	// one of the game's shadow polygons, a little over the floor (drawn raster-only: it halves what's under it)
	const float blob[4][3] = { { -250, -99, 300 }, { -250, -99, 380 }, { -150, -99, 380 }, { -150, -99, 300 } };
	add_quad(s, blob, 0x000000, 0x000000, MOIRE, 380);

	target t;
	std::vector<uint32_t> px(size_t(W) * H, 0x202020);
	t.px = px.data(); t.w = W; t.h = H; t.stride = W; t.cx0 = 0; t.cy0 = 0; t.cx1 = W - 1; t.cy1 = H - 1;

	options o;
	result r = render(s, t, o);
	write_ppm(dir + "/rt_raster.ppm", px, W, H);
	printf("raster: %zu triangles\n", r.tris);

	auto project = [&](float x, float y, float z, int &sx, int &sy) {
		sx = int((v.xc + x / z * v.zoomx + v.viewx + v.ext) * v.scale);
		sy = int((v.yc - (y / z * v.zoomy + v.viewy)) * v.scale);
	};
	int fails = 0;
	auto expect = [&](const char *what, float x, float y, float z, uint32_t want) {
		int sx, sy;
		project(x, y, z, sx, sy);
		const uint32_t got = px[size_t(sy) * W + sx] & 0xffffff;
		printf("  %-34s (%4d,%4d) %06x %s\n", what, sx, sy, got, got == want ? "ok" : "WRONG");
		if (got != want) { printf("    wanted %06x\n", want); fails++; }
	};
	expect("floor, open", -300, -100, 1200, floor_lit);
	expect("red box's front", -10, -40, 450, 0xc04040);
	expect("floor under the game's shadow", -200, -100, 340, (floor_lit >> 1) & 0x7f7f7f);

	std::fill(px.begin(), px.end(), 0x202020);
	o.shadows = true;
	o.game_shadows = false;
	r = render(s, t, o);
	write_ppm(dir + "/rt_shadows.ppm", px, W, H);
	printf("shadows: %zu rays, %zu in shadow\n", r.rays, r.shadowed);
	// the red box's shadow falls from its top (y 20, 120 over the floor) away from the light: +x, +z
	// by 120 * (0.5 / 0.8) = 75 and 120 * (0.3 / 0.8) = 45
	expect("floor in the red box's shadow", 40 + 30, -100, 520 + 40, floor_shade);
	expect("floor, open", -300, -100, 1200, floor_lit);
	expect("floor beside the red box (lit side)", -120, -100, 500, floor_lit);
	// the floating box's shadow: 140 to 200 over the floor, so 87 to 125 in x, 52 to 75 in z further on
	expect("floor in the blue box's shadow", 160 + 105, -100, 690 + 63, floor_shade);
	expect("red box's front", -10, -40, 450, 0xc04040);
	expect("floor where the game's shadow was", -200, -100, 340, floor_lit);
	return fails;
}

int main(int argc, char **argv)
{
	const std::string dir = argc > 1 ? argv[1] : ".";
	const int fails = run(dir, 1) + run(dir, -1);
	printf(fails ? "%d WRONG\n" : "all ok\n", fails);
	return fails ? 1 : 0;
}
