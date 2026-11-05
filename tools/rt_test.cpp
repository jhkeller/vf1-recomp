// The ray-traced renderer (src/raytrace) on a scene made up by hand, no ROM needed: a floor, a
// box standing on it and one floating over it, lit from above and to the left. Writes the
// picture raster-only and with shadows as PPMs, and checks a few pixels: the floor in the boxes'
// shadows takes its shadow colour, the floor away from them stays lit, and the near box hides the far one.
//   build/bin/rttest [out_dir]
#include "../src/raytrace/rt_render.h"
#ifdef RT_TEST_GPU
#include "../src/raytrace/rt_gpu.h"
#include <SDL3/SDL.h>
#include <memory>
#endif
#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
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

// draws a picture: the CPU's renderer, or (rttest-gpu) the GPU's
using draw_fn = std::function<result(const scene &, const target &, const options &)>;

static int run(const std::string &dir, float sign, const draw_fn &draw)
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
	// a marking painted on the floor, far off (in the floor's plane: painted later, it must win every pixel,
	// and the floor under it can't shadow it)
	const float mark[4][3] = { { 203, -100, 1003 }, { 199, -100, 1297 }, { 283, -100, 1301 }, { 279, -100, 999 } };
	add_quad(s, mark, 0xe0e0e0, 0x606060, STAGE, 1300);

	target t;
	std::vector<uint32_t> px(size_t(W) * H, 0x202020);
	t.px = px.data(); t.w = W; t.h = H; t.stride = W; t.cx0 = 0; t.cy0 = 0; t.cx1 = W - 1; t.cy1 = H - 1;

	options o;
	result r = draw(s, t, o);
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
	auto mark_whole = [&](const char *what) {
		int bad = 0, n = 0;
		for (float z = 1030; z <= 1270; z += 4)
			for (float x = 210; x <= 270; x += 2) {
				int sx, sy;
				project(x, -100, z, sx, sy);
				n++;
				if ((px[size_t(sy) * W + sx] & 0xffffff) != 0xe0e0e0) bad++;
			}
		printf("  %-34s %d of %d pixels %s\n", what, n - bad, n, bad ? "WRONG" : "ok");
		if (bad) fails++;
	};
	mark_whole("marking on the floor, every pixel");
	expect("red box's front", -10, -40, 450, 0xc04040);
	expect("floor under the game's shadow", -200, -100, 340, (floor_lit >> 1) & 0x7f7f7f);

	std::fill(px.begin(), px.end(), 0x202020);
	o.shadows = true;
	o.game_shadows = false;
	r = draw(s, t, o);
	write_ppm(dir + "/rt_shadows.ppm", px, W, H);
	printf("shadows: %zu rays, %zu in shadow\n", r.rays, r.shadowed);
	// the red box's shadow falls from its top (y 20, 120 over the floor) away from the light: +x, +z
	// by 120 * (0.5 / 0.8) = 75 and 120 * (0.3 / 0.8) = 45
	expect("floor in the red box's shadow", 40 + 30, -100, 520 + 40, floor_shade);
	expect("floor, open", -300, -100, 1200, floor_lit);
	expect("floor beside the red box (lit side)", -120, -100, 500, floor_lit);
	// the floating box's shadow: 140 to 200 over the floor, so 87 to 125 in x, 52 to 75 in z further on
	expect("floor in the blue box's shadow", 160 + 105, -100, 690 + 63, floor_shade);
	mark_whole("marking on the floor, lit, every pixel");
	expect("red box's front", -10, -40, 450, 0xc04040);
	expect("floor where the game's shadow was", -200, -100, 340, floor_lit);

	// global illumination: the floor at the red box's foot loses some of its ambient light (ambient
	// occlusion) against the open floor, and takes on some of the box's red
	s.lightparams.assign(256, light_param{ 0.4f, 0.6f, 0, 0 });
	o.gi = true;
	o.gi_rays = 32;
	auto at = [&](float x, float y, float z) { int sx, sy; project(x, y, z, sx, sy); return px[size_t(sy) * W + sx] & 0xffffff; };
	auto r8 = [](uint32_t c) { return int((c >> 16) & 255); };
	auto lum = [](uint32_t c) { return int((c >> 16) & 255) + int((c >> 8) & 255) + int(c & 255); };
	auto check = [&](const char *what, bool ok, uint32_t a, uint32_t b) {
		printf("  %-34s %06x vs %06x %s\n", what, a, b, ok ? "ok" : "WRONG");
		if (!ok) fails++;
	};
	// (the occlusion alone first: no bounced light)
	o.gi_strength = 0;
	for (int k = 0; k < 8; k++) { std::fill(px.begin(), px.end(), 0x202020); r = draw(s, t, o); }
	{
		const uint32_t foot = at(-80, -100, 470), open = at(-300, -100, 1200);
		check("floor at the box's foot darker (AO)", lum(foot) < lum(open), foot, open);
	}
	o.gi_strength = options().gi_strength;
	for (int k = 0; k < 8; k++) { std::fill(px.begin(), px.end(), 0x202020); r = draw(s, t, o); }
	write_ppm(dir + "/rt_gi.ppm", px, W, H);
	printf("gi: %zu rays a picture\n", r.gi_rays);
	{
		const uint32_t foot = at(-80, -100, 470), open = at(-300, -100, 1200);
		check("floor at the box's foot redder (bounce)", r8(foot) > r8(open), foot, open);
	}
	return fails;
}

// rttest --scene <raster|shadows|gi|both> <out dir> scene_0000.rts ...: the scenes the game recorded
// (F10 in the game) drawn in turn as the game would, each written as a PPM, with the time each took.
static int replay(int argc, char **argv, const draw_fn &draw)
{
	if (argc < 5) { fprintf(stderr, "rttest --scene <raster|shadows|gi|both> <out dir> <scene.rts>...\n"); return 2; }
	const std::string mode = argv[2], out = argv[3];
	options o;
	o.shadows = mode == "shadows" || mode == "both";
	o.gi = mode == "gi" || mode == "both";
	o.game_shadows = !o.shadows;
	double total = 0;
	int n = 0;
	for (int a = 4; a < argc; a++) {
		FILE *f = fopen(argv[a], "rb");
		if (!f) { fprintf(stderr, "can't read %s\n", argv[a]); return 1; }
		scene_file_header h;
		scene s;
		bool ok = fread(&h, sizeof h, 1, f) == 1 && !memcmp(h.magic, "VFRT1", 5) && h.quad_size == sizeof(quad) && h.view_size == sizeof(view);
		if (ok) {
			s.quads.resize(h.quads); s.views.resize(h.views); s.lightparams.resize(h.lightparams);
			ok = fread(s.quads.data(), sizeof(quad), h.quads, f) == h.quads && fread(s.views.data(), sizeof(view), h.views, f) == h.views &&
			     fread(s.lightparams.data(), sizeof(light_param), h.lightparams, f) == h.lightparams;
		}
		fclose(f);
		if (!ok) { fprintf(stderr, "%s: not a scene this build can read\n", argv[a]); return 1; }
		std::vector<uint32_t> px(size_t(h.w) * h.h, 0x000000);
		target t;
		t.px = px.data(); t.w = h.w; t.h = h.h; t.stride = size_t(h.w);
		t.cx0 = h.cx0; t.cy0 = h.cy0; t.cx1 = h.cx1; t.cy1 = h.cy1;
		const auto t0 = std::chrono::steady_clock::now();
		const result r = draw(s, t, o);
		const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
		total += ms; n++;
		printf("%s: %ux%u, %u quads, %zu tris, %zu shadow rays (%zu in shadow), %zu gi rays: %.2f ms\n",
			argv[a], h.w, h.h, h.quads, r.tris, r.rays, r.shadowed, r.gi_rays, ms);
		for (auto &p : px) p &= 0xffffff;
		char name[32];
		snprintf(name, sizeof name, "/%04d.ppm", n - 1);
		write_ppm(out + name, px, h.w, h.h);
	}
	if (n) printf("%d scenes, %.2f ms each on average\n", n, total / n);
	return 0;
}

int main(int argc, char **argv)
{
	draw_fn draw = [](const scene &s, const target &t, const options &o) { return render(s, t, o); };
#ifdef RT_TEST_GPU
	// rttest-gpu: the GPU's renderer (a Vulkan device: SDL's offscreen video driver, no window)
	SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "offscreen");
	if (!SDL_Init(SDL_INIT_VIDEO)) { fprintf(stderr, "SDL: %s\n", SDL_GetError()); return 1; }
	std::unique_ptr<gpu_renderer> gpu = gpu_create();
	if (!gpu) { fprintf(stderr, "no GPU renderer here\n"); return 1; }
	printf("GPU: %s\n", gpu->name());
	draw = [&](const scene &s, const target &t, const options &o) {
		result r;
		if (!gpu->render(s, t, o, r)) { fprintf(stderr, "the GPU didn't draw it\n"); exit(1); }
		return r;
	};
#endif
	if (argc > 1 && !strcmp(argv[1], "--scene")) return replay(argc, argv, draw);
	const std::string dir = argc > 1 ? argv[1] : ".";
	const int fails = run(dir, 1, draw) + run(dir, -1, draw);
	printf(fails ? "%d WRONG\n" : "all ok\n", fails);
	return fails ? 1 : 0;
}
