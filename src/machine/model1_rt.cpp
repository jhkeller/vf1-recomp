// Ray-traced lighting: the scene capture (src/raytrace/rt_scene.h), and handing it to the renderer.
//
// The arcade's renderer flattens each polygon to one colour and draws them far to near with no
// depth buffer, so the picture it leaves has nothing left to light again. push_object hands every
// quad here as well, in view space and before culling, clipping and projection, with its colour
// unlit, lit and in shadow, its normal and its light mode; tgp_render marks where a picture's list
// starts, where each viewport begins and where it ends. With VF_RT set, rt_end draws the picture
// from it (src/raytrace/rt_render.cpp) where the arcade's rasterizer would have.
//
// VF_RT_DUMP=<dir> writes the scenes out as Wavefront OBJ + MTL to check the capture in a 3D
// program: every VF_RT_DUMP_EVERY'th picture (default 60), VF_RT_DUMP_COUNT of them (default 10).
// Coordinates are turned into OBJ's right-handed y-up convention (z negated), the camera at the
// origin looking down -z; each viewport is its own group, and culled polygons a group of their own.
#include "model1.h"
#include "../raytrace/rt_render.h"
#include "../raytrace/rt_gpu.h"
#include <glm/geometric.hpp>
#include <cstdio>
#include <map>
#include <string>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#endif

void model1::rt_begin()
{
	m_rt_scene.clear();
	m_rt_scene.frame = m_frame;
	m_rt_scene.moment = m_draw_time;
	rt_view_set();   // (the polygons before the list's first viewport: the view as it stands)
}

// The light, and the normals, as the points are: transform_point turns x and z about y by the
// view's ayy after the object's matrix; transform_vector (the normals, which the light is lit against) doesn't
namespace {
void turn_y(float &x, float &z, float c, float s) { const float t = c * x - s * z; z = s * x + c * z; x = t; }
}

void model1::rt_view_set()
{
	auto set_light = [&](rt::view &v) {
		v.light[0] = m_view->light.x; v.light[1] = m_view->light.y; v.light[2] = m_view->light.z;
		turn_y(v.light[0], v.light[2], m_view->ayyc, m_view->ayys);
	};
	auto &views = m_rt_scene.views;
	if (!views.empty() && views.back().count == 0) views.pop_back();   // (a viewport with nothing drawn in it)
	rt::view v;
	const view_t &w = *m_view;
	v.xc = float(w.xc); v.yc = float(w.yc);
	v.zoomx = w.zoomx; v.zoomy = w.zoomy; v.viewx = w.viewx; v.viewy = w.viewy;
	v.ext = w.ext; v.scale = w.scale;
	v.x1 = w.x1; v.y1 = w.y1; v.x2 = w.x2; v.y2 = w.y2;
	set_light(v);
	v.spec_enable = w.spec_enable;
	v.first = uint32_t(m_rt_scene.quads.size());
	views.push_back(v);
}

void model1::rt_add(const point_t *a, const point_t *b, const point_t *c, const point_t *d, glm::vec3 n,
	uint32_t albedo, uint32_t lit, uint32_t shade, uint32_t obj, int lightmode, uint8_t flags, float sortz)
{
	if (m_rt_scene.views.empty()) rt_view_set();
	rt::view &v = m_rt_scene.views.back();
	// (the light and the specular switch as they are when its polygons are drawn: the list may set them after the viewport)
	v.light[0] = m_view->light.x; v.light[1] = m_view->light.y; v.light[2] = m_view->light.z;
	turn_y(v.light[0], v.light[2], m_view->ayyc, m_view->ayys);
	turn_y(n.x, n.z, m_view->ayyc, m_view->ayys);
	v.spec_enable = m_view->spec_enable;
	rt::quad q;
	const point_t *p[4] = { a, b, c, d };
	for (int i = 0; i < 4; i++) { q.v[i][0] = p[i]->x; q.v[i][1] = p[i]->y; q.v[i][2] = p[i]->z; }
	q.n[0] = n.x; q.n[1] = n.y; q.n[2] = n.z;
	q.albedo = albedo & 0xffffff;
	q.lit = lit & 0xffffff;
	q.shade = shade & 0xffffff;
	q.obj = obj;
	q.sortz = sortz;
	q.view = uint16_t(m_rt_scene.views.size() - 1);
	q.lightmode = uint8_t(lightmode);
	q.flags = flags;
	m_rt_scene.quads.push_back(q);
	v.count++;
}

uint32_t model1::rt_palette_colour(uint16_t tex_data) const
{
	const int color = m_paletteram16[0x1000 | (tex_data & 0x3ff)];
	const int r = color & 0x1f, g = (color >> 5) & 0x1f, b = (color >> 10) & 0x1f;
	// (fully lit through the colour table, as push_object's colour_of at its brightest)
	const int lum = 0x3f;
	const int rr = (m_color_xlat[(r << 8) | lum | 0x0] >> 3) & 0x1f;
	const int gg = (m_color_xlat[(g << 8) | lum | 0x2000] >> 3) & 0x1f;
	const int bb = (m_color_xlat[(b << 8) | lum | 0x4000] >> 3) & 0x1f;
	return uint32_t(pal5bit(rr) << 16) | (pal5bit(gg) << 8) | pal5bit(bb);
}

void model1::rt_end(bitmap_rgb32 &bitmap, const rectangle &cliprect)
{
	auto &views = m_rt_scene.views;
	if (!views.empty() && views.back().count == 0) views.pop_back();
	m_rt_scene.lightparams.resize(256);
	for (int i = 0; i < 256; i++) {
		const lightparam_t &l = m_view->lightparams[i];
		m_rt_scene.lightparams[i] = { l.a, l.d, l.s, l.p };
	}

	if (rt_on()) {
		rt::target t;
		t.px = &bitmap.pix(0);
		t.w = bitmap.width(); t.h = bitmap.height(); t.stride = size_t(bitmap.width());
		t.cx0 = cliprect.min_x; t.cx1 = cliprect.max_x; t.cy0 = cliprect.min_y; t.cy1 = cliprect.max_y;
		rt::options o;
		o.shadows = m_rt_shadows;
		o.gi = m_rt_gi;
		o.fog = m_rt_fog;
		o.spot = m_rt_spot;
		if (const char *e = getenv("VF_RT_SPOT"))   // (tuning: inner,outer,dim,edge,height,head)
			sscanf(e, "%f,%f,%f,%f,%f,%f", &o.spot_inner, &o.spot_outer, &o.spot_dim, &o.spot_edge, &o.spot_height, &o.spot_head);
		o.game_shadows = !o.shadows;
		o.blend_shadows = s_moire_blend;
		o.key = KEY; o.shadow_key = SHADOW_KEY; o.mark = 0x01000000;   // (model1_video.cpp's MOIRE_MARK)
		o.gi_view = m_rt_view;
		if (m_rt_gpu && !m_rt_gpu_failed) {   // (the GPU's quality: a grid point a game pixel, twice the rays)
			o.gi_step = std::max(1, m_scale);
			o.gi_rays = 12;
		}
		const uint64_t t0 = trace_ns();
		rt::result r;
		bool drawn = false;
		if (m_rt_gpu && !m_rt_gpu_failed) {
			drawn = m_rt_gpu->render(m_rt_scene, t, o, r);
			if (!drawn) { m_rt_gpu_failed = true; fprintf(stderr, "rt: the GPU couldn't draw a picture: on the CPU from now on\n"); }
		}
		if (!drawn) r = rt::render(m_rt_scene, t, o);
		m_rt_ms = (trace_ns() - t0) / 1e6;
		m_raster_ms += m_rt_ms;
		if (r.mark_y1 <= r.mark_y2) { s_mark_y1 = std::min(s_mark_y1, r.mark_y1); s_mark_y2 = std::max(s_mark_y2, r.mark_y2); }

		// rt_record: the scene as drawn, to replay it away from the game (rttest --scene)
		if (m_rt_rec_left > 0) {
			m_rt_rec_left--;
			char name[32];
			snprintf(name, sizeof name, "/scene_%04d.rts", m_rt_rec_n++);
			const std::string path = m_rt_rec_dir + name;
			if (FILE *f = fopen(path.c_str(), "wb")) {
				rt::scene_file_header h;
				h.quads = uint32_t(m_rt_scene.quads.size()); h.views = uint32_t(m_rt_scene.views.size());
				h.lightparams = uint32_t(m_rt_scene.lightparams.size());
				h.w = t.w; h.h = t.h; h.cx0 = t.cx0; h.cy0 = t.cy0; h.cx1 = t.cx1; h.cy1 = t.cy1;
				h.frame = m_rt_scene.frame;
				fwrite(&h, sizeof h, 1, f);
				fwrite(m_rt_scene.quads.data(), sizeof(rt::quad), h.quads, f);
				fwrite(m_rt_scene.views.data(), sizeof(rt::view), h.views, f);
				fwrite(m_rt_scene.lightparams.data(), sizeof(rt::light_param), h.lightparams, f);
				fclose(f);
			} else {
				fprintf(stderr, "rt: can't write %s\n", path.c_str());
				m_rt_rec_left = 0;
			}
			if (!m_rt_rec_left) fprintf(stderr, "rt: recorded %d scenes to %s\n", m_rt_rec_n, m_rt_rec_dir.c_str());
		}
	}

	static const char *dir = getenv("VF_RT_DUMP");
	if (!dir || !*dir || m_rt_scene.quads.empty()) return;
	static const int every = getenv("VF_RT_DUMP_EVERY") ? std::max(1, atoi(getenv("VF_RT_DUMP_EVERY"))) : 60;
	static const int count = getenv("VF_RT_DUMP_COUNT") ? atoi(getenv("VF_RT_DUMP_COUNT")) : 10;
	static int pictures = 0, written = 0;
	if (written >= count || pictures++ % every) return;
	rt_dump(dir);
	written++;
}

void model1::rt_dump(const char *dir) const
{
#ifdef _WIN32
	_mkdir(dir);
#else
	mkdir(dir, 0755);
#endif
	char base[64];
	snprintf(base, sizeof base, "rt_%08llu", (unsigned long long)m_rt_scene.frame);
	const std::string obj_path = std::string(dir) + "/" + base + ".obj", mtl_path = std::string(dir) + "/" + base + ".mtl";
	FILE *o = fopen(obj_path.c_str(), "w"), *m = fopen(mtl_path.c_str(), "w");
	if (!o || !m) { if (o) fclose(o); if (m) fclose(m); fprintf(stderr, "rt: can't write %s\n", obj_path.c_str()); return; }

	fprintf(o, "# vf1-recomp ray-traced lighting capture: frame %llu, moment %.3f, %zu quads, %zu views\n",
		(unsigned long long)m_rt_scene.frame, m_rt_scene.moment, m_rt_scene.quads.size(), m_rt_scene.views.size());
	fprintf(o, "# view space turned to OBJ's: x right, y up, z toward the camera (the game's z negated)\nmtllib %s.mtl\n", base);
	for (size_t i = 0; i < m_rt_scene.views.size(); i++) {
		const rt::view &v = m_rt_scene.views[i];
		fprintf(o, "# view %zu: centre (%g, %g) zoom (%g, %g) clip %d,%d..%d,%d light (%g, %g, %g) -> OBJ (%g, %g, %g)\n",
			i, v.xc, v.yc, v.zoomx, v.zoomy, v.x1, v.y1, v.x2, v.y2, v.light[0], v.light[1], v.light[2], v.light[0], v.light[1], -v.light[2]);
	}

	std::map<uint32_t, int> mats;   // albedo -> its material
	auto mat_of = [&](uint32_t c) {
		auto [it, added] = mats.emplace(c, int(mats.size()));
		if (added)
			fprintf(m, "newmtl c%06x\nKd %.4f %.4f %.4f\n\n", c, ((c >> 16) & 0xff) / 255.f, ((c >> 8) & 0xff) / 255.f, (c & 0xff) / 255.f);
		return c;
	};

	size_t vn = 0;
	for (int pass = 0; pass < 2; pass++)   // (the drawn first, then the culled)
		for (size_t vi = 0; vi < m_rt_scene.views.size(); vi++) {
			const rt::view &v = m_rt_scene.views[vi];
			bool grouped = false;
			for (uint32_t k = v.first; k < v.first + v.count; k++) {
				const rt::quad &q = m_rt_scene.quads[k];
				if (bool(q.flags & rt::CULLED) != bool(pass)) continue;
				if (!grouped) { fprintf(o, "g view%zu%s\n", vi, pass ? "_culled" : ""); grouped = true; }
				fprintf(o, "usemtl c%06x\n", mat_of(q.albedo));
				for (int i = 0; i < 4; i++) fprintf(o, "v %g %g %g\n", q.v[i][0], q.v[i][1], -q.v[i][2]);
				fprintf(o, "vn %g %g %g\n", q.n[0], q.n[1], -q.n[2]);
				vn++;
				// (z negated flips the winding: the faces listed the other way round to face as they did)
				fprintf(o, "f -1//%zu -2//%zu -3//%zu -4//%zu\n", vn, vn, vn, vn);
			}
		}
	fclose(o);
	fclose(m);
	fprintf(stderr, "rt: wrote %s (%zu quads)\n", obj_path.c_str(), m_rt_scene.quads.size());
}
