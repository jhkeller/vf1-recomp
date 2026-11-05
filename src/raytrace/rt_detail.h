// Ray-traced lighting's pieces shared by the CPU renderer (rt_render.cpp) and the GPU's (rt_gpu.cpp):
// small vectors, the triangles projected for the raster, the BVH, and a view made ready to draw
// (prepare: the polygons in painted order, projected; their planes; the BVH; the light's side).
#pragma once
#include "rt_scene.h"
#include "rt_render.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace rt {
namespace detail {

struct v3 { float x, y, z; };
inline v3 operator-(v3 a, v3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
inline v3 operator+(v3 a, v3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
inline v3 operator*(v3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
inline float dot(v3 a, v3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline v3 cross(v3 a, v3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
inline v3 vmin(v3 a, v3 b) { return { std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z) }; }
inline v3 vmax(v3 a, v3 b) { return { std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z) }; }
inline float comp(v3 a, int i) { return i == 0 ? a.x : i == 1 ? a.y : a.z; }
inline v3 corner(const quad &q, int i) { return { q.v[i][0], q.v[i][1], q.v[i][2] }; }

constexpr float ZNEAR = 1e-3f;   // (the game draws what's in front of z = 0; a hair in front, for 1/z)
constexpr float DEPTH_EPS = 2e-5f;   // relative: a polygon this near another's depth there counts as level with it
constexpr float LAYER_EPS = 2e-3f;   // relative: two polygons in one plane (a floor and its markings) this near: painted order decides

// ---- the depth-buffered raster ----

struct stri
{
	float x[3], y[3], w[3];   // render pixels, and 1/z
	float area;               // twice its signed area
	int minx, maxx, miny, maxy;
	int32_t quad;
};

// A view-space triangle clipped to z >= ZNEAR, projected, and added (as a fan) to `out`.
inline void add_projected(const view &v, const v3 (&tri)[3], int32_t qi, const int clip[4], std::vector<stri> &out)
{
	v3 poly[4];
	int n = 0;
	for (int i = 0; i < 3; i++) {
		const v3 a = tri[i], b = tri[(i + 1) % 3];
		const bool ain = a.z >= ZNEAR, bin = b.z >= ZNEAR;
		if (ain) poly[n++] = a;
		if (ain != bin) {
			const float t = (ZNEAR - a.z) / (b.z - a.z);
			poly[n++] = { a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, ZNEAR };
		}
	}
	if (n < 3) return;
	float sx[4], sy[4], sw[4];
	for (int i = 0; i < n; i++) {
		const float w = 1.0f / poly[i].z;
		sx[i] = (v.xc + (poly[i].x * w * v.zoomx + v.viewx) + v.ext) * v.scale;
		sy[i] = (v.yc - (poly[i].y * w * v.zoomy + v.viewy)) * v.scale;
		sw[i] = w;
	}
	for (int k = 1; k + 1 < n; k++) {
		stri t;
		const int id[3] = { 0, k, k + 1 };
		float fx0 = 1e30f, fx1 = -1e30f, fy0 = 1e30f, fy1 = -1e30f;
		for (int j = 0; j < 3; j++) {
			t.x[j] = sx[id[j]]; t.y[j] = sy[id[j]]; t.w[j] = sw[id[j]];
			fx0 = std::min(fx0, t.x[j]); fx1 = std::max(fx1, t.x[j]);
			fy0 = std::min(fy0, t.y[j]); fy1 = std::max(fy1, t.y[j]);
		}
		t.area = (t.x[1] - t.x[0]) * (t.y[2] - t.y[0]) - (t.x[2] - t.x[0]) * (t.y[1] - t.y[0]);
		if (!(std::fabs(t.area) > 1e-12f)) continue;
		// the pixels whose centres (x + 0.5) it may cover, inside the clip
		t.minx = std::max(clip[0], int(std::floor(fx0 - 0.5f)));
		t.maxx = std::min(clip[2], int(std::ceil(fx1 - 0.5f)));
		t.miny = std::max(clip[1], int(std::floor(fy0 - 0.5f)));
		t.maxy = std::min(clip[3], int(std::ceil(fy1 - 0.5f)));
		if (t.minx > t.maxx || t.miny > t.maxy) continue;
		t.quad = qi;
		out.push_back(t);
	}
}

// Calls f(x, y, w) for each pixel of `t` in rows y0..y1 (its centre inside it, edges included).
template <class F>
inline void scan(const stri &t, int y0, int y1, F &&f)
{
	const float inv = 1.0f / t.area;
	// barycentric weights as planes in (px, py): l_i = a_i * px + b_i * py + c_i
	float a[3], b[3], c[3];
	for (int i = 0; i < 3; i++) {
		const int j = (i + 1) % 3, k = (i + 2) % 3;
		// the edge j -> k, the weight of vertex i
		a[i] = (t.y[j] - t.y[k]) * inv;
		b[i] = (t.x[k] - t.x[j]) * inv;
		c[i] = (t.x[j] * t.y[k] - t.x[k] * t.y[j]) * inv;
	}
	const float aw = a[0] * t.w[0] + a[1] * t.w[1] + a[2] * t.w[2];
	const float bw = b[0] * t.w[0] + b[1] * t.w[1] + b[2] * t.w[2];
	const float cw = c[0] * t.w[0] + c[1] * t.w[1] + c[2] * t.w[2];
	const float tol = -1e-6f;
	for (int y = std::max(y0, t.miny); y <= std::min(y1, t.maxy); y++) {
		const float py = y + 0.5f;
		// the row's span: where every weight is >= 0
		float xl = float(t.minx) + 0.5f, xr = float(t.maxx) + 0.5f;
		bool none = false;
		for (int i = 0; i < 3 && !none; i++) {
			const float r = b[i] * py + c[i];   // l_i = a_i * px + r >= 0
			if (a[i] > 0) xl = std::max(xl, -r / a[i]);
			else if (a[i] < 0) xr = std::min(xr, -r / a[i]);
			else if (r < tol) none = true;
		}
		if (none || xl > xr) continue;
		int x0 = std::max(t.minx, int(std::ceil(xl - 0.5f)));
		int x1 = std::min(t.maxx, int(std::floor(xr - 0.5f)));
		// (the ends by the weights themselves: the division above may round a centre on an edge either way)
		auto inside = [&](int x) {
			const float px = x + 0.5f;
			for (int i = 0; i < 3; i++) if (a[i] * px + b[i] * py + c[i] < tol) return false;
			return true;
		};
		if (x0 - 1 >= t.minx && inside(x0 - 1)) x0--;
		if (x1 + 1 <= t.maxx && inside(x1 + 1)) x1++;
		for (int x = x0; x <= x1; x++) {
			const float px = x + 0.5f;
			f(x, y, aw * px + bw * py + cw);
		}
	}
}

// ---- the BVH, for the shadows' rays ----

struct btri { v3 p0, e1, e2; int32_t quad; v3 n; };   // n: its unit normal
struct bnode { v3 lo, hi; int32_t left = -1, first = 0, count = 0; };   // left < 0: a leaf of [first, first + count)

struct bvh
{
	std::vector<btri> tris;
	std::vector<bnode> nodes;

	void build()
	{
		nodes.clear();
		if (tris.empty()) return;
		nodes.reserve(tris.size() * 2);
		std::vector<v3> cen(tris.size());
		for (size_t i = 0; i < tris.size(); i++) {
			const btri &t = tris[i];
			cen[i] = t.p0 + (t.e1 + t.e2) * (1.0f / 3.0f);
		}
		std::vector<int32_t> idx(tris.size());
		std::iota(idx.begin(), idx.end(), 0);
		nodes.push_back({});
		split(0, 0, int32_t(idx.size()), idx, cen);
		std::vector<btri> sorted(tris.size());
		for (size_t i = 0; i < idx.size(); i++) sorted[i] = tris[idx[i]];
		tris.swap(sorted);
	}

	void split(int32_t ni, int32_t first, int32_t count, std::vector<int32_t> &idx, const std::vector<v3> &cen)
	{
		v3 lo{ 1e30f, 1e30f, 1e30f }, hi{ -1e30f, -1e30f, -1e30f }, clo = lo, chi = hi;
		for (int32_t i = first; i < first + count; i++) {
			const btri &t = tris[idx[i]];
			const v3 a = t.p0, b = t.p0 + t.e1, c = t.p0 + t.e2;
			lo = vmin(lo, vmin(a, vmin(b, c)));
			hi = vmax(hi, vmax(a, vmax(b, c)));
			clo = vmin(clo, cen[idx[i]]);
			chi = vmax(chi, cen[idx[i]]);
		}
		nodes[ni].lo = lo;
		nodes[ni].hi = hi;
		const v3 ext = chi - clo;
		const int axis = ext.x >= ext.y && ext.x >= ext.z ? 0 : ext.y >= ext.z ? 1 : 2;
		if (count <= 4) { nodes[ni].first = first; nodes[ni].count = count; return; }   // (at most 4 a leaf: the GPU's nodes keep it in 3 bits)
		const int32_t mid = first + count / 2;
		if (comp(ext, axis) > 0)   // (else their centres all in one place: split as they stand)
			std::nth_element(idx.begin() + first, idx.begin() + mid, idx.begin() + first + count,
				[&](int32_t p, int32_t q) { return comp(cen[p], axis) < comp(cen[q], axis); });
		const int32_t l = int32_t(nodes.size());
		nodes.push_back({});
		nodes.push_back({});
		nodes[ni].left = l;
		split(l, first, mid - first, idx, cen);
		split(l + 1, mid, first + count - mid, idx, cen);
	}

	// The nearest of what lies along o + t d, t > tmin, but `skip`: false if nothing.
	// What lies in the plane of `on` (the point the ray leaves, within `tol`) is passed over too:
	// a floor's markings, or the tile beside, can't shadow the floor they lie on.
	bool closest(v3 o, v3 d, float tmin, int32_t skip, v3 on, float tol, float &t_hit, int32_t &quad_hit) const
	{
		if (nodes.empty()) return false;
		const v3 inv{ 1.0f / d.x, 1.0f / d.y, 1.0f / d.z };
		int32_t stack[64];
		int sp = 0;
		stack[sp++] = 0;
		float best = 1e30f;
		int32_t found = -1;
		while (sp) {
			const bnode &n = nodes[stack[--sp]];
			float t0 = tmin, t1 = best;
			for (int a = 0; a < 3; a++) {
				const float ia = comp(inv, a), oa = comp(o, a);
				float ta = (comp(n.lo, a) - oa) * ia, tb = (comp(n.hi, a) - oa) * ia;
				if (ta > tb) std::swap(ta, tb);
				t0 = std::max(t0, ta);
				t1 = std::min(t1, tb);
			}
			if (!(t0 <= t1)) continue;
			if (n.left < 0) {
				for (int32_t i = n.first; i < n.first + n.count; i++) {
					const btri &t = tris[i];
					if (t.quad == skip) continue;
					if (std::fabs(dot(t.n, on - t.p0)) < tol) continue;
					const v3 p = cross(d, t.e2);
					const float det = dot(t.e1, p);
					if (std::fabs(det) < 1e-12f) continue;
					const float id = 1.0f / det;
					const v3 s = o - t.p0;
					const float u = dot(s, p) * id;
					if (u < 0 || u > 1) continue;
					const v3 q = cross(s, t.e1);
					const float v = dot(d, q) * id;
					if (v < 0 || u + v > 1) continue;
					const float th = dot(t.e2, q) * id;
					if (th > tmin && th < best) { best = th; found = t.quad; }
				}
			} else if (sp + 2 <= 64) {
				stack[sp++] = n.left;
				stack[sp++] = n.left + 1;
			}
		}
		if (found < 0) return false;
		t_hit = best;
		quad_hit = found;
		return true;
	}

	// Whether anything but `skip` lies along o + t d, t > tmin.
	bool any_hit(v3 o, v3 d, float tmin, int32_t skip, v3 on, float tol) const
	{
		if (nodes.empty()) return false;
		const v3 inv{ 1.0f / d.x, 1.0f / d.y, 1.0f / d.z };
		int32_t stack[64];
		int sp = 0;
		stack[sp++] = 0;
		while (sp) {
			const bnode &n = nodes[stack[--sp]];
			// the slabs (an infinite ray: no far end)
			float t0 = tmin, t1 = 1e30f;
			for (int a = 0; a < 3; a++) {
				const float ia = comp(inv, a), oa = comp(o, a);
				float ta = (comp(n.lo, a) - oa) * ia, tb = (comp(n.hi, a) - oa) * ia;
				if (ta > tb) std::swap(ta, tb);
				t0 = std::max(t0, ta);
				t1 = std::min(t1, tb);
			}
			if (!(t0 <= t1)) continue;
			if (n.left < 0) {
				for (int32_t i = n.first; i < n.first + n.count; i++) {
					const btri &t = tris[i];
					if (t.quad == skip) continue;
					if (std::fabs(dot(t.n, on - t.p0)) < tol) continue;
					// Moller-Trumbore, both faces
					const v3 p = cross(d, t.e2);
					const float det = dot(t.e1, p);
					if (std::fabs(det) < 1e-12f) continue;
					const float id = 1.0f / det;
					const v3 s = o - t.p0;
					const float u = dot(s, p) * id;
					if (u < 0 || u > 1) continue;
					const v3 q = cross(s, t.e1);
					const float v = dot(d, q) * id;
					if (v < 0 || u + v > 1) continue;
					if (dot(t.e2, q) * id > tmin) return true;
				}
			} else if (sp + 2 <= 64) {
				stack[sp++] = n.left;
				stack[sp++] = n.left + 1;
			}
		}
		return false;
	}
};


// Two polygons in one plane (a floor and its markings) at a pixel of depth 1/w: the raster lets the
// one painted later win there, whatever their rounded depths say.
inline bool one_plane(const std::vector<float> &planes, int32_t p, int32_t q, float w)
{
	const float *a = &planes[size_t(p) * 4], *c = &planes[size_t(q) * 4];
	const float d = a[0] * c[0] + a[1] * c[1] + a[2] * c[2];
	if (std::fabs(d) < 0.999f) return false;
	const float z = 1.0f / w;
	return std::fabs(a[3] - (d > 0 ? c[3] : -c[3])) < LAYER_EPS * z;
}

// A view made ready to draw: what both renderers need before any pixel.
struct prepared
{
	int clip[4] = { 0, 0, -1, -1 };      // its pixels (inclusive), inside the target
	std::vector<int32_t> order;          // its drawn polygons, in the order the game paints them
	std::vector<stri> tris, shadow_tris; // projected, in that order: solid ones, the game's shadow polygons
	bvh accel;                           // the rays' (empty unless shadows or global illumination)
	v3 L{ 0, 1, 0 };                     // toward the light (unit)
	float ll = 0;                        // the light vector's length (0: no light)
	float zref = 1;                      // the view's typical depth (its drawn polygons' median)
	v3 up{ 0, 1, 0 };                    // the floor's normal, on the light's side (unit)
	float floor = 0;                     // the floor's height along it
};
// false: nothing of this view to draw. planes: a polygon's plane (unit normal, offset), 4 floats a
// polygon, by its place in s.quads (this view's filled); light_sign: the view's kept light side.
bool prepare(const scene &s, size_t vi, const target &t, const options &o, std::vector<float> &planes, int &light_sign, prepared &p);

} // namespace detail
} // namespace rt
