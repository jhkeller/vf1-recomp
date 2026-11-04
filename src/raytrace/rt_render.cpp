// Ray-traced lighting's renderer: see rt_render.h.
#include "rt_render.h"
#include "../machine/parallel.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <numeric>
#include <vector>

namespace rt {
namespace {

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

// ---- the depth-buffered raster ----

struct stri
{
	float x[3], y[3], w[3];   // render pixels, and 1/z
	float area;               // twice its signed area
	int minx, maxx, miny, maxy;
	int32_t quad;
};

// A view-space triangle clipped to z >= ZNEAR, projected, and added (as a fan) to `out`.
void add_projected(const view &v, const v3 (&tri)[3], int32_t qi, const int clip[4], std::vector<stri> &out)
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

struct btri { v3 p0, e1, e2; int32_t quad; };
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
		if (count <= 4 || comp(ext, axis) <= 0) { nodes[ni].first = first; nodes[ni].count = count; return; }
		const int32_t mid = first + count / 2;
		std::nth_element(idx.begin() + first, idx.begin() + mid, idx.begin() + first + count,
			[&](int32_t p, int32_t q) { return comp(cen[p], axis) < comp(cen[q], axis); });
		const int32_t l = int32_t(nodes.size());
		nodes.push_back({});
		nodes.push_back({});
		nodes[ni].left = l;
		split(l, first, mid - first, idx, cen);
		split(l + 1, mid, first + count - mid, idx, cen);
	}

	// Whether anything but `skip` lies along o + t d, t > tmin.
	bool any_hit(v3 o, v3 d, float tmin, int32_t skip) const
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

// ---- the shadow polygons' rule (model1::draw_hline_moired) ----

inline void shade_pixel(uint32_t &p, uint32_t colour, int x, int y, const options &o)
{
	if (o.blend_shadows) {
		if (p == o.key) p = o.shadow_key | ((colour >> 1) & 0x7f7f7f);
		else if (!(p & o.mark)) p = o.mark | (((p >> 1) & 0x7f7f7f) + ((colour >> 1) & 0x7f7f7f));
	} else if (!((x ^ y) & 1))
		p = colour & 0xffffff;
}

struct buffers
{
	std::vector<float> depth;
	std::vector<int32_t> id;
	std::vector<stri> tris, shadow_tris;
	bvh accel;
	std::vector<int32_t> order;
	std::mutex mu;
};

} // namespace

result render(const scene &s, const target &t, const options &o)
{
	static buffers b;
	std::lock_guard<std::mutex> lk(b.mu);
	result res;
	const size_t n = size_t(t.w) * t.h;
	if (b.depth.size() < n) { b.depth.resize(n); b.id.resize(n); }

	for (size_t vi = 0; vi < s.views.size(); vi++) {
		const view &v = s.views[vi];
		const int clip[4] = { std::max({ v.x1, t.cx0, 0 }), std::max({ v.y1, t.cy0, 0 }),
		                      std::min({ v.x2, t.cx1, t.w - 1 }), std::min({ v.y2, t.cy1, t.h - 1 }) };
		if (clip[0] > clip[2] || clip[1] > clip[3] || !v.count) continue;

		// the polygons in the order the game paints them: far to near by its depth, ties in list order
		b.order.clear();
		for (uint32_t k = v.first; k < v.first + v.count; k++) {
			const quad &q = s.quads[k];
			if (q.flags & CULLED) continue;
			b.order.push_back(int32_t(k));
		}
		std::stable_sort(b.order.begin(), b.order.end(), [&](int32_t p, int32_t q) { return s.quads[p].sortz > s.quads[q].sortz; });

		b.tris.clear();
		b.shadow_tris.clear();
		for (int32_t k : b.order) {
			const quad &q = s.quads[k];
			const v3 c[4] = { corner(q, 0), corner(q, 1), corner(q, 2), corner(q, 3) };
			const v3 t0[3] = { c[0], c[1], c[2] }, t1[3] = { c[0], c[2], c[3] };
			std::vector<stri> &out = (q.flags & MOIRE) ? b.shadow_tris : b.tris;
			add_projected(v, t0, k, clip, out);
			add_projected(v, t1, k, clip, out);
		}
		res.tris += b.tris.size();

		// the depths and which polygon each pixel shows, a band of rows a thread
		parallel_rows(clip[3] - clip[1] + 1, [&](int r0, int r1) {
			const int y0 = clip[1] + r0, y1 = clip[1] + r1 - 1;
			for (int y = y0; y <= y1; y++) {
				std::fill(&b.depth[size_t(y) * t.w + clip[0]], &b.depth[size_t(y) * t.w + clip[2]] + 1, 0.0f);
				std::fill(&b.id[size_t(y) * t.w + clip[0]], &b.id[size_t(y) * t.w + clip[2]] + 1, -1);
			}
			for (const stri &tr : b.tris)
				scan(tr, y0, y1, [&](int x, int y, float w) {
					const size_t i = size_t(y) * t.w + x;
					if (w <= 0) return;
					// (as near as what's there, or nearer: painted later, it wins)
					if (w * (1 + DEPTH_EPS) >= b.depth[i]) { b.depth[i] = std::max(w, b.depth[i]); b.id[i] = tr.quad; }
				});
		}, 4);

		// the shadows' BVH: every polygon of the view that can block the light, those facing away too
		if (o.shadows) {
			b.accel.tris.clear();
			for (uint32_t k = v.first; k < v.first + v.count; k++) {
				const quad &q = s.quads[k];
				if (q.flags & (MOIRE | UNLIT)) continue;
				const v3 c[4] = { corner(q, 0), corner(q, 1), corner(q, 2), corner(q, 3) };
				for (int h = 0; h < 2; h++) {
					const v3 a = c[0], bb = c[1 + h], cc = c[2 + h];
					const v3 e1 = bb - a, e2 = cc - a;
					if (dot(cross(e1, e2), cross(e1, e2)) <= 0) continue;
					b.accel.tris.push_back({ a, e1, e2, int32_t(k) });
				}
			}
			b.accel.build();
		}
		// Toward the light. A polygon is lit when its normal (the game's) and the light vector agree;
		// whether the game's normals point out of a model or into it, its polygons facing the camera
		// say (their outward side faces it): their votes turn the light vector into "toward the light"
		// for outward normals, which the rays need.
		float lx = v.light[0], ly = v.light[1], lz = v.light[2];
		if (o.shadows) {
			int votes = 0;
			for (int32_t k : b.order) {
				const quad &q = s.quads[k];
				if (q.flags & (MOIRE | UNLIT | TWO_SIDED)) continue;
				const v3 c0 = corner(q, 0);
				v3 N = cross(corner(q, 1) - c0, corner(q, 2) - c0);
				if (dot(N, c0) > 0) N = N * -1.0f;   // (facing the camera at the origin)
				const float d = dot(N, v3{ q.n[0], q.n[1], q.n[2] });
				votes += d > 0 ? 1 : d < 0 ? -1 : 0;
			}
			if (votes < 0) { lx = -lx; ly = -ly; lz = -lz; }
		}
		const float ll = std::sqrt(lx * lx + ly * ly + lz * lz);
		const v3 L = ll > 0 ? v3{ lx / ll, ly / ll, lz / ll } : v3{ 0, 1, 0 };
		const bool trace = o.shadows && ll > 0 && !b.accel.nodes.empty();

		// the colours
		std::atomic<size_t> rays{ 0 }, shadowed{ 0 };
		parallel_rows(clip[3] - clip[1] + 1, [&](int r0, int r1) {
			size_t my_rays = 0, my_shadowed = 0;
			for (int y = clip[1] + r0; y < clip[1] + r1; y++) {
				uint32_t *row = t.px + size_t(y) * t.stride;
				for (int x = clip[0]; x <= clip[2]; x++) {
					const size_t i = size_t(y) * t.w + x;
					const int32_t qi = b.id[i];
					if (qi < 0) continue;
					const quad &q = s.quads[qi];
					uint32_t c = q.lit;
					if (trace && q.shade != q.lit && !(q.flags & UNLIT)) {
						// the pixel's place in view space, from its depth
						const float z = 1.0f / b.depth[i];
						const float px = ((x + 0.5f) / v.scale - v.ext - v.xc - v.viewx) / v.zoomx * z;
						const float py = (v.yc - (y + 0.5f) / v.scale - v.viewy) / v.zoomy * z;
						const v3 P{ px, py, z };
						// off the surface toward the camera, by about a pixel's width there
						const v3 c0 = corner(q, 0);
						v3 N = cross(corner(q, 1) - c0, corner(q, 2) - c0);
						if (dot(N, N) <= 0) N = cross(corner(q, 2) - c0, corner(q, 3) - c0);
						const float nl = std::sqrt(dot(N, N));
						if (nl > 0) {
							N = N * (1.0f / nl);
							if (dot(N, P) > 0) N = N * -1.0f;
						}
						const float eps = 1.5f * z / (std::fabs(v.zoomx) * v.scale + 1e-6f);
						const v3 O = P + N * eps + L * eps;
						my_rays++;
						if (b.accel.any_hit(O, L, 0.0f, qi)) { c = q.shade; my_shadowed++; }
					}
					row[x] = c;
				}
			}
			rays += my_rays;
			shadowed += my_shadowed;
		}, 4);
		res.rays += rays;
		res.shadowed += shadowed;

		// the game's shadow polygons: over what's there, behind nothing nearer
		if (o.game_shadows && !b.shadow_tris.empty()) {
			std::mutex mm;
			parallel_rows(clip[3] - clip[1] + 1, [&](int r0, int r1) {
				const int y0 = clip[1] + r0, y1 = clip[1] + r1 - 1;
				int m0 = 1 << 30, m1 = -1;
				for (const stri &tr : b.shadow_tris) {
					const uint32_t colour = s.quads[tr.quad].lit;
					scan(tr, y0, y1, [&](int x, int y, float w) {
						const size_t i = size_t(y) * t.w + x;
						if (w <= 0 || w * (1 + DEPTH_EPS) < b.depth[i]) return;
						shade_pixel(t.px[size_t(y) * t.stride + x], colour, x, y, o);
						m0 = std::min(m0, y); m1 = std::max(m1, y);
					});
				}
				if (o.blend_shadows && m0 <= m1) {
					std::lock_guard<std::mutex> l(mm);
					res.mark_y1 = std::min(res.mark_y1, m0);
					res.mark_y2 = std::max(res.mark_y2, m1);
				}
			}, 4);
		}
	}
	return res;
}

} // namespace rt
