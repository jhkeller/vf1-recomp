// Ray-traced lighting's renderer: see rt_render.h.
#include "rt_render.h"
#include "rt_detail.h"
#include "../machine/parallel.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <numeric>
#include <vector>

namespace rt {
using namespace detail;
namespace {

// ---- the shadow polygons' rule (model1::draw_hline_moired) ----

inline void shade_pixel(uint32_t &p, uint32_t colour, int x, int y, const options &o)
{
	if (o.blend_shadows) {
		if (p == o.key) p = o.shadow_key | ((colour >> 1) & 0x7f7f7f);
		else if (!(p & o.mark)) p = o.mark | (((p >> 1) & 0x7f7f7f) + ((colour >> 1) & 0x7f7f7f));
	} else if (!((x ^ y) & 1))
		p = colour & 0xffffff;
}

// ---- global illumination's grid ----

struct gi_grid
{
	int w = 0, h = 0, step = 0, x0 = 0, y0 = 0;
	std::vector<int32_t> id;        // the polygon under each point (-1: none), for keeping its history
	std::vector<float> ao, rgb;     // its ambient light left (1: open), its bounced light (r, g, b)
	std::vector<float> nz;          // its surface there: unit normal and depth (z), for which neighbours' light it shares
};

inline uint32_t hash32(uint32_t x)
{
	x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
	return x;
}
inline float unit(uint32_t &s) { s = hash32(s + 0x9e3779b9U); return float(s >> 8) * (1.0f / 16777216.0f); }
inline v3 rgb_of(uint32_t c) { return { float((c >> 16) & 255) / 255.0f, float((c >> 8) & 255) / 255.0f, float(c & 255) / 255.0f }; }
inline uint32_t pack(v3 c)
{
	auto ch = [](float f) { return uint32_t(std::lround(std::clamp(f, 0.0f, 1.0f) * 255.0f)); };
	return ch(c.x) << 16 | ch(c.y) << 8 | ch(c.z);
}

struct buffers
{
	std::vector<gi_grid> grids;     // a view's, by its place in the list
	uint32_t pictures = 0;
	std::vector<float> depth;
	std::vector<int32_t> id;
	prepared prep;
	std::vector<float> planes;      // each polygon's plane (unit normal, offset), for telling layers in one plane
	std::vector<uint8_t> vis;       // the shadows: 0 not traced, 1 lit, 2 in shadow
	std::vector<int> light_sign;    // a view's light side, by its place in the list (kept: no flipping picture to picture)
	std::mutex mu;
};

} // namespace

namespace detail {

bool prepare(const scene &s, size_t vi, const target &t, const options &o, std::vector<float> &planes, int &light_sign, prepared &p)
{
	if (planes.size() < s.quads.size() * 4) planes.resize(s.quads.size() * 4);
	const view &v = s.views[vi];
	int *clip = p.clip;
	clip[0] = std::max({ v.x1, t.cx0, 0 }); clip[1] = std::max({ v.y1, t.cy0, 0 });
	clip[2] = std::min({ v.x2, t.cx1, t.w - 1 }); clip[3] = std::min({ v.y2, t.cy1, t.h - 1 });
	if (clip[0] > clip[2] || clip[1] > clip[3] || !v.count) return false;

	// the polygons in the order the game paints them: far to near by its depth, ties in list order
	p.order.clear();
	for (uint32_t k = v.first; k < v.first + v.count; k++) {
		const quad &q = s.quads[k];
		if (q.flags & CULLED) continue;
		p.order.push_back(int32_t(k));
	}
	std::stable_sort(p.order.begin(), p.order.end(), [&](int32_t a, int32_t c) { return s.quads[a].sortz > s.quads[c].sortz; });

	// each polygon's plane (for the layers of a floor: see the raster; the culled too, which rays hit)
	for (uint32_t k = v.first; k < v.first + v.count; k++) {
		const quad &q = s.quads[k];
		const v3 c0 = corner(q, 0);
		v3 N = cross(corner(q, 1) - c0, corner(q, 2) - c0);
		if (dot(N, N) <= 0) N = cross(corner(q, 2) - c0, corner(q, 3) - c0);
		const float nl = std::sqrt(dot(N, N));
		float *pl = &planes[size_t(k) * 4];
		if (nl > 0) { N = N * (1.0f / nl); pl[0] = N.x; pl[1] = N.y; pl[2] = N.z; pl[3] = dot(N, c0); }
		else pl[0] = pl[1] = pl[2] = pl[3] = 0;
	}
	p.tris.clear();
	p.shadow_tris.clear();
	for (int32_t k : p.order) {
		const quad &q = s.quads[k];
		const v3 c[4] = { corner(q, 0), corner(q, 1), corner(q, 2), corner(q, 3) };
		const v3 t0[3] = { c[0], c[1], c[2] }, t1[3] = { c[0], c[2], c[3] };
		std::vector<stri> &out = (q.flags & MOIRE) ? p.shadow_tris : p.tris;
		add_projected(v, t0, k, clip, out);
		add_projected(v, t1, k, clip, out);
	}

	// the rays' BVH: every polygon of the view that can block the light or send it on, those facing away too
	const bool need_bvh = o.shadows || o.gi || o.fog;
	if (!p.order.empty()) {
		std::vector<float> zs;
		zs.reserve(p.order.size());
		for (int32_t k : p.order) { const quad &q = s.quads[k]; zs.push_back(0.25f * (q.v[0][2] + q.v[1][2] + q.v[2][2] + q.v[3][2])); }
		std::nth_element(zs.begin(), zs.begin() + zs.size() / 2, zs.end());
		p.zref = std::max(1e-3f, std::fabs(zs[zs.size() / 2]));
	}
	p.accel.tris.clear();
	p.accel.nodes.clear();
	if (need_bvh) {
		for (uint32_t k = v.first; k < v.first + v.count; k++) {
			const quad &q = s.quads[k];
			if (q.flags & (MOIRE | UNLIT)) continue;
			const v3 c[4] = { corner(q, 0), corner(q, 1), corner(q, 2), corner(q, 3) };
			for (int h = 0; h < 2; h++) {
				const v3 a = c[0], bb = c[1 + h], cc = c[2 + h];
				const v3 e1 = bb - a, e2 = cc - a;
				const v3 nn = cross(e1, e2);
				const float nl = dot(nn, nn);
				if (nl <= 0) continue;
				p.accel.tris.push_back({ a, e1, e2, int32_t(k), nn * (1.0f / std::sqrt(nl)) });
			}
		}
		p.accel.build();
	}
	// Toward the light. A polygon is lit when its normal (the game's) and the light vector agree;
	// whether the game's normals point out of a model or into it, its polygons facing the camera
	// say (their outward side faces it): their votes turn the light vector into "toward the light"
	// for outward normals, which the rays need. Kept until the votes are clearly the other way
	// (a near tie flipping it picture to picture would flicker the whole picture).
	float lx = v.light[0], ly = v.light[1], lz = v.light[2];
	if (need_bvh) {
		int votes = 0, voters = 0;
		for (int32_t k : p.order) {
			const quad &q = s.quads[k];
			if (q.flags & (MOIRE | UNLIT | TWO_SIDED)) continue;
			const v3 c0 = corner(q, 0);
			v3 N = cross(corner(q, 1) - c0, corner(q, 2) - c0);
			if (dot(N, c0) > 0) N = N * -1.0f;   // (facing the camera at the origin)
			const float d = dot(N, v3{ q.n[0], q.n[1], q.n[2] });
			votes += d > 0 ? 1 : d < 0 ? -1 : 0;
			voters++;
		}
		int &sign = light_sign;
		if (votes * sign < 0 && std::abs(votes) * 5 > voters) sign = -sign;
		if (sign < 0) { lx = -lx; ly = -ly; lz = -lz; }
	}
	p.ll = std::sqrt(lx * lx + ly * ly + lz * lz);
	p.L = p.ll > 0 ? v3{ lx / p.ll, ly / p.ll, lz / p.ll } : v3{ 0, 1, 0 };
	return true;
}

} // namespace detail

result render(const scene &s, const target &t, const options &o)
{
	static buffers b;
	std::lock_guard<std::mutex> lk(b.mu);
	result res;
	const size_t n = size_t(t.w) * t.h;
	if (b.depth.size() < n) { b.depth.resize(n); b.id.resize(n); b.vis.resize(n); }
	if (b.planes.size() < s.quads.size() * 4) b.planes.resize(s.quads.size() * 4);
	if (b.light_sign.size() < s.views.size()) b.light_sign.resize(s.views.size(), 1);

	for (size_t vi = 0; vi < s.views.size(); vi++) {
		const view &v = s.views[vi];
		prepared &P_ = b.prep;
		if (!prepare(s, vi, t, o, b.planes, b.light_sign[vi], P_)) continue;
		const int *clip = P_.clip;
		res.tris += P_.tris.size();
		auto one_plane = [&](int32_t p, int32_t q, float w) { return detail::one_plane(b.planes, p, q, w); };
		const v3 L = P_.L;
		const float ll = P_.ll;
		const bool trace = o.shadows && ll > 0 && !P_.accel.nodes.empty();
		const bool gi = o.gi && !P_.accel.nodes.empty();
		bvh &accel_ = P_.accel;
		(void)accel_;

		// the depths and which polygon each pixel shows, a band of rows a thread
		parallel_rows(clip[3] - clip[1] + 1, [&](int r0, int r1) {
			const int y0 = clip[1] + r0, y1 = clip[1] + r1 - 1;
			for (int y = y0; y <= y1; y++) {
				std::fill(&b.depth[size_t(y) * t.w + clip[0]], &b.depth[size_t(y) * t.w + clip[2]] + 1, 0.0f);
				std::fill(&b.id[size_t(y) * t.w + clip[0]], &b.id[size_t(y) * t.w + clip[2]] + 1, -1);
			}
			for (const stri &tr : P_.tris)
				scan(tr, y0, y1, [&](int x, int y, float w) {
					const size_t i = size_t(y) * t.w + x;
					if (w <= 0) return;
					// (as near as what's there, or nearer, or in its plane: painted later, it wins)
					const float d = b.depth[i];
					if (w * (1 + DEPTH_EPS) >= d || (w * (1 + 4 * LAYER_EPS) >= d && b.id[i] >= 0 && one_plane(b.id[i], tr.quad, w))) {
						b.depth[i] = std::max(w, d);
						b.id[i] = tr.quad;
					}
				});
		}, 4);

		// a pixel's place in view space, from its depth
		auto place = [&](float sx, float sy, float depth) {
			const float z = 1.0f / depth;
			return v3{ (sx / v.scale - v.ext - v.xc - v.viewx) / v.zoomx * z, (v.yc - sy / v.scale - v.viewy) / v.zoomy * z, z };
		};
		// a polygon's unit normal on the side facing `toward`
		auto facing = [&](int32_t k, v3 toward) {
			const float *pl = &b.planes[size_t(k) * 4];
			const v3 N{ pl[0], pl[1], pl[2] };
			return dot(N, toward) < 0 ? N * -1.0f : N;
		};
		// about a pixel's width at depth z: how far a ray starts off its surface
		auto eps_at = [&](float z) { return 1.5f * std::fabs(z) / (std::fabs(v.zoomx) * v.scale + 1e-6f); };

		// Global illumination, on a grid of points a few render pixels apart: from each, rays over the
		// hemisphere above its surface (cosine-weighted); each ray that hits a polygon brings back that
		// polygon's colour as lit where it hit (the game's ambient, and its diffuse unless that point is
		// in shadow), and a hit closer than `near` takes some of the ambient light away. Half the points
		// (a checkerboard, the other half next picture) are traced again each picture, and what each
		// found is kept and blended with the pictures before while the same surface stays under it.
		if (b.grids.size() < s.views.size()) b.grids.resize(s.views.size());
		gi_grid &G = b.grids[vi];
		if (gi) {
			const int g = o.gi_step > 0 ? o.gi_step : std::max(2, int(v.scale * 2 + 0.5f));
			const int gw = (clip[2] - clip[0]) / g + 1, gh = (clip[3] - clip[1]) / g + 1;
			if (G.w != gw || G.h != gh || G.step != g || G.x0 != clip[0] || G.y0 != clip[1]) {
				G.w = gw; G.h = gh; G.step = g; G.x0 = clip[0]; G.y0 = clip[1];
				G.id.assign(size_t(gw) * gh, -1);
				G.ao.assign(size_t(gw) * gh, 1.0f);
				G.rgb.assign(size_t(gw) * gh * 3, 0.0f);
				G.nz.assign(size_t(gw) * gh * 4, 0.0f);
			}
			const bnode &root = P_.accel.nodes[0];
			const v3 ext = root.hi - root.lo;
			const float diag = std::sqrt(dot(ext, ext));
			const int nrays = std::max(1, o.gi_rays);
			const uint32_t pic = ++b.pictures;
			std::atomic<size_t> gr{ 0 };
			parallel_rows(gh, [&](int r0, int r1) {
				size_t my = 0;
				for (int gy = r0; gy < r1; gy++)
					for (int gx = 0; gx < gw; gx++) {
						const size_t gi_ = size_t(gy) * gw + gx;
						const int x = std::min(clip[2], clip[0] + gx * g + g / 2), y = std::min(clip[3], clip[1] + gy * g + g / 2);
						const size_t i = size_t(y) * t.w + x;
						const int32_t qi = b.id[i];
						if (qi < 0 || (s.quads[qi].flags & UNLIT)) { G.id[gi_] = -1; continue; }
						const v3 P = place(x + 0.5f, y + 0.5f, b.depth[i]);
						const v3 N = facing(qi, P * -1.0f);
						if (dot(N, N) <= 0) { G.id[gi_] = -1; continue; }
						// what was found here before still holds? The same polygon, or a surface much like it
						float *nz = &G.nz[gi_ * 4];
						const bool same = G.id[gi_] == qi;
						const bool like = G.id[gi_] >= 0 && nz[0] * N.x + nz[1] * N.y + nz[2] * N.z > 0.95f && std::fabs(nz[3] - P.z) < 0.03f * P.z;
						nz[0] = N.x; nz[1] = N.y; nz[2] = N.z; nz[3] = P.z;
						if ((same || like) && ((gx + gy + pic) & 1)) { G.id[gi_] = qi; continue; }   // (its turn next picture)
						const v3 O = P + N * eps_at(P.z);
						const float tol = eps_at(P.z);
						const float near = std::min(0.15f * diag, o.ao_radius * P.z);
						// a frame around N
						const v3 T = std::fabs(N.x) < 0.9f ? cross(N, v3{ 1, 0, 0 }) : cross(N, v3{ 0, 1, 0 });
						const v3 Tn = T * (1.0f / std::sqrt(dot(T, T)));
						const v3 B = cross(N, Tn);
						uint32_t seed = hash32(uint32_t(gx) * 73856093u ^ uint32_t(gy) * 19349663u ^ pic * 83492791u);
						float occluded = 0;
						v3 bounce{ 0, 0, 0 };
						for (int r = 0; r < nrays; r++) {
							// stratified over the rays: each takes its own slice of the turn
							const float u1 = (r + unit(seed)) / nrays, u2 = unit(seed);
							const float rad = std::sqrt(u1), phi = 6.2831853f * u2;
							const v3 d = Tn * (rad * std::cos(phi)) + B * (rad * std::sin(phi)) + N * std::sqrt(std::max(0.0f, 1 - u1));
							float th;
							int32_t hq;
							my++;
							if (!P_.accel.closest(O, d, 0.0f, qi, P, tol, th, hq)) continue;   // (open sky: the ambient light as it is)
							if (th < near) occluded += 1 - th / near;
							const quad &h = s.quads[hq];
							const v3 H = O + d * th;
							const v3 Nh = facing(hq, d * -1.0f);
							float lambert = ll > 0 ? std::max(0.0f, dot(Nh, L)) : 0;
							if (lambert > 0) {
								my++;
								if (P_.accel.any_hit(H + Nh * eps_at(H.z), L, 0.0f, hq, H, eps_at(H.z))) lambert = 0;
							}
							light_param lp{ 0.5f, 0.5f, 0, 0 };
							if (h.lightmode < s.lightparams.size()) lp = s.lightparams[h.lightmode];
							const float lum = std::min(1.0f, lp.a + lp.d * lambert);
							const v3 alb = rgb_of(h.albedo);
							bounce = bounce + alb * lum;
						}
						const float ao = 1 - occluded / nrays;
						bounce = bounce * (1.0f / nrays);
						const float keep = same ? 0.85f : like ? 0.6f : 0.0f;
						G.ao[gi_] = G.ao[gi_] * keep + ao * (1 - keep);
						G.rgb[gi_ * 3 + 0] = G.rgb[gi_ * 3 + 0] * keep + bounce.x * (1 - keep);
						G.rgb[gi_ * 3 + 1] = G.rgb[gi_ * 3 + 1] * keep + bounce.y * (1 - keep);
						G.rgb[gi_ * 3 + 2] = G.rgb[gi_ * 3 + 2] * keep + bounce.z * (1 - keep);
						G.id[gi_] = qi;
					}
				gr += my;
			}, 2);
			res.gi_rays += gr;
		}
		// A pixel's global illumination: the grid's points around it, weighed by nearness and by how
		// alike their surface is to its own (the same polygon fully; another facing the same way at
		// about the same depth, partly), so a small polygon takes its neighbours' light, not one point's noise.
		auto gi_at = [&](int x, int y, int32_t qi, v3 N, float z, float &ao, v3 &bounce) {
			const float fx = float(x - G.x0 - G.step / 2) / G.step, fy = float(y - G.y0 - G.step / 2) / G.step;
			const int gx0 = int(std::floor(fx)), gy0 = int(std::floor(fy));
			float wsum = 0, a = 0;
			v3 c{ 0, 0, 0 };
			for (int dy = -2; dy <= 3; dy++)
				for (int dx = -2; dx <= 3; dx++) {
					const int gx = gx0 + dx, gy = gy0 + dy;
					if (gx < 0 || gy < 0 || gx >= G.w || gy >= G.h) continue;
					const size_t k = size_t(gy) * G.w + gx;
					if (G.id[k] < 0) continue;
					float like = 1;
					if (G.id[k] != qi) {
						const float *nz = &G.nz[k * 4];
						const float nd = nz[0] * N.x + nz[1] * N.y + nz[2] * N.z;
						if (nd < 0.7f) continue;
						const float dz = std::fabs(nz[3] - z) / (0.05f * z);
						if (dz >= 1) continue;
						like = (nd - 0.7f) / 0.3f * (1 - dz) * 0.5f;
					}
					const float ddx = fx - gx, ddy = fy - gy;
					const float w = like * std::exp(-0.35f * (ddx * ddx + ddy * ddy));
					wsum += w;
					a += G.ao[k] * w;
					c = c + v3{ G.rgb[k * 3], G.rgb[k * 3 + 1], G.rgb[k * 3 + 2] } * w;
				}
			if (wsum <= 1e-4f) { ao = 1; bounce = { 0, 0, 0 }; return false; }
			ao = a / wsum;
			bounce = c * (1.0f / wsum);
			return true;
		};

		// The shadows. A ray toward the light from each pixel the game lit (unless its side faces away
		// from the light: then it's left lit, not guessed). Every other pixel each way first; a pixel
		// between ones on its own polygon that all agree takes their answer, the rest (a shadow's edges)
		// trace their own.
		auto needs_ray = [&](const quad &q) { return trace && q.shade != q.lit && !(q.flags & UNLIT); };
		auto shadow_ray = [&](int x, int y, size_t i, int32_t qi) -> uint8_t {
			const v3 P = place(x + 0.5f, y + 0.5f, b.depth[i]);
			const v3 N = facing(qi, P * -1.0f);
			if (dot(N, L) <= 0) return 1;
			const float eps = eps_at(P.z);
			return P_.accel.any_hit(P + N * eps + L * eps, L, 0.0f, qi, P, 2 * eps) ? 2 : 1;
		};
		std::atomic<size_t> rays{ 0 }, shadowed{ 0 };
		if (trace)
			parallel_rows((clip[3] - clip[1]) / 2 + 1, [&](int r0, int r1) {
				size_t my = 0;
				for (int yy = r0; yy < r1; yy++) {
					const int y = clip[1] + yy * 2;
					for (int x = clip[0]; x <= clip[2]; x += 2) {
						const size_t i = size_t(y) * t.w + x;
						const int32_t qi = b.id[i];
						if (qi < 0 || !needs_ray(s.quads[qi])) { b.vis[i] = 0; continue; }
						b.vis[i] = shadow_ray(x, y, i, qi);
						my++;
					}
				}
				rays += my;
			}, 4);

		// the colours
		parallel_rows(clip[3] - clip[1] + 1, [&](int r0, int r1) {
			size_t my_rays = 0, my_shadowed = 0;
			for (int y = clip[1] + r0; y < clip[1] + r1; y++) {
				uint32_t *row = t.px + size_t(y) * t.stride;
				const bool yodd = (y - clip[1]) & 1;
				const int ys[2] = { yodd ? y - 1 : y, yodd && y + 1 <= clip[3] ? y + 1 : yodd ? y - 1 : y };
				for (int x = clip[0]; x <= clip[2]; x++) {
					const size_t i = size_t(y) * t.w + x;
					const int32_t qi = b.id[i];
					if (qi < 0) continue;
					const quad &q = s.quads[qi];
					bool lit = true;
					if (needs_ray(q)) {
						const bool even = !((x - clip[0]) & 1) && !((y - clip[1]) & 1);
						uint8_t vis = even ? b.vis[i] : 0;
						if (!even) {
							// the traced pixels around it: on its polygon and all agreeing?
							const bool xodd = (x - clip[0]) & 1;
							const int xs[2] = { xodd ? x - 1 : x, xodd && x + 1 <= clip[2] ? x + 1 : xodd ? x - 1 : x };
							uint8_t got = 0;
							bool ok = true;
							for (int a = 0; a < 2 && ok; a++)
								for (int c = 0; c < 2 && ok; c++) {
									const size_t j = size_t(ys[a]) * t.w + xs[c];
									if (b.id[j] != qi || !b.vis[j] || (got && b.vis[j] != got)) ok = false;
									else got = b.vis[j];
								}
							if (ok) vis = got;
						}
						if (!vis) { vis = shadow_ray(x, y, i, qi); my_rays++; }
						if (vis == 2) { lit = false; my_shadowed++; }
					}
					uint32_t c = lit ? q.lit : q.shade;
					if (gi && !(q.flags & UNLIT)) {
						const v3 P = place(x + 0.5f, y + 0.5f, b.depth[i]);
						float ao;
						v3 bounce;
						if (gi_at(x, y, qi, facing(qi, P * -1.0f), P.z, ao, bounce)) {
							// the ambient part (its colour in shadow) darkened where it's hemmed in, the direct
							// part as lit or not, and the light bounced onto it from round about
							const v3 amb = rgb_of(q.shade), full = rgb_of(q.lit), alb = rgb_of(q.albedo);
							const v3 direct = lit ? v3{ std::max(0.0f, full.x - amb.x), std::max(0.0f, full.y - amb.y), std::max(0.0f, full.z - amb.z) } : v3{ 0, 0, 0 };
							const float occ = 1 - o.ao_strength * (1 - ao), docc = 1 - o.ao_direct * o.ao_strength * (1 - ao);
							const v3 add = v3{ alb.x * bounce.x, alb.y * bounce.y, alb.z * bounce.z } * o.gi_strength;
							const v3 out = o.gi_view == 1 ? v3{ occ, occ, occ } : o.gi_view == 2 ? add : amb * occ + direct * docc + add;
							c = pack(out);
						}
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
		if (o.game_shadows && !P_.shadow_tris.empty()) {
			std::mutex mm;
			parallel_rows(clip[3] - clip[1] + 1, [&](int r0, int r1) {
				const int y0 = clip[1] + r0, y1 = clip[1] + r1 - 1;
				int m0 = 1 << 30, m1 = -1;
				for (const stri &tr : P_.shadow_tris) {
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
