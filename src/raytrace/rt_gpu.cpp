// Ray-traced lighting on the GPU (rt_gpu.h): SDL_GPU's Vulkan, the kernels of rt_gpu.comp
// (precompiled into rt_gpu_shaders.h by tools/gpu/rt_shaders.sh). For each view the CPU makes it
// ready (rt::detail::prepare) and lists the rows each triangle covers; the polygons, rows, BVH and
// light modes go up, the kernels run in turn (each its own compute pass: SDL_GPU orders passes, not
// the dispatches inside one), and after the last view each pixel's result comes back and goes into
// the target on every core. Global illumination's grid stays on the GPU from picture to picture.
#include "rt_gpu.h"
#include "rt_detail.h"
#include "rt_gpu_shaders.h"
#include "../machine/parallel.h"
#include <SDL3/SDL.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace rt {
using namespace detail;
namespace {

using clk = std::chrono::steady_clock;

enum : uint32_t { F_TRACE = 1, F_GI = 2, F_GAME_SHADOWS = 4, F_BLEND = 8, F_CLEAR_OUT = 16, F_FOG = 32, F_SPOT = 64 };
enum { RO_TRIS, RO_JOBS, RO_QUADS, RO_NODES, RO_BTRIS, RO_LIGHTS, RO_N };
enum { RW_DEPTH, RW_NEAR, RW_ID, RW_VIS, RW_MOIRE, RW_OUT, RW_GRID, RW_FOG, RW_N };
enum { K_CLEAR, K_DEPTH, K_NEAR, K_ID, K_MOIRE, K_GI, K_SHADOW, K_FOG, K_COLOUR, K_N };

struct uniforms   // (rt_gpu.comp's U, std140: all four-word rows)
{
	uint32_t u0[4], u1[4];
	float v0[4], v1[4], lt[4];
	uint32_t g0[4], g1[4];
	float g2[4];
	uint32_t k0[4];
	float x0[4];
	uint32_t x1[4];
	float x2[4];
};

inline uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

class sdl_gpu_renderer final : public gpu_renderer
{
public:
	explicit sdl_gpu_renderer(SDL_GPUDevice *dev) : m_dev(dev) {}
	~sdl_gpu_renderer() override
	{
		for (SDL_GPUComputePipeline *p : m_k) if (p) SDL_ReleaseGPUComputePipeline(m_dev, p);
		for (buf &b : m_ro) if (b.b) SDL_ReleaseGPUBuffer(m_dev, b.b);
		for (buf &b : m_rw) if (b.b) SDL_ReleaseGPUBuffer(m_dev, b.b);
		for (grid &g : m_grids) { if (g.b.b) SDL_ReleaseGPUBuffer(m_dev, g.b.b); if (g.fog.b) SDL_ReleaseGPUBuffer(m_dev, g.fog.b); }
		if (m_up) SDL_ReleaseGPUTransferBuffer(m_dev, m_up);
		if (m_down) SDL_ReleaseGPUTransferBuffer(m_dev, m_down);
		SDL_DestroyGPUDevice(m_dev);
	}
	bool init()
	{
		if (!(SDL_GetGPUShaderFormats(m_dev) & SDL_GPU_SHADERFORMAT_SPIRV)) { fprintf(stderr, "rt gpu: %s takes no SPIR-V\n", SDL_GetGPUDeviceDriver(m_dev)); return false; }
		struct { const uint8_t *code; size_t n; const char *name; } ks[K_N] = {
			{ rt_gpu_clear_spv, sizeof rt_gpu_clear_spv, "clear" }, { rt_gpu_depth_spv, sizeof rt_gpu_depth_spv, "depth" },
			{ rt_gpu_near_spv, sizeof rt_gpu_near_spv, "near" }, { rt_gpu_id_spv, sizeof rt_gpu_id_spv, "id" },
			{ rt_gpu_moire_spv, sizeof rt_gpu_moire_spv, "moire" }, { rt_gpu_gi_spv, sizeof rt_gpu_gi_spv, "gi" },
			{ rt_gpu_shadow_spv, sizeof rt_gpu_shadow_spv, "shadow" }, { rt_gpu_fog_spv, sizeof rt_gpu_fog_spv, "fog" }, { rt_gpu_colour_spv, sizeof rt_gpu_colour_spv, "colour" },
		};
		// (debugging: RT_GPU_SPV_DIR=dir loads <dir>/<kernel>.spv instead, tools/gpu/rt_shaders.sh's build/shaders/rt)
		const char *spv_dir = getenv("RT_GPU_SPV_DIR");
		std::vector<uint8_t> spv;
		for (int k = 0; k < K_N; k++) {
			SDL_GPUComputePipelineCreateInfo ci = {};
			ci.code = ks[k].code;
			ci.code_size = ks[k].n;
			if (spv_dir) {
				const std::string path = std::string(spv_dir) + "/" + ks[k].name + ".spv";
				if (FILE *f = fopen(path.c_str(), "rb")) {
					spv.assign(1 << 20, 0);
					spv.resize(fread(spv.data(), 1, spv.size(), f));
					fclose(f);
					ci.code = spv.data();
					ci.code_size = spv.size();
				}
			}
			ci.format = SDL_GPU_SHADERFORMAT_SPIRV;
			ci.entrypoint = "main";
			ci.num_readonly_storage_buffers = RO_N;
			ci.num_readwrite_storage_buffers = RW_N;
			ci.num_uniform_buffers = 1;
			ci.threadcount_x = 64; ci.threadcount_y = 1; ci.threadcount_z = 1;
			m_k[k] = SDL_CreateGPUComputePipeline(m_dev, &ci);
			if (!m_k[k]) { fprintf(stderr, "rt gpu: %s: %s\n", ks[k].name, SDL_GetError()); return false; }
		}
		const char *dn = SDL_GetStringProperty(SDL_GetGPUDeviceProperties(m_dev), SDL_PROP_GPU_DEVICE_NAME_STRING, "?");
		snprintf(m_name, sizeof m_name, "%s on %s", SDL_GetGPUDeviceDriver(m_dev), dn);
		return true;
	}
	const char *name() const override { return m_name; }

	bool render(const scene &s, const target &t, const options &o, result &res) override
	{
		res = result{};
		const auto t0 = clk::now();
		if (t.w <= 0 || t.h <= 0) return false;
		const uint32_t npx = uint32_t(t.w) * uint32_t(t.h);
		if (m_light_sign.size() < s.views.size()) m_light_sign.resize(s.views.size(), 1);
		if (m_grids.size() < s.views.size()) m_grids.resize(s.views.size());

		// the polygons and light modes, the picture's (every view's polygons by their place in s.quads)
		m_planes.assign(s.quads.size() * 4, 0.0f);
		m_views.resize(s.views.size());
		int ry0 = t.h, ry1 = -1;   // the rows any view draws: what comes back
		bool any = false;
		for (size_t vi = 0; vi < s.views.size(); vi++) {
			m_views[vi].ok = prepare(s, vi, t, o, m_planes, m_light_sign[vi], m_views[vi].p);
			if (!m_views[vi].ok) continue;
			any = true;
			ry0 = std::min(ry0, m_views[vi].p.clip[1]);
			ry1 = std::max(ry1, m_views[vi].p.clip[3]);
		}
		if (!any) return true;
		m_quads.resize(std::max<size_t>(s.quads.size(), 1) * 8);
		for (size_t k = 0; k < s.quads.size(); k++) {
			const quad &q = s.quads[k];
			uint32_t *d = &m_quads[k * 8];
			d[0] = q.lit; d[1] = q.shade; d[2] = q.albedo; d[3] = uint32_t(q.flags) | uint32_t(q.lightmode) << 8;
			for (int i = 0; i < 4; i++) d[4 + i] = bits(m_planes[k * 4 + i]);
		}
		m_lights.assign(256 * 4, 0);
		for (size_t i = 0; i < 256; i++) {
			light_param lp{ 0.5f, 0.5f, 0, 0 };
			if (i < s.lightparams.size()) lp = s.lightparams[i];
			m_lights[i * 4] = bits(lp.a); m_lights[i * 4 + 1] = bits(lp.d); m_lights[i * 4 + 2] = bits(lp.s); m_lights[i * 4 + 3] = uint32_t(lp.p);
		}

		// the picture's buffers (each pixel's words), and what comes back
		const uint32_t px_bytes = npx * 4;
		for (int i = 0; i < RW_GRID; i++) if (!grow(m_rw[i], px_bytes, true)) return false;
		const uint32_t back_rows = uint32_t(ry1 - ry0 + 1), back_bytes = back_rows * uint32_t(t.w) * 4;
		if (!transfer(m_down, m_down_n, back_bytes, SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD)) return false;

		SDL_GPUCommandBuffer *cb = SDL_AcquireGPUCommandBuffer(m_dev);
		if (!cb) return false;
		const uint32_t pic = ++m_pictures;
		bool first = true;
		size_t vlast = 0;
		for (size_t vi = 0; vi < s.views.size(); vi++) if (m_views[vi].ok) vlast = vi;
		for (size_t vi = 0; vi < s.views.size(); vi++) {
			if (!m_views[vi].ok) continue;
			const view &v = s.views[vi];
			prepared &P = m_views[vi].p;
			const int *clip = P.clip;
			res.tris += P.tris.size();
			const bool trace = o.shadows && P.ll > 0 && !P.accel.nodes.empty();
			const bool gi = o.gi && !P.accel.nodes.empty();
			const bool moire = o.game_shadows && !P.shadow_tris.empty();

			// the triangles (solid, then the game's shadows), their rows, the BVH
			const size_t nt = P.tris.size(), ns = P.shadow_tris.size();
			m_tris.assign(std::max<size_t>(nt + ns, 1) * 16, 0);
			m_jobs.clear();
			auto put_tri = [&](const stri &tr, size_t at) {
				uint32_t *d = &m_tris[at * 16];
				for (int i = 0; i < 3; i++) { d[i * 3] = bits(tr.x[i]); d[i * 3 + 1] = bits(tr.y[i]); d[i * 3 + 2] = bits(tr.w[i]); }
				d[9] = bits(tr.area);
				d[10] = uint32_t(tr.minx); d[11] = uint32_t(tr.maxx); d[12] = uint32_t(tr.miny); d[13] = uint32_t(tr.maxy);
				d[14] = uint32_t(tr.quad);
				for (int y = tr.miny; y <= tr.maxy; y++) { m_jobs.push_back(uint32_t(at)); m_jobs.push_back(uint32_t(y)); }
			};
			for (size_t i = 0; i < nt; i++) put_tri(P.tris[i], i);
			const uint32_t solid_jobs = uint32_t(m_jobs.size() / 2);
			for (size_t i = 0; i < ns; i++) put_tri(P.shadow_tris[i], nt + i);
			const uint32_t all_jobs = uint32_t(m_jobs.size() / 2);
			if (m_jobs.empty()) m_jobs.assign(2, 0);
			m_nodes.assign(std::max<size_t>(P.accel.nodes.size(), 1) * 8, 0);
			for (size_t i = 0; i < P.accel.nodes.size(); i++) {
				const bnode &n = P.accel.nodes[i];
				uint32_t *d = &m_nodes[i * 8];
				d[0] = bits(n.lo.x); d[1] = bits(n.lo.y); d[2] = bits(n.lo.z); d[3] = uint32_t(n.left);
				d[4] = bits(n.hi.x); d[5] = bits(n.hi.y); d[6] = bits(n.hi.z); d[7] = uint32_t(n.first) << 3 | uint32_t(n.count);
			}
			m_btris.assign(std::max<size_t>(P.accel.tris.size(), 1) * 16, 0);
			for (size_t i = 0; i < P.accel.tris.size(); i++) {
				const btri &b = P.accel.tris[i];
				uint32_t *d = &m_btris[i * 16];
				const v3 vs[4] = { b.p0, b.e1, b.e2, b.n };
				for (int k = 0; k < 4; k++) { d[k * 3] = bits(vs[k].x); d[k * 3 + 1] = bits(vs[k].y); d[k * 3 + 2] = bits(vs[k].z); }
				d[12] = uint32_t(b.quad);
			}

			// global illumination's grid (made again when the view's size or place changes)
			const int g = o.gi_step > 0 ? o.gi_step : std::max(2, int(v.scale * 2 + 0.5f));
			const int gw = (clip[2] - clip[0]) / g + 1, gh = (clip[3] - clip[1]) / g + 1;
			grid &G = m_grids[vi];
			bool grid_new = false;
			if (gi && (G.w != gw || G.h != gh || G.step != g || G.x0 != clip[0] || G.y0 != clip[1] || !G.b.b)) {
				G.w = gw; G.h = gh; G.step = g; G.x0 = clip[0]; G.y0 = clip[1];
				if (!grow(G.b, uint32_t(gw) * gh * 48, true)) { SDL_CancelGPUCommandBuffer(cb); return false; }
				grid_new = true;
			}
			if (!G.b.b && !grow(G.b, 48, true)) { SDL_CancelGPUCommandBuffer(cb); return false; }
			// volumetric lighting's grid: a point a game pixel (cleared to "nothing kept" when made)
			const bool fog = o.fog && P.ll > 0 && !P.accel.nodes.empty();
			const int fs = std::max(1, int(v.scale + 0.5f));
			const int fw = (clip[2] - clip[0]) / fs + 1, fh = (clip[3] - clip[1]) / fs + 1;
			bool fog_new = false;
			if (fog && (G.fw != fw || G.fh != fh || G.fstep != fs || !G.fog.b)) {
				G.fw = fw; G.fh = fh; G.fstep = fs;
				if (!grow(G.fog, uint32_t(fw) * fh * 16, true)) { SDL_CancelGPUCommandBuffer(cb); return false; }
				fog_new = true;
			}
			if (!G.fog.b && !grow(G.fog, 16, true)) { SDL_CancelGPUCommandBuffer(cb); return false; }

			// up
			const size_t sizes[RO_N] = { m_tris.size() * 4, m_jobs.size() * 4, m_quads.size() * 4, m_nodes.size() * 4, m_btris.size() * 4, m_lights.size() * 4 };
			const void *srcs[RO_N] = { m_tris.data(), m_jobs.data(), m_quads.data(), m_nodes.data(), m_btris.data(), m_lights.data() };
			size_t up = 0;
			for (int i = 0; i < RO_N; i++) { if (!grow(m_ro[i], uint32_t(sizes[i]), false)) { SDL_CancelGPUCommandBuffer(cb); return false; } up += sizes[i]; }
			std::vector<uint32_t> grid_init;
			if (grid_new) {
				grid_init.assign(size_t(gw) * gh * 12, 0);
				for (size_t k = 0; k < size_t(gw) * gh; k++) { grid_init[k * 12] = 0xffffffffu; grid_init[k * 12 + 1] = bits(1.0f); }
				up += grid_init.size() * 4;
			}
			const size_t fog_bytes = fog_new ? size_t(fw) * fh * 16 : 0;
			up += fog_bytes;
			if (!transfer(m_up, m_up_n, uint32_t(up), SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD)) { SDL_CancelGPUCommandBuffer(cb); return false; }
			uint8_t *map = (uint8_t *)SDL_MapGPUTransferBuffer(m_dev, m_up, true);
			if (!map) { SDL_CancelGPUCommandBuffer(cb); return false; }
			size_t at = 0;
			uint32_t offs[RO_N + 1];
			for (int i = 0; i < RO_N; i++) { offs[i] = uint32_t(at); memcpy(map + at, srcs[i], sizes[i]); at += sizes[i]; }
			offs[RO_N] = uint32_t(at);
			if (grid_new) { memcpy(map + at, grid_init.data(), grid_init.size() * 4); at += grid_init.size() * 4; }
			const uint32_t fog_at = uint32_t(at);
			if (fog_new) memset(map + at, 0, fog_bytes);
			SDL_UnmapGPUTransferBuffer(m_dev, m_up);
			{
				SDL_GPUCopyPass *cp = SDL_BeginGPUCopyPass(cb);
				for (int i = 0; i < RO_N; i++) {
					SDL_GPUTransferBufferLocation l = { m_up, offs[i] };
					SDL_GPUBufferRegion r = { m_ro[i].b, 0, uint32_t(sizes[i]) };
					SDL_UploadToGPUBuffer(cp, &l, &r, false);
				}
				if (grid_new) {
					SDL_GPUTransferBufferLocation l = { m_up, offs[RO_N] };
					SDL_GPUBufferRegion r = { G.b.b, 0, uint32_t(grid_init.size() * 4) };
					SDL_UploadToGPUBuffer(cp, &l, &r, false);
				}
				if (fog_new) {
					SDL_GPUTransferBufferLocation l = { m_up, fog_at };
					SDL_GPUBufferRegion r = { G.fog.b, 0, uint32_t(fog_bytes) };
					SDL_UploadToGPUBuffer(cp, &l, &r, false);
				}
				SDL_EndGPUCopyPass(cp);
			}

			// the passes
			uniforms u = {};
			u.u0[0] = uint32_t(t.w); u.u0[1] = uint32_t(clip[0]); u.u0[2] = uint32_t(clip[1]); u.u0[3] = uint32_t(clip[2]);
			u.u1[0] = uint32_t(clip[3]);
			u.u1[3] = (trace ? F_TRACE : 0) | (gi ? F_GI : 0) | (moire ? F_GAME_SHADOWS : 0) | (o.blend_shadows ? F_BLEND : 0) | (fog ? F_FOG : 0) | (o.spot && P.ll > 0 ? F_SPOT : 0);
			u.v0[0] = v.xc; u.v0[1] = v.yc; u.v0[2] = v.zoomx; u.v0[3] = v.zoomy;
			u.v1[0] = v.viewx; u.v1[1] = v.viewy; u.v1[2] = v.ext; u.v1[3] = v.scale;
			u.lt[0] = P.L.x; u.lt[1] = P.L.y; u.lt[2] = P.L.z; u.lt[3] = P.ll;
			u.g0[0] = uint32_t(g); u.g0[1] = uint32_t(gw); u.g0[2] = uint32_t(gh); u.g0[3] = uint32_t(std::max(1, o.gi_rays));
			u.g1[0] = pic; u.g1[1] = uint32_t(clip[0]); u.g1[2] = uint32_t(clip[1]); u.g1[3] = uint32_t(nt);
			if (!P.accel.nodes.empty()) { const v3 e = P.accel.nodes[0].hi - P.accel.nodes[0].lo; u.g2[0] = std::sqrt(dot(e, e)); }
			u.g2[1] = o.gi_strength; u.g2[2] = o.ao_strength; u.g2[3] = o.ao_direct;
			u.k0[0] = o.key; u.k0[1] = o.shadow_key; u.k0[2] = o.mark; u.k0[3] = uint32_t(o.gi_view);
			u.x0[0] = o.ao_radius; u.x0[1] = o.fog_density / P.zref; u.x0[2] = o.fog_strength; u.x0[3] = P.zref;
			u.x1[0] = uint32_t(fs); u.x1[1] = uint32_t(fw); u.x1[2] = uint32_t(fh); u.x1[3] = uint32_t(std::max(1, o.fog_steps));
			u.x2[0] = o.spot_inner; u.x2[1] = o.spot_outer; u.x2[2] = o.spot_dim; u.x2[3] = o.spot_sky;
			static const unsigned skip = getenv("RT_GPU_SKIP") ? unsigned(strtoul(getenv("RT_GPU_SKIP"), nullptr, 0)) : 0;   // (debugging: kernels left out, a bit each)
			auto pass = [&](int k, uint32_t threads, const uniforms &uu) {
				if (!threads || (skip >> k & 1)) return;
				SDL_GPUStorageBufferReadWriteBinding bw[RW_N] = {};
				for (int i = 0; i < RW_GRID; i++) bw[i].buffer = m_rw[i].b;
				bw[RW_GRID].buffer = G.b.b;
				bw[RW_FOG].buffer = G.fog.b;
				SDL_GPUComputePass *c = SDL_BeginGPUComputePass(cb, nullptr, 0, bw, RW_N);
				SDL_BindGPUComputePipeline(c, m_k[k]);
				SDL_GPUBuffer *ro[RO_N];
				for (int i = 0; i < RO_N; i++) ro[i] = m_ro[i].b;
				SDL_BindGPUComputeStorageBuffers(c, 0, ro, RO_N);
				uniforms q = uu;
				q.u1[1] = threads;
				SDL_PushGPUComputeUniformData(cb, 0, &q, sizeof q);
				const uint32_t groups = (threads + 63) / 64;
				SDL_DispatchGPUCompute(c, std::min<uint32_t>(groups, 65535), (groups + 65534) / 65535, 1);
				SDL_EndGPUComputePass(c);
			};
			if (first) {   // (the results' rows cleared once, before the first view)
				uniforms c = u;
				c.u0[1] = 0; c.u0[2] = uint32_t(ry0); c.u0[3] = uint32_t(t.w - 1); c.u1[0] = uint32_t(ry1);
				c.u1[3] = F_CLEAR_OUT;
				pass(K_CLEAR, back_rows * uint32_t(t.w), c);
				first = false;
			}
			const uint32_t cw = uint32_t(clip[2] - clip[0] + 1), ch = uint32_t(clip[3] - clip[1] + 1);
			pass(K_CLEAR, cw * ch, u);
			pass(K_DEPTH, solid_jobs, u);
			pass(K_NEAR, solid_jobs, u);
			pass(K_ID, solid_jobs, u);
			if (moire) { uniforms m = u; m.u1[2] = solid_jobs; pass(K_MOIRE, all_jobs - solid_jobs, m); }
			if (gi) pass(K_GI, uint32_t(gw) * uint32_t(gh), u);
			if (trace) pass(K_SHADOW, ((cw - 1) / 2 + 1) * ((ch - 1) / 2 + 1), u);
			if (fog) pass(K_FOG, uint32_t(fw) * uint32_t(fh), u);
			pass(K_COLOUR, cw * ch, u);

			if (vi != vlast) {
				// (the next view's data goes up through the same buffers: this one's passes done first)
				SDL_GPUFence *fe = SDL_SubmitGPUCommandBufferAndAcquireFence(cb);
				if (!fe) return false;
				const bool ok = SDL_WaitForGPUFences(m_dev, true, &fe, 1);
				SDL_ReleaseGPUFence(m_dev, fe);
				if (!ok) return false;
				cb = SDL_AcquireGPUCommandBuffer(m_dev);
				if (!cb) return false;
			}
		}

		// back, and into the picture
		{
			SDL_GPUCopyPass *cp = SDL_BeginGPUCopyPass(cb);
			SDL_GPUBufferRegion r = { m_rw[RW_OUT].b, uint32_t(ry0) * uint32_t(t.w) * 4, back_bytes };
			SDL_GPUTransferBufferLocation l = { m_down, 0 };
			SDL_DownloadFromGPUBuffer(cp, &r, &l);
			SDL_EndGPUCopyPass(cp);
		}
		SDL_GPUFence *fe = SDL_SubmitGPUCommandBufferAndAcquireFence(cb);
		if (!fe) { fprintf(stderr, "rt gpu: submit: %s\n", SDL_GetError()); return false; }
		const bool ok = SDL_WaitForGPUFences(m_dev, true, &fe, 1);
		SDL_ReleaseGPUFence(m_dev, fe);
		if (!ok) { fprintf(stderr, "rt gpu: a picture failed (%s)\n", SDL_GetError()); return false; }
		const uint32_t *back = (const uint32_t *)SDL_MapGPUTransferBuffer(m_dev, m_down, false);
		if (!back) return false;
		const uint32_t keep = 0xffffffu | o.mark;
		std::mutex mm;
		parallel_rows(int(back_rows), [&](int a, int b) {
			int m0 = 1 << 30, m1 = -1;
			for (int r = a; r < b; r++) {
				const int y = ry0 + r;
				uint32_t *p = t.px + size_t(y) * t.stride;
				const uint32_t *src = back + size_t(r) * t.w;
				for (int x = 0; x < t.w; x++) {
					const uint32_t v = src[x];
					if (!v) continue;
					if (v & 0x80000000u) {
						p[x] = v & keep;
						if (v & o.mark) { m0 = std::min(m0, y); m1 = std::max(m1, y); }
					} else if (v & 0x20000000u) {   // (the haze over the CPU's picture: the sky)
						const uint32_t c = p[x];
						if (c == o.key || c == o.shadow_key) continue;
						const float th = float((v >> 12) & 4095u) / 4095.0f, sc = float(v & 4095u) / 4095.0f * 4.0f * o.fog_strength;
						auto ch = [&](int sh, float haze) { return uint32_t(std::clamp(float((c >> sh) & 255u) * th + haze * sc * 255.0f + 0.5f, 0.0f, 255.0f)) << sh; };
						p[x] = (c & 0xff000000u) | ch(16, 1.0f) | ch(8, 0.96f) | ch(0, 0.88f);
					} else {
						const uint32_t c = p[x], half = v & 0x7f7f7fu;
						if (c == o.key) p[x] = o.shadow_key | half;
						else if (!(c & o.mark)) p[x] = o.mark | (((c >> 1) & 0x7f7f7fu) + half);
						m0 = std::min(m0, y); m1 = std::max(m1, y);
					}
				}
			}
			if (o.blend_shadows && m0 <= m1) {
				std::lock_guard<std::mutex> l(mm);
				res.mark_y1 = std::min(res.mark_y1, m0);
				res.mark_y2 = std::max(res.mark_y2, m1);
			}
		});
		SDL_UnmapGPUTransferBuffer(m_dev, m_down);
		last_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
		return true;
	}

private:
	struct buf { SDL_GPUBuffer *b = nullptr; uint32_t n = 0; };
	struct grid { buf b, fog; int w = 0, h = 0, step = 0, x0 = 0, y0 = 0, fw = 0, fh = 0, fstep = 0; };
	struct view_state { prepared p; bool ok = false; };

	bool grow(buf &b, uint32_t n, bool rw)
	{
		n = std::max<uint32_t>(n, 16);
		if (b.b && b.n >= n) return true;
		if (b.b) SDL_ReleaseGPUBuffer(m_dev, b.b);
		SDL_GPUBufferCreateInfo bi = {};
		bi.usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | (rw ? SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE : 0);
		bi.size = b.n = n + n / 4;
		b.b = SDL_CreateGPUBuffer(m_dev, &bi);
		if (!b.b) { fprintf(stderr, "rt gpu: a buffer of %u bytes: %s\n", b.n, SDL_GetError()); b.n = 0; }
		return b.b != nullptr;
	}
	bool transfer(SDL_GPUTransferBuffer *&t, uint32_t &have, uint32_t n, SDL_GPUTransferBufferUsage usage)
	{
		n = std::max<uint32_t>(n, 16);
		if (t && have >= n) return true;
		if (t) SDL_ReleaseGPUTransferBuffer(m_dev, t);
		SDL_GPUTransferBufferCreateInfo ti = {};
		ti.usage = usage;
		ti.size = have = n + n / 4;
		t = SDL_CreateGPUTransferBuffer(m_dev, &ti);
		if (!t) { fprintf(stderr, "rt gpu: a transfer buffer of %u bytes: %s\n", have, SDL_GetError()); have = 0; }
		return t != nullptr;
	}

	SDL_GPUDevice *m_dev;
	char m_name[128] = "vulkan";
	SDL_GPUComputePipeline *m_k[K_N] = {};
	buf m_ro[RO_N], m_rw[RW_GRID];
	std::vector<grid> m_grids;
	SDL_GPUTransferBuffer *m_up = nullptr, *m_down = nullptr;
	uint32_t m_up_n = 0, m_down_n = 0;
	uint32_t m_pictures = 0;
	std::vector<int> m_light_sign;
	std::vector<float> m_planes;
	std::vector<view_state> m_views;
	std::vector<uint32_t> m_quads, m_lights, m_tris, m_jobs, m_nodes, m_btris;
};

} // namespace

std::unique_ptr<gpu_renderer> gpu_create()
{
	SDL_GPUDevice *dev = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, getenv("GPU3D_VALIDATE") != nullptr, "vulkan");
	if (!dev) { fprintf(stderr, "rt gpu: no Vulkan device: %s\n", SDL_GetError()); return nullptr; }
	auto g = std::make_unique<sdl_gpu_renderer>(dev);
	if (!g->init()) return nullptr;
	fprintf(stderr, "rt gpu: %s\n", g->name());
	return g;
}

} // namespace rt
