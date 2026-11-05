// Ray-traced lighting on the GPU: rt_render.cpp's picture drawn by compute kernels (rt_gpu.comp)
// through SDL_GPU's Vulkan, on a device of its own; each picture's result comes back into the
// target (the CPU only makes each view ready: its polygons in order and projected, the BVH).
#pragma once
#include "rt_render.h"
#include <memory>

namespace rt {

class gpu_renderer
{
public:
	virtual ~gpu_renderer() = default;
	virtual const char *name() const = 0;
	// As rt::render, on the GPU. False: not drawn (the target as it was: the caller draws it on the CPU).
	virtual bool render(const scene &s, const target &t, const options &o, result &r) = 0;
	double last_ms = 0;   // the last picture's time, from the passes' start to its pixels in the target
};

// A Vulkan device of its own (SDL_GPU, SDL's video started). Null if none.
std::unique_ptr<gpu_renderer> gpu_create();

} // namespace rt
