#!/bin/bash
# Ray-traced lighting's GPU kernels (src/raytrace/rt_gpu.comp, GLSL) compiled with glslang into SPIR-V
# for SDL_GPU's Vulkan, into src/raytrace/rt_gpu_shaders.h, which is committed: the builds don't need
# glslang. Run after changing rt_gpu.comp.   GLSLANG=/path/to/glslangValidator (default: on the PATH)
set -e
cd "$(dirname "$0")/../.."
OUT=build/shaders/rt
mkdir -p $OUT
KERNELS="clear depth near id moire gi shadow fog colour"
GLSLANG=${GLSLANG:-glslangValidator}
for k in $KERNELS; do
	K=$(echo $k | tr a-z A-Z)
	$GLSLANG -V --target-env vulkan1.0 -DK_$K -S comp src/raytrace/rt_gpu.comp -o $OUT/$k.spv >/dev/null
done
python3 - "$OUT" $KERNELS > src/raytrace/rt_gpu_shaders.h <<'PY'
import sys, hashlib
out, kernels = sys.argv[1], sys.argv[2:]
print("// Made by tools/gpu/rt_shaders.sh from src/raytrace/rt_gpu.comp (glslang): don't edit.")
print("#pragma once")
print("#include <cstdint>")
for k in kernels:
    data = open(f"{out}/{k}.spv", "rb").read()
    print(f"alignas(4) static const uint8_t rt_gpu_{k}_spv[{len(data)}] = {{")
    for i in range(0, len(data), 24):
        print("\t" + ",".join(str(b) for b in data[i:i + 24]) + ",")
    print("};")
print(f'static const char rt_gpu_shaders_source_sha1[] = "{hashlib.sha1(open("src/raytrace/rt_gpu.comp", "rb").read()).hexdigest()}";')
PY
