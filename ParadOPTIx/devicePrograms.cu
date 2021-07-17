#include <optix_device.h>

#include "LaunchParams.h"

using namespace ParadOPTIx;

namespace ParadOPTIx {

	/*!
		Launch parameters in constant memory, filled in by optix upon
		optixLaunch. This gets filled in from the buffer we pass to optixLaunch,
		passed from host to device
	*/
	extern "C" __constant__ LaunchParams optixLaunchParams;

	// For this simple example we have a single ray type
	enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

	static __forceinline__ __device__
	void* unpackPointer(uint32_t i0, uint32_t i1)
	{
		const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
		void* ptr			= reinterpret_cast<void*>( uptr );
		return ptr;
	}

	static __forceinline__ __device__
	void packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
	{
		const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
		i0 = uptr >> 32;
		i1 = uptr & 0x00000000ffffffff;
	}

	template<typename T>
	static __forceinline__ __device__ T* getPRD()
	{
		const uint32_t u0 = optixGetPayload_0();
		const uint32_t u1 = optixGetPayload_1();
		return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
	}

	// Closest hit and anyhit programs for radiance-type rays
	// Will eventually need one pair for each ray type and each geometry type
	// we want to render

	extern "C" __global__ void __closesthit__radiance()
	{
		const TriangleMeshSBTData& sbtData
			= *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

		// Compute normal
		const int primID   = optixGetPrimitiveIndex();
		const vec3i index  = sbtData.index[primID];
		const vec3f& A     = sbtData.vertex[index.x];
		const vec3f& B     = sbtData.vertex[index.y];
		const vec3f& C     = sbtData.vertex[index.z];
		const vec3f Ng     = normalize(cross(B - A, C - A));

		const vec3f rayDir = optixGetWorldRayDirection();
		const float cosDN = 0.2f + 0.8f * fabsf(dot(rayDir, Ng));
		vec3f& prd = *(vec3f*)getPRD<vec3f>();
		prd = cosDN * sbtData.color;
	}

	extern "C" __global__ void __anyhit__radiance()
	{}

	// Miss program that gets called for any ray that did not have a
	// valid intersection

	extern "C" __global__ void __miss__radiance()
	{
		vec3f& prd = *(vec3f*)getPRD<vec3f>();
		// set to constant white as background colour
		prd = vec3f(1.f);
	}

	// Ray gen program - the actual rendering happens here
	extern "C" __global__ void __raygen__renderFrame()
	{
		const int ix = optixGetLaunchIndex().x;
		const int iy = optixGetLaunchIndex().y;

		const auto& camera = optixLaunchParams.camera;

		// Per ray data - will be overwritten by miss or hit program
		vec3f pixelColorPRD = vec3f(0.f);

		// The values we store the PRD pointer in
		uint32_t u0, u1;
		packPointer( &pixelColorPRD, u0, u1 );

		// Normalized screen plane position, in [0, 1]^2
		const vec2f screen(vec2f(ix + .5f, iy + .5f) / vec2f(optixLaunchParams.frame.size));

		// Generate ray direction
		vec3f rayDir = normalize(camera.direction
								 + (screen.x - 0.5f) * camera.horizontal
								 + (screen.y - 0.5f) * camera.vertical);

		optixTrace(optixLaunchParams.traversable,
				   camera.position,
				   rayDir,
				   0.f,			// tmin
				   1e20f,			// tmax
				   0.0f,			// rayTime
				   OptixVisibilityMask(255),
				   OPTIX_RAY_FLAG_DISABLE_ANYHIT,
				   SURFACE_RAY_TYPE,			// SBT offset
				   RAY_TYPE_COUNT,				// SBT stride
				   SURFACE_RAY_TYPE,			// missSBTIndex
				   u0,
				   u1
				  );

		const int r = int(255.99f * pixelColorPRD.x);
		const int g = int(255.99f * pixelColorPRD.y);
		const int b = int(255.99f * pixelColorPRD.z);

		// Convert to 32-bit rgba value setting alpha to 0xff
		const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);

		// Write to frame buffer
		const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
		optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
	}

}