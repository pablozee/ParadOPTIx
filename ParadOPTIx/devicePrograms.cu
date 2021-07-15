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

	// Closest hit and anyhit programs for radiance-type rays
	// Will eventually need one pair for each ray type and each geometry type
	// we want to render

	extern "C" __global__ void __closesthit__radiance()
	{}

	extern "C" __global__ void __anyhit__radiance()
	{}

	// Miss program that gets called for any ray that did not have a
	// valid intersection

	extern "C" __global__ void __miss__radiance()
	{}

	// Ray gen program - the actual rendering happens here
	extern "C" __global__ void __raygen__renderFrame()
	{
		const int frameID = optixLaunchParams.frameID;
		if (frameID == 0 &&
			optixGetLaunchIndex().x   == 0 &&
			optixGetLaunchIndex().y   == 0)
		{
			/*!
			* We use the optixLaunchParams rather than querying the optixGetLaunchDims
			* to ensure the optixLaunchParams are not being optimized away when not 
			* being used
			*/
			printf("############################################\n");
			printf("Hello world from OptiX 7 raygen program!\n(within a %ix%i-sized launch)\n",
				optixLaunchParams.fbSize.x,
				optixLaunchParams.fbSize.y);
			printf("############################################\n");
		}

		// Compute a test pattern based on pixel ID
		const int ix = optixGetLaunchIndex().x;
		const int iy = optixGetLaunchIndex().y;

		const int r = ((ix + frameID) % 256);
		const int g = ((iy + frameID) % 256);
		const int b = ((ix + iy + frameID) % 256);

		// Convert to 32-bit rgba value setting alpha to 0xff
		const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);

		// Write to frame buffer
		const uint32_t fbIndex = ix + iy * optixLaunchParams.fbSize.x;
		optixLaunchParams.colorBuffer[fbIndex] = rgba;
	}

}