#pragma once

#include "include/gdt/gdt/math/vec.h"

namespace ParadOPTIx {
	using namespace gdt;

	struct TriangleMeshSBTData
	{
		vec3f color;
		vec3f* vertex;
		vec3i* index;
	};

	struct LaunchParams
	{
	
		struct 
		{
			uint32_t*	colorBuffer;
			vec2i		size;
		} frame;

		struct
		{
			vec3f position;
			vec3f direction;
			vec3f horizontal;
			vec3f vertical;
		} camera;

		OptixTraversableHandle traversable;
	};
} // ::ParadOPTIx