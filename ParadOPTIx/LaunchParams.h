#pragma once

#include "include/gdt/gdt/math/vec.h"

namespace ParadOPTIx {
	using namespace gdt;

	struct LaunchParams
	{
		int			frameID{ 0 };
		uint32_t*	colorBuffer;
		vec2i		fbSize;
	};
} // ::ParadOPTIx