#include "Renderer.h"
// The following include may only appear in a single source file
#include <optix_function_table_definition.h>

namespace ParadOPTIx {

	extern "C" char embedded_ptx_code[];

	// SBT record for a raygen program
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

		void* data;
	};

	// SBT record for a miss program
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

		void* data;
	}

	// SBT record for a hitgroup program
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

		int objectID;
	}
}