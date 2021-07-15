#include "include/gdt/gdt/gdt.h"
#include "Renderer.h"
// The following include may only appear in a single source file
#include <optix_function_table_definition.h>
using namespace ParadOPTIx;


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
	};

	// SBT record for a hitgroup program
	struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

		int objectID;
	};

	/**
	 * Constructor - performs all setup, including initializing optix,
	 * creates module, pipeline, programs, SBT, etc
	 */
	Renderer::Renderer()
	{
		initOptix();

		std::cout << "#ParadOPTIx: creating OptiX context ..." << std::endl;
		createContext();

		std::cout << "#ParadOPTIx: setting up module ..." << std::endl;
		createModule();

		std::cout << "#ParadOPTIx: creating raygen programs ..." << std::endl;
		createRaygenPrograms();

		std::cout << "#ParadOPTIx: creating miss programs ..." << std::endl;
		createMissPrograms();

		std::cout << "#ParadOPTIx: creating hitgroup programs ..." << std::endl;
		createHitgroupPrograms();

		std::cout << "#ParadOPTIx: setting up OptiX pipeline ..." << std::endl;
		createPipeline();

		std::cout << "#ParadOPTIx: building shader binding table ..." << std::endl;
		buildSBT();

		launchParamsBuffer.alloc(sizeof(LaunchParams));

		std::cout << "#ParadOPTIx: context, module, programs, pipeline, SBT all set up ..." << std::endl;

		std::cout << GDT_TERMINAL_GREEN;
		std::cout << "#ParadOPTIx: OptiX 7 all set up" << std::endl;
		std::cout << GDT_TERMINAL_DEFAULT;
	}

	// Helper function that initializes optix and checks for errors
	void Renderer::initOptix()
	{
		std::cout << "#ParadOPTIx: initializing OptiX ..." << std::endl;

		// Check for available optix7 capable devices
		cudaFree(0);

		int numDevices;
		cudaGetDeviceCount(&numDevices);

		if (numDevices == 0)
			throw std::runtime_error("#ParadOPTIx: no CUDA capable devices found!");

		std::cout << "#ParadOPTIx: found " << numDevices << " CUDA devices" << std::endl;

		// Initialize Optix
		OPTIX_CHECK(optixInit());
		
		std::cout << GDT_TERMINAL_GREEN
				  << "#ParadOPTIx: Successfully initialized Optix"
				  << GDT_TERMINAL_DEFAULT << std::endl;
	}

	static void context_log_cb(unsigned int level,
							   const char* tag,
							   const char* message,
							   void*)
	{
		fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
	}

	// Creates and configures an OptiX device context
	void Renderer::createContext()
	{
		const int deviceID = 0;
		CUDA_CHECK(SetDevice(deviceID));
		CUDA_CHECK(StreamCreate(&cudaStream));

		cudaGetDeviceProperties(&deviceProps, deviceID);
		std::cout << "#ParadOPTIx: running on device: " << deviceProps.name << std::endl;

		CUresult cuRes = cuCtxGetCurrent(&cudaContext);
		if ( cuRes != CUDA_SUCCESS )
		{
			fprintf( stderr, "Error querying current context: error code %d\n", cuRes );
		}

		OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
		OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
	}

	/**
	 * Creates the module that contains all the programs we are going to use.
	 * We use a module from a .cu file using an embedded ptx string
	 */
	void Renderer::createModule()
	{
		moduleCompileOptions.maxRegisterCount = 50;
		moduleCompileOptions.optLevel		  = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
		moduleCompileOptions.debugLevel		  = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

		pipelineCompileOptions = {};
		pipelineCompileOptions.traversableGraphFlags			= OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		pipelineCompileOptions.usesMotionBlur					= false;
		pipelineCompileOptions.numPayloadValues					= 2;
		pipelineCompileOptions.numAttributeValues				= 2;
		pipelineCompileOptions.exceptionFlags					= OPTIX_EXCEPTION_FLAG_NONE;
		pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

		pipelineLinkOptions.maxTraceDepth = 2;

		const std::string ptxCode = embedded_ptx_code;

		char log[2048];
		size_t sizeof_log = sizeof( log );
		OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
											 &moduleCompileOptions,
											 &pipelineCompileOptions,
											 ptxCode.c_str(),
											 ptxCode.size(),
											 log,
											 &sizeof_log,
											 &module
										   ));
		if (sizeof_log > 1) PRINT(log);
	}

	// Performs setup for the raygen programs we are going to use
	void Renderer::createRaygenPrograms()
	{
		// we create a single ray gen program in this example
		raygenPGs.resize(1);

		OptixProgramGroupOptions pgOptions = {};
		OptixProgramGroupDesc pgDesc	   = {};
		pgDesc.kind						   = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		pgDesc.raygen.module			   = module;
		pgDesc.raygen.entryFunctionName	   = "__raygen__renderFrame";

		char log[2048];
		size_t sizeof_log = sizeof( log );
		OPTIX_CHECK(optixProgramGroupCreate(optixContext,
											&pgDesc,
											1,
											&pgOptions,
											log,
											&sizeof_log,
											&raygenPGs[0]
										  ));
		if (sizeof_log > 1) PRINT(log);
	}

	// Performs setup for the miss programs we are going to use
	void Renderer::createMissPrograms()
	{
		missPGs.resize(1);

		OptixProgramGroupOptions pgOptions = {};
		OptixProgramGroupDesc pgDesc	   = {};
		pgDesc.kind						   = OPTIX_PROGRAM_GROUP_KIND_MISS;
		pgDesc.miss.module				   = module;
		pgDesc.miss.entryFunctionName	   = "__miss__radiance";

		char log[2048];
		size_t sizeof_log = sizeof( log );
		OPTIX_CHECK(optixProgramGroupCreate(optixContext,
											&pgDesc,
											1,
											&pgOptions,
											log,
											&sizeof_log,
											&missPGs[0]
										  ));
		if (sizeof_log > 1) PRINT(log);
	}

	// Performs setup for the hitgroups programs we are going to use
	void Renderer::createHitgroupPrograms()
	{
		hitgroupPGs.resize(1);

		OptixProgramGroupOptions pgOptions = {};
		OptixProgramGroupDesc pgDesc	   = {};
		pgDesc.kind								= OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		pgDesc.hitgroup.moduleCH				= module;
		pgDesc.hitgroup.entryFunctionNameCH	    = "__closesthit__radiance";
		pgDesc.hitgroup.moduleAH				= module;
		pgDesc.hitgroup.entryFunctionNameAH	    = "__anyhit__radiance";

		char log[2048];
		size_t sizeof_log = sizeof( log );
		OPTIX_CHECK(optixProgramGroupCreate(optixContext,
											&pgDesc,
											1,
											&pgOptions,
											log,
											&sizeof_log,
											&hitgroupPGs[0]
										  ));
		if (sizeof_log > 1) PRINT(log);
	}

	// Assembles the full pipeline of all programs
	void Renderer::createPipeline()
	{
		std::vector<OptixProgramGroup> programGroups;
		for (auto pg : raygenPGs)
			programGroups.push_back(pg);
		for (auto pg : missPGs)
			programGroups.push_back(pg);
		for (auto pg : hitgroupPGs)
			programGroups.push_back(pg);

		char log[2048];
		size_t sizeof_log = sizeof( log );
		OPTIX_CHECK(optixPipelineCreate(optixContext,
								  		&pipelineCompileOptions,
								  		&pipelineLinkOptions,
								  		programGroups.data(),
								  		(int)programGroups.size(),
								  		log,
								  		&sizeof_log,
								  		&pipeline
									  ));
		if (sizeof_log > 1) PRINT(log);

		OPTIX_CHECK(optixPipelineSetStackSize(
										      // The pipeline to configure the stack size for
										      pipeline,
										      /**
										       * The direct stack size requirement for direct
										       * callables invoked from IS or AH
										       */
										      2 * 1024,
										      /**
										       * The direct stack size requirement for direct
										       * callables invoked from RG, MS or CH
										       */
										      2 * 1024,
										      // The continuation stack requirement
										      2 * 1024,
										      /**
										       * The maximum depth of a traversable graph passed
										       * to a trace
										       */
										      1
										    ));
		if (sizeof_log > 1) PRINT(log);
	}

	// Constructs the shader binding table
	void Renderer::buildSBT()
	{
		// Build raygen records
		std::vector<RaygenRecord> raygenRecords;

		for (int i = 0; i < raygenPGs.size(); i++)
		{
			RaygenRecord rec;
			OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
			rec.data = nullptr;
			raygenRecords.push_back(rec);
		}
		raygenRecordsBuffer.alloc_and_upload(raygenRecords);
		sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

		// Build miss records
		std::vector<MissRecord> missRecords;

		for (int i = 0; i < missPGs.size(); i++)
		{
			MissRecord rec;
			OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
			rec.data = nullptr;
			missRecords.push_back(rec);
		}
		missRecordsBuffer.alloc_and_upload(missRecords);
		sbt.missRecordBase				= missRecordsBuffer.d_pointer();
		sbt.missRecordStrideInBytes		= sizeof(MissRecord);
		sbt.missRecordCount				= (int)missRecords.size();

		// Build hitgroup records
		int numObjects = 1;

		std::vector<HitgroupRecord> hitgroupRecords;

		for (int i = 0; i < numObjects; i++)
		{
			int objectType = 0;
			HitgroupRecord rec;
			OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[objectType], &rec));
			rec.objectID = i;
			hitgroupRecords.push_back(rec);
		}
		hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
		sbt.hitgroupRecordBase				= hitgroupRecordsBuffer.d_pointer();
		sbt.hitgroupRecordStrideInBytes		= sizeof(HitgroupRecord);
		sbt.hitgroupRecordCount				= (int)hitgroupRecords.size();
	}

	// Render one frame
	void Renderer::render()
	{
		// Sanity check - make sure we launch only after first resize is done
		if (launchParams.fbSize.x == 0) return;

		launchParamsBuffer.upload(&launchParams, 1);
		launchParams.frameID++;

		OPTIX_CHECK(optixLaunch(
								// Pipeline we're launching
								pipeline,
								cudaStream,
								// Parameters and SBT
								launchParamsBuffer.d_pointer(),
								launchParamsBuffer.sizeInBytes,
								&sbt,
								// Dimensions of the launch
								launchParams.fbSize.x,
								launchParams.fbSize.y,
								1
							  ));

		/**
		 * Sync - make sure the frame is rendered before we download
		 * and display. For a high-performance application you want
		 * to use streams and double-buffering.
		 */
		CUDA_SYNC_CHECK();
	}

	// Resize frame buffer to given resolution
	void Renderer::resize(const vec2i& newSize)
	{
		// If window is minimized
		if (newSize.x == 0 | newSize.y == 0) return;

		// Resize our cuda frame buffer
		colorBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));

		// Update the launch parameters that we'll pass to the optix launch
		launchParams.fbSize			= newSize;
		launchParams.colorBuffer	= (uint32_t*)colorBuffer.d_ptr;
	}

	// Download the rendered color buffer
	void Renderer::downloadPixels(uint32_t h_pixels[])
	{
		colorBuffer.download(h_pixels, launchParams.fbSize.x * launchParams.fbSize.y);
	}
}