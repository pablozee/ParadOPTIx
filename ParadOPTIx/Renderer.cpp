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
}