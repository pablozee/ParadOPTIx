#pragma once
#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "Model.h"

using namespace ParadOPTIx;

namespace ParadOPTIx {

	struct Camera 
	{
		vec3f from;
		vec3f at;
		vec3f up;
	};

	/**
	 * A sample OptiX-7 renderer that demonstrates how to up
	 * context, module, programs, pipeline, SBT, etc, and perform a
	 * valid launch that renders some pixels (using a simple test
	 * pattern)
	 */
	class Renderer
	{
		// Publicly accessible interface
	public:
		/**
		 * Constructor - performs all setup, including initializing
		 * optix, creates module, pipeline, programs, SBT, etc
		 */
		Renderer(const Model *model);

		// Render one frame
		void render();

		// Resize frame buffer to given resolution
		void resize(const vec2i& newSize);

		// Download the rendered color buffer
		void downloadPixels(uint32_t h_pixels[]);

		// Set camera to render with
		void setCamera(const Camera& camera);

	protected:
		// internal helper functions

		// Helper function that initializes optix and checks for errors
		void initOptix();

		// Creates and configures an optix device context for the primary GPU device
		void createContext();

		/**
		 * Creates the module that contains all the programs we are going to use.
		 * The module is created from a .cu file using an embedded ptx (parallel thread
		 * exchange) string
		 */
		void createModule();

		// Does all the setup for the ray programs we are going to use
		void createRaygenPrograms();

		// Does all the setup for the miss programs we are going to use
		void createMissPrograms();

		// Does all the setup for the hitgroup programs we are going to use
		void createHitgroupPrograms();

		// Assembles the full pipeline of all programs
		void createPipeline();

		// Constructs the shader binding table
		void buildSBT();

		// Build an acceleration structure for the given triangle mesh
		OptixTraversableHandle buildAccel();

		/**
		 * CUDA device context and stream that optix pipeline will run on,
		 * as well as device properties for this device
		 */
		CUcontext			cudaContext;
		CUstream			cudaStream;
		cudaDeviceProp		deviceProps;

		// Optix context that our pipeline will run in
		OptixDeviceContext  optixContext;

		// The pipeline we're building
		OptixPipeline					pipeline;
		OptixPipelineCompileOptions		pipelineCompileOptions = {};
		OptixPipelineLinkOptions		pipelineLinkOptions = {};

		// The module that contains our device programs
		OptixModule						module;
		OptixModuleCompileOptions		moduleCompileOptions = {};

		/**
		 * Vector for each program group, buffer for records
		 * and SBT built around them
		 */
		std::vector<OptixProgramGroup> raygenPGs;
		std::vector<OptixProgramGroup> missPGs;
		std::vector<OptixProgramGroup> hitgroupPGs;
		CUDABuffer					   raygenRecordsBuffer;
		CUDABuffer					   missRecordsBuffer;
		CUDABuffer					   hitgroupRecordsBuffer;

		OptixShaderBindingTable		   sbt = {};

		/**
		 * Our launch parameters, on the host, and the buffer to
		 * store them on the device
		 */
		LaunchParams	launchParams;
		CUDABuffer		launchParamsBuffer;

		CUDABuffer		colorBuffer;

		// The camera we are to render with
		Camera lastSetCamera;

		// The model we are going to trace rays against
		const Model* model;
		
		// One buffer per input mesh
		std::vector <CUDABuffer> vertexBuffer;
		std::vector <CUDABuffer> indexBuffer;
		// Buffer that keeps the final, compacted acceleration structure
		CUDABuffer asBuffer;
	};
}