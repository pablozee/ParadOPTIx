#include "include/gdt/gdt/gdt.h"
#include "optix7.h"

namespace ParadOPTIx {

	// Helper function that initializes optix and checks for errors
	void initOptix()
	{

		// Check for available optix7 capable devices

		cudaFree(0);

		int numDevices;
		cudaGetDeviceCount(&numDevices);

		if (numDevices == 0)
			throw std::runtime_error("#ParadOPTIx: no CUDA capable devices found!");

		std::cout << "#ParadOPTIx: found " << numDevices << " CUDA devices" << std::endl;

		// Initialize Optix
		OPTIX_CHECK( optixInit() );
	}

	// main entry point - initialize optix, print hello world, then exit
	extern "C" int main(int ac, char** av)
	{
		try {
			std::cout << "#ParadOPTIx: initializing Optix..." << std::endl;

			initOptix();

			std::cout << GDT_TERMINAL_GREEN
					  << "#ParadOPTIx: Successfully initialized Optix"
					  << GDT_TERMINAL_DEFAULT << std::endl;

			// For this simple hello world example, don't do anything else
			std::cout << "#ParadOPTIx: Done. Clean exit" << std::endl;
		}
		catch (std::runtime_error& e) {
			std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
					  << GDT_TERMINAL_DEFAULT << std::endl;
			exit( 1 );
		}
		return 0;
	}

}
