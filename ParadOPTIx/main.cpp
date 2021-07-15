#include "Renderer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/3rdParty/stb_image_write.h"

namespace ParadOPTIx {

	// main entry point - initialize optix, print hello world, then exit
	extern "C" int main(int ac, char** av)
	{
		try {
			Renderer renderer;

			const vec2i fbSize(vec2i(1200, 1024));
			renderer.resize(fbSize);
			renderer.render();

			std::vector<uint32_t> pixels(fbSize.x * fbSize.y);
			renderer.downloadPixels(pixels.data());

			const std::string fileName = "ParadOPTIx_example2.png";
			stbi_write_png(fileName.c_str(), fbSize.x, fbSize.y, 4,
				pixels.data(), fbSize.x * sizeof(uint32_t));

			std::cout << GDT_TERMINAL_GREEN
					  << std::endl
					  << "Image rendered, and saved to " << fileName << " ... done." << std::endl
					  << GDT_TERMINAL_DEFAULT 
					  << std::endl;
		}
		catch (std::runtime_error& e) {
			std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
					  << GDT_TERMINAL_DEFAULT << std::endl;
			exit( 1 );
		}
		return 0;
	}

}
