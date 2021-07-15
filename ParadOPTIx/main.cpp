#include "Renderer.h"
#include "include/gdt/gdt/gdt.h"

#include "glfWindow/GLFWindow.h"
#include "GL/GL.h"

namespace ParadOPTIx {

	struct RendererWindow : public GLFWindow
	{
		RendererWindow(const std::string& title)
			:
			GLFWindow(title)
		{

		}

		virtual void render() override 
		{
			renderer.render();
		}

		virtual void draw() override
		{
			renderer.downloadPixels(pixels.data());

			if (fbTexture == 0)
				glGenTextures(1, &fbTexture);

			glBindTexture(GL_TEXTURE_2D, fbTexture);
			GLenum texFormat = GL_RGBA;
			GLenum texelType = GL_UNSIGNED_BYTE;
			glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, 
						 GL_RGBA, texelType, pixels.data());

			glDisable(GL_LIGHTING);
			glColor3f(1, 1, 1);

			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();

			glEnable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D, fbTexture);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

			glDisable(GL_DEPTH_TEST);

			glViewport(0, 0, fbSize.x, fbSize.y);

			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

			glBegin(GL_QUADS);
			{
				glTexCoord2f(0.f, 0.f);
				glVertex3f(0.f, 0.f, 0.f);

				glTexCoord2f(0.f, 1.f);
				glVertex3f(0.f, (float)fbSize.y, 0.f);

				glTexCoord2f(1.f, 1.f);
				glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

				glTexCoord2f(1.f, 0.f);
				glVertex3f((float)fbSize.x, 0.f, 0.f);
			}
			glEnd();
		}

		virtual void resize(const vec2i& newSize)
		{
			fbSize = newSize;
			renderer.resize(newSize);
			pixels.resize(newSize.x * newSize.y);
		}

		vec2i					fbSize;
		GLuint					fbTexture{ 0 };
		Renderer				renderer;
		std::vector<uint32_t>	pixels;
	};

	// main entry point
	extern "C" int main(int ac, char** av)
	{
		try {
			RendererWindow* window = new RendererWindow("ParadOPTIx");
			window->run();
		}
		catch (std::runtime_error& e) {
			std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
					  << GDT_TERMINAL_DEFAULT << std::endl;
			exit( 1 );
		}
		return 0;
	}

}
