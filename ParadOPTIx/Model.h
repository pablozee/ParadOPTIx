#pragma once
#include "gdt/math/AffineSpace.h"
#include <vector>

namespace ParadOPTIx {
	using namespace gdt;

	// A simple indexed triangle mesh that our sample renderer will 
	// render
	struct TriangleMesh {
		std::vector<vec3f> vertex;
		std::vector<vec3f> normal;
		std::vector<vec2f> texcoord;
		std::vector<vec3i> index;

		// material data
		vec3f			   diffuse;
	};

	struct Model {
		~Model()
		{
			for (auto mesh : meshes) delete mesh;
		}

		std::vector<TriangleMesh*> meshes;
		
		// Bounding box of all vertices in the model
		box3f bounds;
	};

	Model* loadOBJ(const std::string& objFile);
}