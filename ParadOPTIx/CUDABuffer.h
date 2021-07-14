#pragma once

#include "optix7.h"
#include <vector>
#include <assert.h>

namespace ParadOPTIx {

	// Simple wrapper for creating and managing a device-side CUDA buffer
	struct CUDABuffer {
		inline CUdeviceptr d_pointer() const
		{
			return (CUdeviceptr)d_ptr;
		}

		// Resize buffer to given number of bytes
		void resize(size_t size)
		{
			if (d_ptr) free();
			alloc(size);
		}

		// Allocate to given number of bytes
		void alloc(size_t size)
		{
			assert(d_ptr == nullptr);
			this->sizeInBytes = size;
			CUDA_CHECK(Malloc((void**) & d_ptr, sizeInBytes));
		}

		// Free allocated memory
		void free()
		{
			CUDA_CHECK(Free(d_ptr));
			d_ptr = nullptr;
			sizeInBytes = 0;
		}

		template<typename T>
		void upload(const T* t, size_t count)
		{
			assert(d_ptr != nullptr);
			assert(sizeInBytes == count * sizeof(T));
			CUDA_CHECK(Memcpy(d_ptr, (void*)t,
							  count * sizeof(T), cudaMemcpyHostToDevice));
		}

		template<typename T>
		void download(T* t, size_t count)
		{
			assert(d_ptr != nullptr);
			assert(sizeInBytes == count * sizeof(T));
			CUDA_CHECK(Memcpy((void*)t, d_ptr,
							  count * sizeof(T), cudaMemcpyDeviceToHost));
		}

		size_t sizeInBytes{ 0 };
		void* d_ptr{ nullptr };
	};
}
