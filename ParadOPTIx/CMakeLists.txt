include_directories(${OptiX_INCLUDE})

cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

add_executable(ParadOPTIx
  ${embedded_ptx_code}
  optix7.h
  CUDABuffer.h
  Renderer.h
  Renderer.cpp
  main.cpp
  )
target_link_libraries(ParadOPTIx
  gdt
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  )
