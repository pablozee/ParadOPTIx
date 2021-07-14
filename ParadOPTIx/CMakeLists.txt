include_directories(${OptiX_INCLUDE})

add_executable(ParadOPTIx
  optix7.h
  main.cpp
  )
target_link_libraries(ParadOPTIx
  gdt
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  )
