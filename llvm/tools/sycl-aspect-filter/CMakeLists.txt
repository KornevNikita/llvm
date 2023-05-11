set(LLVM_LINK_COMPONENTS
  BinaryFormat
  Core
  Support
  SPIRVLib
  )

add_llvm_tool(sycl-aspect-filter
  sycl-aspect-filter.cpp
  )
