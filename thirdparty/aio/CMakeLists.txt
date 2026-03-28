if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i686|i386|x64")
  set(aio_INCLUDE_DIR "${EXTERNAL_BINARY_DIR}/usr/local/include")
  set(aio_LIBRARY_DIR "${EXTERNAL_BINARY_DIR}/usr/local/lib/")
  file(MAKE_DIRECTORY ${aio_INCLUDE_DIR})
  file(MAKE_DIRECTORY ${aio_LIBRARY_DIR})
  
  
  include(ExternalProject)
  include(ProcessorCount)
  
  ProcessorCount(NPROC)
  
  ExternalProject_Add(aio.BUILD
      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libaio-0.3/src
      BUILD_IN_SOURCE ON
      CONFIGURE_COMMAND ""
      BUILD_COMMAND make -j${NPROC}
      INSTALL_COMMAND make install prefix=${EXTERNAL_BINARY_DIR}/usr/local
      BUILD_BYPRODUCTS ${aio_LIBRARY_DIR}/libaio.a
  )
  
  add_custom_command(
      OUTPUT ${EXTERNAL_LIB_DIR}/libaio.a
             ${EXTERNAL_INC_DIR}/libaio.h
      DEPENDS libaio
      COMMAND ${CMAKE_COMMAND} -E copy
              ${CMAKE_CURRENT_SOURCE_DIR}/libaio-0.3/src/libaio.a
              ${EXTERNAL_LIB_DIR}/libaio.a
      COMMAND ${CMAKE_COMMAND} -E copy
              ${CMAKE_CURRENT_SOURCE_DIR}/libaio-0.3/src/libaio.h
              ${EXTERNAL_INC_DIR}/libaio.h
      COMMENT "Copying libaio static library and header"
  )
  
  add_library(aio STATIC IMPORTED GLOBAL)
  add_dependencies(aio aio.BUILD)
  
  set(aio_FOUND TRUE PARENT_SCOPE)
  set(aio_LIBRARY ${aio_LIBRARY_DIR}/libaio.a PARENT_SCOPE)
  set(aio_LIBRARIES ${aio_LIBRARY_DIR}/libaio.a PARENT_SCOPE)
  set(aio_INCLUDE_DIR "${EXTERNAL_BINARY_DIR}/usr/local/include" PARENT_SCOPE)
  set(aio_INCLUDE_DIRS "${EXTERNAL_BINARY_DIR}/usr/local/include" PARENT_SCOPE)
  set(aio_VERSION 0.3 PARENT_SCOPE)
endif()
