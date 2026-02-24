cmake_minimum_required(VERSION 3.20)
project(pdekernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# NumPy include path: robust method via Python
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NUMPY_INCLUDE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

pybind11_add_module(_core
    cpp/bindings.cpp
)

target_include_directories(_core PRIVATE ${NUMPY_INCLUDE})

# A couple of safe performance flags (don’t go too wild here)
if (APPLE)
  target_compile_options(
      _core PRIVATE
      -Ofast -mcpu=native
      -Xpreprocessor -fopenmp
      -I/opt/homebrew/Cellar/libomp/21.1.8/include
    )
  target_link_libraries(_core PRIVATE /opt/homebrew/Cellar/libomp/21.1.8/lib/libomp.dylib)
else()
  target_compile_options(
  _core PRIVATE
  -Ofast -mcpu=native
  -fopenmp
)
endif()
install(TARGETS _core
        LIBRARY DESTINATION ader_dg_transport
        ARCHIVE DESTINATION ader_dg_transport
        RUNTIME DESTINATION ader_dg_transport)