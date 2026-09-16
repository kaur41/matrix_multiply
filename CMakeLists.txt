cmake_minimum_required(VERSION 3.25)
project(KaurSparseMatrixParallel CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Максимальная оптимизация кода для Visual Studio в конфигурации Release
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Oi /Ot /Gy")
endif()

# =========================================================================
#  1. ПОДКЛЮЧЕНИЕ ГОТОВЫХ БИНАРНИКОВ TBB С ДИСКА D
# =========================================================================
find_package(OpenMP REQUIRED)

# Указываем компилятору, где искать заголовочные файлы (.hpp)
include_directories("D:/oneapi-tbb-2023.1.0/include")

# Указываем линкеру точный путь, где лежит tbb.lib
link_directories("D:/oneapi-tbb-2023.1.0/lib/intel64/vc14")

# =========================================================================
#  2. СБОРКА ИСПОЛНЯЕМОГО ФАЙЛА ПРОЕКТА + ВСТРОЕННЫЙ GOOGLE TEST
# =========================================================================
include_directories(common/include)
include_directories(ext/googletest)
include_directories(ext/googletest/include)

add_executable(matrix_multiply_project
    all/all.cpp
    seq/seq_task.cpp
    omp/omp_task.cpp
    tbb/tbb_task.cpp
    tests/main.cpp
    ext/googletest/src/gtest-all.cc
)

# Линкуем OpenMP и tbb.lib напрямую по имени пакета
target_link_libraries(matrix_multiply_project PRIVATE 
    OpenMP::OpenMP_CXX 
    tbb
)

enable_testing()
add_test(NAME SparseMatrixAllTests COMMAND matrix_multiply_project)
