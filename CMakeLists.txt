cmake_minimum_required(VERSION 3.25)
project(KaurSparseMatrixParallel CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Жестко включаем максимальную оптимизацию компилятора Visual Studio для Release
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Oi /Ot /Gy")

# =========================================================================
#  1. ПОДКЛЮЧЕНИЕ ГОТОВЫХ ЛОКАЛЬНЫХ БИБЛИОТЕК (oneTBB 2023.1.0)
# =========================================================================
find_package(OpenMP REQUIRED)

# Прописываем точные пути к заголовочным файлам и либам Intel TBB на диске D
include_directories("D:/oneapi-tbb-2023.1.0/include")
link_directories("D:/oneapi-tbb-2023.1.0/lib/intel64/vc14")

# =========================================================================
#  2. СБОРКА ИСПОЛНЯЕМОГО ФАЙЛА ПРОЕКТА + ЛОКАЛЬНЫЙ GOOGLE TEST
# =========================================================================
# Указываем пути к инклудам лабы и встроенного нативного gtest
include_directories(common/include)
include_directories(ext/googletest)
include_directories(ext/googletest/include)

add_executable(matrix_multiply_project
    all/all.cpp
    seq/seq_task.cpp
    omp/omp_task.cpp
    tbb/tbb_task.cpp
    
    # Два каноничных раздельных файла тестов (Functional и Performance)
    tests/functional_tests.cpp
    tests/performance_tests.cpp
    
    # Нативно компилируем Google Test вместе с проектом
    ext/googletest/src/gtest-all.cc
)

# Линкуем проект с OpenMP и tbb.lib (gtest слинкуется нативно)
target_link_libraries(matrix_multiply_project PRIVATE 
    OpenMP::OpenMP_CXX 
    tbb
)

enable_testing()
add_test(NAME SparseMatrixAllTests COMMAND matrix_multiply_project)
