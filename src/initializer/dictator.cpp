
#include "cppdict/include/dict.hpp"
#include "initializer/python_data_provider.h"

#include "python3/pybind_def.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


namespace py = pybind11;

using PHARE::initializer::ScalarFunction;
using PHARE::initializer::VectorFunction;


template<typename T>
void add(std::string path, T&& value)
{
    cppdict::add(path, std::forward<T>(value),
                 PHARE::initializer::PHAREDictHandler::INSTANCE().dict());
}


/* This function exists as the above "add" has issues differentiating between int and std::size_t
   for input
*/
void add_size_t(std::string path, std::size_t&& value)
{
    cppdict::add(path, std::forward<std::size_t>(value),
                 PHARE::initializer::PHAREDictHandler::INSTANCE().dict());
}

void add_optional_size_t(std::string path, std::optional<std::size_t>&& value)
{
    cppdict::add(path, std::forward<std::optional<std::size_t>>(value),
                 PHARE::initializer::PHAREDictHandler::INSTANCE().dict());
}

template<typename T>
void add_array_as_vector(std::string path, PHARE::pydata::py_array_t<T>& array)
{
    auto buf = array.request();

    if (buf.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");

    T* ptr = static_cast<T*>(buf.ptr);

    cppdict::add(path, std::vector<T>(ptr, ptr + buf.size),
                 PHARE::initializer::PHAREDictHandler::INSTANCE().dict());
}


PYBIND11_MODULE(dictator, m)
{
    m.def("add_size_t", add_size_t, "add_size_t");
    m.def("add_optional_size_t", add_optional_size_t, "add_optional_size_t");

    m.def("add", add<int>, "add");
    m.def("add", add<double>, "add");
    m.def("add", add<std::string>, "add");
    m.def("addScalarFunction1D", add<ScalarFunction<1>>, "add");
    m.def("addScalarFunction2D", add<ScalarFunction<2>>, "add");
    m.def("addScalarFunction3D", add<ScalarFunction<3>>, "add");

    m.def("addPyArrayFunction1D", add<VectorFunction<1>>, "add");
    m.def("addPyArrayFunction2D", add<VectorFunction<2>>, "add");
    m.def("addPyArrayFunction3D", add<VectorFunction<3>>, "add");

    m.def("add_array_as_vector", add_array_as_vector<double>, "add_array_as_vector");
}
