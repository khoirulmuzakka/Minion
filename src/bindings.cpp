#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // Include for std::vector support
#include <pybind11/functional.h>
#include "minimizer_base.h" // Include your MinionResult struct definition
#include "m_ljade_amr.h"
#include "m_lshade_amr.h"
#include "utility.h"

namespace py = pybind11;

PYBIND11_MODULE(pyminioncpp, m) {
    py::class_<MinionResult>(m, "MinionResult")
        .def(py::init<>()) // Default constructor
        .def(py::init<const std::vector<double>&, double, int, int, bool, const std::string&>()) // Parameterized constructor
        .def_readwrite("x", &MinionResult::x)
        .def_readwrite("fun", &MinionResult::fun)
        .def_readwrite("nit", &MinionResult::nit)
        .def_readwrite("nfev", &MinionResult::nfev)
        .def_readwrite("success", &MinionResult::success)
        .def_readwrite("message", &MinionResult::message);

    py::class_<MinimizerBase>(m, "MinimizerBase")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      const std::vector<double>&, void*, std::function<void(MinionResult*)>,
                      double, int, std::string, int>())
        .def_readwrite("callback", &MinimizerBase::callback)
        .def_readwrite("history", &MinimizerBase::history)
        .def("optimize", &MinimizerBase::optimize);

    py::class_<DE_Base, MinimizerBase>(m, "DE_Base")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      void*, const std::vector<double>&, int, int, std::string,
                      double, int, std::function<void(MinionResult*)>, std::string, int>())
        .def("getMaxIter", &DE_Base::getMaxIter)
        .def("_initialize_population", &DE_Base::_initialize_population)
        .def("_mutate", &DE_Base::_mutate)
        .def("_crossover_bin", &DE_Base::_crossover_bin)
        .def("_crossover_exp", &DE_Base::_crossover_exp)
        .def("_crossover", &DE_Base::_crossover)
        .def("optimize", &DE_Base::optimize);

    py::class_<M_LJADE_AMR, DE_Base>(m, "M_LJADE_AMR")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&, void*, const std::vector<double>&, int, int, std::string, double, int, double, std::function<void(MinionResult*)>, std::string, int>(),
             py::arg("func"),
             py::arg("bounds"),
             py::arg("data") = nullptr,
             py::arg("x0") = std::vector<double>{},
             py::arg("population_size") = 30,
             py::arg("maxevals") = 100000,
             py::arg("strategy") = "current_to_pbest1bin",
             py::arg("relTol") = 0.0001,
             py::arg("minPopSize") = 10,
             py::arg("c") = 0.5,
             py::arg("callback") = nullptr,
             py::arg("boundStrategy") = "reflect-random",
             py::arg("seed") = -1)
        .def("optimize", &M_LJADE_AMR::optimize)
        .def("_adapt_parameters", &M_LJADE_AMR::_adapt_parameters)
        .def_readwrite("meanCR", &M_LJADE_AMR::meanCR)
        .def_readwrite("meanF", &M_LJADE_AMR::meanF)
        .def_readwrite("c", &M_LJADE_AMR::c)
        .def_readwrite("history", &M_LJADE_AMR::history)
        .def_readwrite("muCR", &M_LJADE_AMR::muCR)
        .def_readwrite("muF", &M_LJADE_AMR::muF)
        .def_readwrite("stdCR", &M_LJADE_AMR::stdCR)
        .def_readwrite("stdF", &M_LJADE_AMR::stdF);

    py::class_<M_LSHADE_AMR, DE_Base>(m, "M_LSHADE_AMR")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&, void*, const std::vector<double>&, int, int, std::string, double, int, double, std::function<void(MinionResult*)>, std::string, int>(),
             py::arg("func"),
             py::arg("bounds"),
             py::arg("data") = nullptr,
             py::arg("x0") = std::vector<double>{},
             py::arg("population_size") = 30,
             py::arg("maxevals") = 100000,
             py::arg("strategy") = "current_to_pbest1bin",
             py::arg("relTol") = 0.0001,
             py::arg("minPopSize") = 10,
             py::arg("memorySize") = 30,
             py::arg("callback") = nullptr,
             py::arg("boundStrategy") = "reflect-random",
             py::arg("seed") = -1)
        .def("optimize", &M_LSHADE_AMR::optimize)
        .def("_adapt_parameters", &M_LSHADE_AMR::_adapt_parameters)
        .def_readwrite("M_CR", &M_LSHADE_AMR::M_CR)
        .def_readwrite("M_F", &M_LSHADE_AMR::M_F)
        .def_readwrite("memorySize", &M_LSHADE_AMR::memorySize)
        .def_readwrite("history", &M_LSHADE_AMR::history)
        .def_readwrite("muCR", &M_LSHADE_AMR::muCR)
        .def_readwrite("muF", &M_LSHADE_AMR::muF)
        .def_readwrite("stdCR", &M_LSHADE_AMR::stdCR)
        .def_readwrite("stdF", &M_LSHADE_AMR::stdF);

}
