#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // Include for std::vector support
#include <pybind11/functional.h>
#include "minimizer_base.h" // Include your MinionResult struct definition
#include "mfade.h"
#include "ebr_lshade.h"
#include "gwo_de.h"
#include "powell.h"
#include "nelder_mead.h"
#include "utility.h"
#include "cec2020.h"
#include "cec2022.h"
#include "lshade.h"
#include <exception>

namespace py = pybind11;

// Exception translator
void translate_exception(const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

PYBIND11_MODULE(pyminioncpp, m) {
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const std::exception &e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

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
                      double, size_t, std::string, int>())
        .def_readwrite("callback", &MinimizerBase::callback)
        .def_readwrite("history", &MinimizerBase::history)
        .def("optimize", &MinimizerBase::optimize);

    py::class_<DE_Base, MinimizerBase>(m, "DE_Base")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      void*, const std::vector<double>&, size_t, size_t, std::string,
                      double, size_t, std::function<void(MinionResult*)>, std::string, int>())
        .def("optimize", &DE_Base::optimize);
    
    py::class_<MFADE, DE_Base>(m, "MFADE")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&, void*, const std::vector<double>&, int, int, std::string, double, int, double, std::function<void(MinionResult*)>, std::string, int>(),
             py::arg("func"),
             py::arg("bounds"),
             py::arg("data") = nullptr,
             py::arg("x0") = std::vector<double>{},
             py::arg("population_size") = 30,
             py::arg("maxevals") = 100000,
             py::arg("strategy") = "current_to_pbest1bin",
             py::arg("relTol_firstRun") = 0.001,
             py::arg("minPopSize") = 10,
             py::arg("memorySize") = 30,
             py::arg("callback") = nullptr,
             py::arg("boundStrategy") = "reflect-random",
             py::arg("seed") = -1)
        .def("optimize", &MFADE::optimize)
        .def_readwrite("history", &MFADE::history)
        .def_readwrite("muCR", &MFADE::muCR)
        .def_readwrite("muF", &MFADE::muF)
        .def_readwrite("stdCR", &MFADE::stdCR)
        .def_readwrite("stdF", &MFADE::stdF);

    py::class_<EBR_LSHADE, DE_Base>(m, "EBR_LSHADE")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      void*, const std::vector<double>&, size_t, int, double, size_t, size_t, 
                      std::function<void(MinionResult*)>, size_t, double, std::string, int>(),
             py::arg("func"),
             py::arg("bounds"),
             py::arg("data") = nullptr,
             py::arg("x0") = std::vector<double>{},
             py::arg("population_size") = 30,
             py::arg("maxevals") = 100000,
             py::arg("relTol_firstRun") = 0.01,
             py::arg("minPopSize") = 5,
             py::arg("memorySize") = 50,
             py::arg("callback") = nullptr,
             py::arg("max_restarts") = 0,
             py::arg("startRefine") = 0.75,
             py::arg("boundStrategy") = "reflect-random",
             py::arg("seed") = -1)
        .def_readwrite("muCR", &EBR_LSHADE::muCR)
        .def_readwrite("muF", &EBR_LSHADE::muF)
        .def_readwrite("stdCR", &EBR_LSHADE::stdCR)
        .def_readwrite("stdF", &EBR_LSHADE::stdF)
        .def_readwrite("history", &EBR_LSHADE::history)
        .def("optimize", &EBR_LSHADE::optimize);

    py::class_<LSHADE, DE_Base>(m, "LSHADE")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&, void*, const std::vector<double>&, size_t, size_t, std::string, double, size_t, double, std::function<void(MinionResult*)>, std::string, int>(),
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
        .def("optimize", &LSHADE::optimize)
        .def_readwrite("history", &LSHADE::history)
        .def_readwrite("muCR", &LSHADE::muCR)
        .def_readwrite("muF", &LSHADE::muF)
        .def_readwrite("stdCR", &LSHADE::stdCR)
        .def_readwrite("stdF", &LSHADE::stdF);

    py::class_<GWO_DE, MinimizerBase>(m, "GWO_DE")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&, const std::vector<double>&, size_t, int, double, double, double, double, std::string, int, void*, std::function<void(MinionResult*)>>(),
             py::arg("func"),
             py::arg("bounds"),
             py::arg("x0") = std::vector<double>{},
             py::arg("population_size") = 30,
             py::arg("maxevals") = 100000,
             py::arg("F") = 0.5,
             py::arg("CR") = 0.7,
             py::arg("elimination_prob") = 0.1,
             py::arg("relTol") = 0.0001,
             py::arg("boundStrategy") = "reflect-random",
             py::arg("seed") = -1,
             py::arg("data") = nullptr,
             py::arg("callback") = nullptr)
        .def("optimize", &GWO_DE::optimize)
        .def_readwrite("alpha_score", &GWO_DE::alpha_score)
        .def_readwrite("beta_score", &GWO_DE::beta_score)
        .def_readwrite("delta_score", &GWO_DE::delta_score)
        .def_readwrite("alpha_pos", &GWO_DE::alpha_pos)
        .def_readwrite("beta_pos", &GWO_DE::beta_pos)
        .def_readwrite("delta_pos", &GWO_DE::delta_pos)
        .def_readwrite("population", &GWO_DE::population)
        .def_readwrite("fitness", &GWO_DE::fitness)
        .def_readwrite("eval_count", &GWO_DE::eval_count);

    py::class_<Powell, MinimizerBase>(m, "Powell")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      const std::vector<double>&, void*, std::function<void(MinionResult*)>,
                      double, int, std::string, int>(),
             py::arg("func"),
             py::arg("bounds"),
             py::arg("x0") = std::vector<double>{},
             py::arg("data") = nullptr,
             py::arg("callback") = nullptr,
             py::arg("relTol") = 0.0001,
             py::arg("maxevals") = 100000,
             py::arg("boundStrategy") = "reflect-random",
             py::arg("seed") = -1)
        .def("optimize", &Powell::optimize);

    py::class_<NelderMead, MinimizerBase>(m, "NelderMead")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      const std::vector<double>&, void*, std::function<void(MinionResult*)>,
                      double, int, std::string, int>(),
             py::arg("func"),
             py::arg("bounds"),
             py::arg("x0") = std::vector<double>{},
             py::arg("data") = nullptr,
             py::arg("callback") = nullptr,
             py::arg("relTol") = 0.0001,
             py::arg("maxevals") = 100000,
             py::arg("boundStrategy") = "reflect-random",
             py::arg("seed") = -1)
        .def("optimize", &NelderMead::optimize);

    py::class_<CEC2020Functions>(m, "CEC2020Functions")
        .def(py::init<int, int>(), py::arg("function_number"), py::arg("dimension"))
        .def("__call__", &CEC2020Functions::operator());

    py::class_<CEC2022Functions>(m, "CEC2022Functions")
        .def(py::init<int, int>(), py::arg("function_number"), py::arg("dimension"))
        .def("__call__", &CEC2022Functions::operator());
}
