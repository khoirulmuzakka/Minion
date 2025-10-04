#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // Include for std::vector support
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include "minimizer_base.h" // Include your MinionResult struct definition
#include "gwo_de.h"
#include "nelder_mead.h"
#include "utility.h"
#include "cec2017.h"
#include "cec2014.h"
#include "cec2019.h"
#include "cec2020.h"
#include "cec2022.h"
#include "lshade.h"
#include "jade.h"
#include "de.h"
#include "arrde.h"
#include "nlshadersp.h"
#include "j2020.h"
#include "lsrtde.h"
#include "jso.h"
#include "abc.h"
#include "pso.h"
#include "spso2011.h"
#include "dmspso.h"
#include "lshadecnepsin.h"
#include "dual_annealing.h"
#include "l_bfgs_b.h"
#include "minimizer.h"
#include "l_bfgs.h"
#include <exception>
#include <pybind11/stl_bind.h>
#include <any>
#include <map>
#include <string>
#include <vector>

namespace py = pybind11;

using namespace minion; 

// Exception translator
void translate_exception(const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
}


PYBIND11_MODULE(minionpycpp, m) {
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
                      const std::vector<std::vector<double>>&, void*, std::function<void(MinionResult*)>,
                      double, size_t, int, std::map<std::string, ConfigValue> >())
        .def_readwrite("callback", &MinimizerBase::callback)
        .def_readwrite("history", &MinimizerBase::history)
        // Release the GIL while the long-running C++ optimize routine executes; pybind11 reacquires it for Python callbacks.
        .def("optimize", &MinimizerBase::optimize, py::call_guard<py::gil_scoped_release>());

    py::class_<Differential_Evolution, MinimizerBase>(m, "Differential_Evolution")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      const std::vector<std::vector<double>>&, void*, std::function<void(MinionResult*)>,
                      double, size_t, int, std::map<std::string, ConfigValue> >(),
            py::arg("func"), 
            py::arg("bounds"), 
            py::arg("x0") = std::vector<std::vector<double>>(), 
            py::arg("data") = nullptr, 
            py::arg("callback") = nullptr, 
            py::arg("tol") = 0.0001, 
            py::arg("maxevals") = 100000, 
            py::arg("seed") = -1, 
            py::arg("options") = std::map<std::string, ConfigValue>())
        .def("optimize", &Differential_Evolution::optimize, py::call_guard<py::gil_scoped_release>())
        .def_readonly("meanCR", &Differential_Evolution::meanCR)
        .def_readonly("meanF", &Differential_Evolution::meanF)
        .def_readonly("stdCR", &Differential_Evolution::stdCR)
        .def_readonly("stdF", &Differential_Evolution::stdF)
        .def_readonly("diversity", &Differential_Evolution::diversity);

    py::class_<j2020, MinimizerBase>(m, "j2020")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      const std::vector<std::vector<double>>&, void*, std::function<void(MinionResult*)>,
                      double, size_t, int, std::map<std::string, ConfigValue> >(),
            py::arg("func"), 
            py::arg("bounds"), 
            py::arg("x0") = std::vector<std::vector<double>>(),
            py::arg("data") = nullptr, 
            py::arg("callback") = nullptr, 
            py::arg("tol") = 0.0001, 
            py::arg("maxevals") = 100000, 
            py::arg("seed") = -1, 
            py::arg("options") = std::map<std::string, ConfigValue>())

        .def("optimize", &j2020::optimize, py::call_guard<py::gil_scoped_release>());

    py::class_<LSRTDE, MinimizerBase>(m, "LSRTDE")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      const std::vector<std::vector<double>>&, void*, std::function<void(MinionResult*)>,
                      double, size_t, int, std::map<std::string, ConfigValue> >(),
            py::arg("func"), 
            py::arg("bounds"), 
            py::arg("x0") = std::vector<std::vector<double>>(),
            py::arg("data") = nullptr, 
            py::arg("callback") = nullptr, 
            py::arg("tol") = 0.0001, 
            py::arg("maxevals") = 100000, 
            py::arg("seed") = -1, 
            py::arg("options") = std::map<std::string, ConfigValue>())

        .def("optimize", &LSRTDE::optimize, py::call_guard<py::gil_scoped_release>());

    py::class_<LSHADE, Differential_Evolution>(m, "LSHADE")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      const std::vector<std::vector<double>>&, void*, std::function<void(MinionResult*)>,
                      double, size_t, int, std::map<std::string, ConfigValue> >(),
            py::arg("func"), 
            py::arg("bounds"), 
            py::arg("x0") = std::vector<std::vector<double>>(),
            py::arg("data") = nullptr, 
            py::arg("callback") = nullptr, 
            py::arg("tol") = 0.0001, 
            py::arg("maxevals") = 100000, 
            py::arg("seed") = -1, 
            py::arg("options") = std::map<std::string, ConfigValue>())

        .def("optimize", &LSHADE::optimize, py::call_guard<py::gil_scoped_release>())
        .def_readonly("meanCR", &LSHADE::meanCR)
        .def_readonly("meanF", &LSHADE::meanF)
        .def_readonly("stdCR", &LSHADE::stdCR)
        .def_readonly("stdF", &LSHADE::stdF)
        .def_readonly("diversity", &LSHADE::diversity);

    py::class_<LSHADE_cnEpSin, Differential_Evolution>(m, "LSHADE_cnEpSin")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      const std::vector<std::vector<double>>&, void*, std::function<void(MinionResult*)>,
                      double, size_t, int, std::map<std::string, ConfigValue> >(),
            py::arg("func"),
            py::arg("bounds"),
            py::arg("x0") = std::vector<std::vector<double>>(),
            py::arg("data") = nullptr,
            py::arg("callback") = nullptr,
            py::arg("tol") = 0.0001,
            py::arg("maxevals") = 100000,
            py::arg("seed") = -1,
            py::arg("options") = std::map<std::string, ConfigValue>())
        .def("optimize", &LSHADE_cnEpSin::optimize, py::call_guard<py::gil_scoped_release>())
        .def_readonly("meanCR", &LSHADE_cnEpSin::meanCR)
        .def_readonly("meanF", &LSHADE_cnEpSin::meanF)
        .def_readonly("stdCR", &LSHADE_cnEpSin::stdCR)
        .def_readonly("stdF", &LSHADE_cnEpSin::stdF)
        .def_readonly("diversity", &LSHADE_cnEpSin::diversity);

    py::class_<jSO, Differential_Evolution>(m, "jSO")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      const std::vector<std::vector<double>>&, void*, std::function<void(MinionResult*)>,
                      double, size_t, int, std::map<std::string, ConfigValue> >(),
            py::arg("func"), 
            py::arg("bounds"), 
            py::arg("x0") = std::vector<std::vector<double>>(),
            py::arg("data") = nullptr, 
            py::arg("callback") = nullptr, 
            py::arg("tol") = 0.0001, 
            py::arg("maxevals") = 100000, 
            py::arg("seed") = -1, 
            py::arg("options") = std::map<std::string, ConfigValue>())

        .def("optimize", &jSO::optimize, py::call_guard<py::gil_scoped_release>())
        .def_readonly("meanCR", &jSO::meanCR)
        .def_readonly("meanF", &jSO::meanF)
        .def_readonly("stdCR", &jSO::stdCR)
        .def_readonly("stdF", &jSO::stdF)
        .def_readonly("diversity", &jSO::diversity);

    py::class_<JADE, Differential_Evolution>(m, "JADE")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      const std::vector<std::vector<double>>&, void*, std::function<void(MinionResult*)>,
                      double, size_t, int, std::map<std::string, ConfigValue> >(),
            py::arg("func"), 
            py::arg("bounds"), 
            py::arg("x0") =std::vector<std::vector<double>>(),
            py::arg("data") = nullptr, 
            py::arg("callback") = nullptr, 
            py::arg("tol") = 0.0001, 
            py::arg("maxevals") = 100000, 
            py::arg("seed") = -1, 
            py::arg("options") = std::map<std::string, ConfigValue>())

        .def("optimize", &JADE::optimize, py::call_guard<py::gil_scoped_release>())
        .def_readonly("meanCR", &JADE::meanCR)
        .def_readonly("meanF", &JADE::meanF)
        .def_readonly("stdCR", &JADE::stdCR)
        .def_readonly("stdF", &JADE::stdF)
        .def_readonly("diversity", &JADE::diversity);

    py::class_<NLSHADE_RSP, MinimizerBase>(m, "NLSHADE_RSP")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      const std::vector<std::vector<double>>&, void*, std::function<void(MinionResult*)>,
                      double, size_t, int, std::map<std::string, ConfigValue> >(),
            py::arg("func"), 
            py::arg("bounds"), 
            py::arg("x0") = std::vector<std::vector<double>>(),
            py::arg("data") = nullptr, 
            py::arg("callback") = nullptr, 
            py::arg("tol") = 0.0001, 
            py::arg("maxevals") = 100000, 
            py::arg("seed") = -1, 
            py::arg("options") = std::map<std::string, ConfigValue>())

        .def_readwrite("history", &MinimizerBase::history)
        .def("optimize", &MinimizerBase::optimize, py::call_guard<py::gil_scoped_release>());

    py::class_<ABC, MinimizerBase>(m, "ABC")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      const std::vector<std::vector<double>>&, void*, std::function<void(MinionResult*)>,
                      double, size_t, int, std::map<std::string, ConfigValue> >(),
            py::arg("func"), 
            py::arg("bounds"), 
            py::arg("x0") = std::vector<std::vector<double>>(),
            py::arg("data") = nullptr, 
            py::arg("callback") = nullptr, 
            py::arg("tol") = 0.0001, 
            py::arg("maxevals") = 100000, 
            py::arg("seed") = -1, 
            py::arg("options") = std::map<std::string, ConfigValue>())

        .def_readwrite("history", &MinimizerBase::history)
        .def("optimize", &ABC::optimize, py::call_guard<py::gil_scoped_release>());

    py::class_<PSO, MinimizerBase>(m, "PSO")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      const std::vector<std::vector<double>>&, void*, std::function<void(MinionResult*)>,
                      double, size_t, int, std::map<std::string, ConfigValue> >(),
            py::arg("func"), 
            py::arg("bounds"), 
            py::arg("x0") = std::vector<std::vector<double>>(),
            py::arg("data") = nullptr, 
            py::arg("callback") = nullptr, 
            py::arg("tol") = 0.0001, 
            py::arg("maxevals") = 100000, 
            py::arg("seed") = -1, 
            py::arg("options") = std::map<std::string, ConfigValue>())

        .def("optimize", &PSO::optimize, py::call_guard<py::gil_scoped_release>())
        .def_readonly("diversity", &PSO::diversity)
        .def_readonly("spatialDiversity", &PSO::spatialDiversity);

    py::class_<SPSO2011, PSO>(m, "SPSO2011")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      const std::vector<std::vector<double>>&, void*, std::function<void(MinionResult*)>,
                      double, size_t, int, std::map<std::string, ConfigValue> >(),
            py::arg("func"), 
            py::arg("bounds"), 
            py::arg("x0") = std::vector<std::vector<double>>(),
            py::arg("data") = nullptr, 
            py::arg("callback") = nullptr, 
            py::arg("tol") = 0.0001, 
            py::arg("maxevals") = 100000, 
            py::arg("seed") = -1, 
            py::arg("options") = std::map<std::string, ConfigValue>())
        .def("optimize", &SPSO2011::optimize, py::call_guard<py::gil_scoped_release>());

    py::class_<DMSPSO, PSO>(m, "DMSPSO")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      const std::vector<std::vector<double>>&, void*, std::function<void(MinionResult*)>,
                      double, size_t, int, std::map<std::string, ConfigValue> >(),
            py::arg("func"), 
            py::arg("bounds"), 
            py::arg("x0") = std::vector<std::vector<double>>(),
            py::arg("data") = nullptr, 
            py::arg("callback") = nullptr, 
            py::arg("tol") = 0.0001, 
            py::arg("maxevals") = 100000, 
            py::arg("seed") = -1, 
            py::arg("options") = std::map<std::string, ConfigValue>())
        .def("optimize", &DMSPSO::optimize, py::call_guard<py::gil_scoped_release>());

    py::class_<Dual_Annealing, MinimizerBase>(m, "Dual_Annealing")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      const std::vector<std::vector<double>>&, void*, std::function<void(MinionResult*)>,
                      double, size_t, int, std::map<std::string, ConfigValue> >(),
            py::arg("func"), 
            py::arg("bounds"), 
            py::arg("x0") = std::vector<std::vector<double>>(),
            py::arg("data") = nullptr, 
            py::arg("callback") = nullptr, 
            py::arg("tol") = 0.0001, 
            py::arg("maxevals") = 100000, 
            py::arg("seed") = -1, 
            py::arg("options") = std::map<std::string, ConfigValue>())

        .def_readwrite("history", &MinimizerBase::history)
        .def("optimize", &Dual_Annealing::optimize, py::call_guard<py::gil_scoped_release>());

    py::class_<L_BFGS_B, MinimizerBase>(m, "L_BFGS_B")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      const std::vector<std::vector<double>>&, void*, std::function<void(MinionResult*)>,
                      double, size_t, int, std::map<std::string, ConfigValue> >(),
            py::arg("func"), 
            py::arg("bounds"), 
            py::arg("x0") = std::vector<std::vector<double>>(),
            py::arg("data") = nullptr, 
            py::arg("callback") = nullptr, 
            py::arg("tol") = 0.0001, 
            py::arg("maxevals") = 100000, 
            py::arg("seed") = -1, 
            py::arg("options") = std::map<std::string, ConfigValue>())

        .def_readwrite("history", &MinimizerBase::history)
        .def("optimize", &L_BFGS_B::optimize, py::call_guard<py::gil_scoped_release>());

    py::class_<L_BFGS, MinimizerBase>(m, "L_BFGS")
        .def(py::init<MinionFunction, const std::vector<std::vector<double>>&, void*, std::function<void(MinionResult*)>,
                      double, size_t, int, std::map<std::string, ConfigValue> >(),
            py::arg("func"), 
            py::arg("x0") = std::vector<std::vector<double>>(),
            py::arg("data") = nullptr, 
            py::arg("callback") = nullptr, 
            py::arg("tol") = 0.0001, 
            py::arg("maxevals") = 100000, 
            py::arg("seed") = -1, 
            py::arg("options") = std::map<std::string, ConfigValue>())

        .def_readwrite("history", &MinimizerBase::history)
        .def("optimize", &L_BFGS::optimize, py::call_guard<py::gil_scoped_release>());

    py::class_<ARRDE, Differential_Evolution>(m, "ARRDE")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      const std::vector<std::vector<double>>&, void*, std::function<void(MinionResult*)>,
                      double, size_t, int, std::map<std::string, ConfigValue> >(),
            py::arg("func"), 
            py::arg("bounds"), 
            py::arg("x0") = std::vector<std::vector<double>>(),
            py::arg("data") = nullptr, 
            py::arg("callback") = nullptr, 
            py::arg("tol") = 0.0001, 
            py::arg("maxevals") = 100000, 
            py::arg("seed") = -1, 
            py::arg("options") = std::map<std::string, ConfigValue>())

        .def("optimize", &ARRDE::optimize, py::call_guard<py::gil_scoped_release>())
        .def_readonly("meanCR", &ARRDE::meanCR)
        .def_readonly("meanF", &ARRDE::meanF)
        .def_readonly("stdCR", &ARRDE::stdCR)
        .def_readonly("stdF", &ARRDE::stdF)
        .def_readonly("diversity", &ARRDE::diversity);

    py::class_<GWO_DE, MinimizerBase>(m, "GWO_DE")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      const std::vector<std::vector<double>>&, void*, std::function<void(MinionResult*)>,
                      double, size_t, int, std::map<std::string, ConfigValue> >(),
            py::arg("func"), 
            py::arg("bounds"), 
            py::arg("x0") = std::vector<std::vector<double>>(),
            py::arg("data") = nullptr, 
            py::arg("callback") = nullptr, 
            py::arg("tol") = 0.0001, 
            py::arg("maxevals") = 100000, 
            py::arg("seed") = -1, 
            py::arg("options") = std::map<std::string, ConfigValue>())
            
        .def("optimize", &GWO_DE::optimize, py::call_guard<py::gil_scoped_release>())
        .def_readwrite("alpha_score", &GWO_DE::alpha_score)
        .def_readwrite("beta_score", &GWO_DE::beta_score)
        .def_readwrite("delta_score", &GWO_DE::delta_score)
        .def_readwrite("alpha_pos", &GWO_DE::alpha_pos)
        .def_readwrite("beta_pos", &GWO_DE::beta_pos)
        .def_readwrite("delta_pos", &GWO_DE::delta_pos)
        .def_readwrite("population", &GWO_DE::population)
        .def_readwrite("fitness", &GWO_DE::fitness)
        .def_readwrite("eval_count", &GWO_DE::eval_count);

    py::class_<NelderMead, MinimizerBase>(m, "NelderMead")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      const std::vector<std::vector<double>>&, void*, std::function<void(MinionResult*)>,
                      double, size_t, int, std::map<std::string, ConfigValue> >(),
            py::arg("func"), 
            py::arg("bounds"), 
            py::arg("x0") = std::vector<std::vector<double>>(),
            py::arg("data") = nullptr, 
            py::arg("callback") = nullptr, 
            py::arg("tol") = 0.0001, 
            py::arg("maxevals") = 100000, 
            py::arg("seed") = -1, 
            py::arg("options") = std::map<std::string, ConfigValue>())
        .def("optimize", &NelderMead::optimize, py::call_guard<py::gil_scoped_release>());

    py::class_<Minimizer>(m, "Minimizer")
        .def(py::init<MinionFunction, const std::vector<std::pair<double, double>>&,
                      const std::vector<std::vector<double>>&, void*, std::function<void(MinionResult*)>,
                      std::string, double, size_t, int, std::map<std::string, ConfigValue> >(),
            py::arg("func"), 
            py::arg("bounds"), 
            py::arg("x0") = std::vector<std::vector<double>>(),
            py::arg("data") = nullptr, 
            py::arg("callback") = nullptr, 
            py::arg("algo") = "ARRDE",
            py::arg("tol") = 0.0001, 
            py::arg("maxevals") = 100000, 
            py::arg("seed") = -1, 
            py::arg("options") = std::map<std::string, ConfigValue>())

        .def_readwrite("history", &Minimizer::history)
        .def("optimize", &Minimizer::optimize, py::call_guard<py::gil_scoped_release>());

    py::class_<CEC2014Functions>(m, "CEC2014Functions")
        .def(py::init<int, int>(), py::arg("function_number"), py::arg("dimension"))
        .def("__call__", &CEC2014Functions::operator());

    py::class_<CEC2017Functions>(m, "CEC2017Functions")
        .def(py::init<int, int>(), py::arg("function_number"), py::arg("dimension"))
        .def("__call__", &CEC2017Functions::operator());

    py::class_<CEC2019Functions>(m, "CEC2019Functions")
        .def(py::init<int, int>(), py::arg("function_number"), py::arg("dimension"))
        .def("__call__", &CEC2019Functions::operator());

    py::class_<CEC2020Functions>(m, "CEC2020Functions")
        .def(py::init<int, int>(), py::arg("function_number"), py::arg("dimension"))
        .def("__call__", &CEC2020Functions::operator());

    py::class_<CEC2022Functions>(m, "CEC2022Functions")
        .def(py::init<int, int>(), py::arg("function_number"), py::arg("dimension"))
        .def("__call__", &CEC2022Functions::operator());
}
