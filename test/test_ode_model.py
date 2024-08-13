from src.ode_model import ODE_Superpredator_Release
import unittest
import numpy as np
from unittest.mock import patch


class TestInvalidInputs(unittest.TestCase):
    r = [2, 0.5, 0.5]
    K = 200
    mu = [0.1, 0.01]
    nu = [0.02, 0.002]
    eta = [0.05, 0.004]
    t_bounds = [0, 50]
    A0 = 50
    B0 = 10
    T_release = 25
    C_T = 5

    def test_r(self):
        regex = ["r should be a list of 3 numbers",
                 "Each rate in r should be either a float or an integer"]
        error_types = [ValueError,
                       TypeError]
        incorrect_inputs = [[1, 2],
                            [1, 2, "a"]]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   ODE_Superpredator_Release,
                                   incorrect_inputs[i],
                                   self.K,
                                   self.mu,
                                   self.nu,
                                   self.eta,
                                   self.t_bounds,
                                   self.A0,
                                   self.B0,
                                   self.T_release,
                                   self.C_T)


    def test_K(self):
        regex = ["The carrying capacity, K, should be an integer",
                 "The carrying capacity, K, should be non negative"]
        error_types = [TypeError,
                       ValueError]
        incorrect_inputs = [0.1,
                            -5]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   ODE_Superpredator_Release,
                                   self.r,
                                   incorrect_inputs[i],
                                   self.mu,
                                   self.nu,
                                   self.eta,
                                   self.t_bounds,
                                   self.A0,
                                   self.B0,
                                   self.T_release,
                                   self.C_T)


    def test_mu(self):
        regex = ["mu should be a list of 2 numbers",
                 "Each rate in mu should be either a float or an integer"]
        error_types = [ValueError,
                       TypeError]
        incorrect_inputs = [[0.1],
                            [0.1, "0.01"]]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   ODE_Superpredator_Release,
                                   self.r,
                                   self.K,
                                   incorrect_inputs[i],
                                   self.nu,
                                   self.eta,
                                   self.t_bounds,
                                   self.A0,
                                   self.B0,
                                   self.T_release,
                                   self.C_T)

    def test_nu(self):
        regex = ["nu should be a list of 2 numbers",
                 "Each rate in nu should be either a float or an integer"]
        error_types = [ValueError,
                       TypeError]
        incorrect_inputs = [[0.1],
                            [0.1, "0.01"]]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   ODE_Superpredator_Release,
                                   self.r,
                                   self.K,
                                   self.mu,
                                   incorrect_inputs[i],
                                   self.eta,
                                   self.t_bounds,
                                   self.A0,
                                   self.B0,
                                   self.T_release,
                                   self.C_T)

    def test_eta(self):
        regex = ["eta should be a list of 2 numbers",
                 "Each rate in eta should be either a float or an integer"]
        error_types = [ValueError,
                       TypeError]
        incorrect_inputs = [[0.1],
                            [0.1, "0.01"]]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   ODE_Superpredator_Release,
                                   self.r,
                                   self.K,
                                   self.mu,
                                   self.nu,
                                   incorrect_inputs[i],
                                   self.t_bounds,
                                   self.A0,
                                   self.B0,
                                   self.T_release,
                                   self.C_T)


    def test_t_bounds(self):
        regex = ["t_bounds should be a list of 2 numbers",
                 "Each time in t_bounds should be either a float or an integer"]
        error_types = [ValueError,
                       TypeError]
        incorrect_inputs = [[0],
                            ["a", "b"]]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   ODE_Superpredator_Release,
                                   self.r,
                                   self.K,
                                   self.mu,
                                   self.nu,
                                   self.eta,
                                   incorrect_inputs[i],
                                   self.A0,
                                   self.B0,
                                   self.T_release,
                                   self.C_T)

    def test_A0(self):
        regex = ["The initial population of species A, A0, must be an integer",
                 "The initial population of species A, A0, must be non-negative"]
        error_types = [TypeError,
                       ValueError]
        incorrect_inputs = [0.5,
                            -50]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   ODE_Superpredator_Release,
                                   self.r,
                                   self.K,
                                   self.mu,
                                   self.nu,
                                   self.eta,
                                   self.t_bounds,
                                   incorrect_inputs[i],
                                   self.B0,
                                   self.T_release,
                                   self.C_T)

    def test_B0(self):
        regex = ["The initial population of species B, B0, must be an integer",
                 "The initial population of species B, B0, must be non-negative"]
        error_types = [TypeError,
                       ValueError]
        incorrect_inputs = [0.5,
                            -50]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   ODE_Superpredator_Release,
                                   self.r,
                                   self.K,
                                   self.mu,
                                   self.nu,
                                   self.eta,
                                   self.t_bounds,
                                   self.A0,
                                   incorrect_inputs[i],
                                   self.T_release,
                                   self.C_T)

    def test_T_release(self):
        regex = ["T_release must be a float or an integer",
                 "T_release must be between the start and end time of the ODE"]
        error_types = [TypeError,
                       ValueError]
        incorrect_inputs = ["10 days",
                            100]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   ODE_Superpredator_Release,
                                   self.r,
                                   self.K,
                                   self.mu,
                                   self.nu,
                                   self.eta,
                                   self.t_bounds,
                                   self.A0,
                                   self.B0,
                                   incorrect_inputs[i],
                                   self.C_T)

    def test_C_T(self):
        regex = ["The initial population of superpredators, C_T, must be an integer",
                 "The initial population of superpredators, C_T, must be non-negative"]
        error_types = [TypeError,
                       ValueError]
        incorrect_inputs = [12.5,
                            -10]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   ODE_Superpredator_Release,
                                   self.r,
                                   self.K,
                                   self.mu,
                                   self.nu,
                                   self.eta,
                                   self.t_bounds,
                                   self.A0,
                                   self.B0,
                                   self.T_release,
                                   incorrect_inputs[i])


class TestODESuperpredatorRelease(unittest.TestCase):

    def setUp(self):
        # Parameters for testing (This is the "Mesopredator-Replacement Case" example)
        self.r = [4, 0.5, 0.5]
        self.K = 100
        self.mu = [0.1, 0.02]
        self.nu = [0.1, 0.01]
        self.eta = [0.25, 0.01]
        self.t_bounds = [0, 40]
        self.A0 = 50
        self.B0 = 10
        self.T_release = 15
        self.C_T = 5

    def test_initialization(self):
        model = ODE_Superpredator_Release(self.r, self.K, self.mu, self.nu, self.eta, self.t_bounds,
                                          self.A0, self.B0, self.T_release, self.C_T)

        self.assertEqual(model.r, [4, 0.5, 0.5])
        self.assertEqual(model.K, 100)
        self.assertEqual(model.mu, [0.1, 0.02])
        self.assertEqual(model.nu, [0.1, 0.01])
        self.assertEqual(model.eta, [0.25, 0.01])
        self.assertEqual(model.t_bounds, [0, 40])
        self.assertEqual(model.A0, 50)
        self.assertEqual(model.B0, 10)
        self.assertEqual(model.T_release, 15)
        self.assertEqual(model.C_T, 5)

    def test_f(self):
        model = ODE_Superpredator_Release(self.r, self.K, self.mu, self.nu, self.eta, self.t_bounds,
                                          self.A0, self.B0, self.T_release, self.C_T)

        t = 0
        y = [80, 20, 0]
        f_result = model.f(t, y, model.r, model.K, model.mu, model.nu, model.eta)
        f_expected = [-96, 22, 0]
        np.testing.assert_allclose(np.array(f_result), f_expected)

    @unittest.mock.patch("matplotlib.pyplot.savefig")
    def test_plot(self, mock_plot):
        model = ODE_Superpredator_Release(self.r, self.K, self.mu, self.nu, self.eta, self.t_bounds,
                                          self.A0, self.B0, self.T_release, self.C_T)

        title = "Population Dynamics Over Time"
        filename = "test_plot.png"

        # Test the plot has been saved
        model.plot(title, filename)
        mock_plot.assert_called_with(filename)

    def test_timespan(self):
        model = ODE_Superpredator_Release(self.r, self.K, self.mu, self.nu, self.eta, self.t_bounds,
                                          self.A0, self.B0, self.T_release, self.C_T)

        timespan = model.timespan
        self.assertTrue(isinstance(timespan, np.ndarray))
        self.assertEqual(len(timespan), len(model.A_t))
        self.assertTrue(len(timespan) > 200)

    def test_population_timeseries(self):
        model = ODE_Superpredator_Release(self.r, self.K, self.mu, self.nu, self.eta, self.t_bounds,
                                          self.A0, self.B0, self.T_release, self.C_T)

        A_t, B_t, C_t = model.population_timeseries
        self.assertEqual(len(A_t), len(model.t_arr))
        self.assertEqual(len(B_t), len(model.t_arr))
        self.assertEqual(len(C_t), len(model.t_arr))

        # time 0
        self.assertEqual(A_t[0], 50)
        self.assertEqual(B_t[0], 10)
        self.assertEqual(C_t[0], 0)

        # time index 100 (before superpredator release)
        np.testing.assert_allclose(A_t[100], 24.984486)
        np.testing.assert_allclose(B_t[100], 30.007274)
        self.assertEqual(C_t[100], 0)


    def test_final_populations(self):
        model = ODE_Superpredator_Release(self.r, self.K, self.mu, self.nu, self.eta, self.t_bounds,
                                          self.A0, self.B0, self.T_release, self.C_T)

        final_populations = model.final_populations
        np.testing.assert_almost_equal(final_populations[0], 50, decimal=3)
        np.testing.assert_almost_equal(final_populations[1], 0, decimal=3)
        np.testing.assert_almost_equal(final_populations[2], 20, decimal=3)

    def test_arguments(self):
        model = ODE_Superpredator_Release(self.r, self.K, self.mu, self.nu, self.eta, self.t_bounds,
                                          self.A0, self.B0, self.T_release, self.C_T)

        arguments = model.arguments
        self.assertTrue(isinstance(arguments, str))

    def test_equation_string(self):
        model = ODE_Superpredator_Release(self.r, self.K, self.mu, self.nu, self.eta, self.t_bounds,
                                          self.A0, self.B0, self.T_release, self.C_T)

        equation_string = model.equation_string
        self.assertTrue(isinstance(equation_string, str))

    def test_get_population_extremes(self):
        model = ODE_Superpredator_Release(self.r, self.K, self.mu, self.nu, self.eta, self.t_bounds,
                                          self.A0, self.B0, self.T_release, self.C_T)

        # Test the get_population_extremes function
        max_A, min_A, max_B, min_B, max_C, min_C = model.get_population_extremes()
        self.assertEqual(max_A, 75)
        self.assertEqual(min_A, 21)
        self.assertEqual(max_B, 33)
        self.assertEqual(min_B, 0)
        self.assertEqual(max_C, 20)
        self.assertEqual(min_C, 0)
