import os
import unittest
import numpy as np
from src.spatial_agent_model import Spatial_Agent_Model, neighbors
from unittest.mock import patch


class TestInvalidInputs(unittest.TestCase):
    breeding_counts = [3, 3, 3]
    minimum_counts = [2, 2, 2]
    overpopulation_counts = [6, 4, 3]
    hunt_success_probs = [1, 1]
    A0 = 100
    B0 = 80
    N_release = 20
    C_N = 100
    n_steps = 100
    m = 40
    n = 40
    animation_length = 10

    def test_breeding_counts(self):
        regex = ["breeding_counts should be a list of 3 numbers",
                 "Each count in list should be an integer",
                 "Each count should be non-negative"]
        error_types = [ValueError,
                      TypeError,
                      ValueError]
        incorrect_inputs = [[3, 5],
                            ["low", "high", "max"],
                            [-1, 0, 2]]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   Spatial_Agent_Model,
                                   incorrect_inputs[i],
                                   self.minimum_counts,
                                   self.overpopulation_counts,
                                   self.hunt_success_probs,
                                   self.A0,
                                   self.B0,
                                   self.N_release,
                                   self.C_N,
                                   self.n_steps,
                                   self.m,
                                   self.n,
                                   self.animation_length)

    def test_minimum_counts(self):
        regex = ["minimum_counts should be a list of 3 numbers",
                 "Each count in list should be an integer",
                 "Each count should be non-negative"]
        error_types = [ValueError,
                       TypeError,
                       ValueError]
        incorrect_inputs = [[3, 5],
                            ["low", "high", "max"],
                            [-1, 0, 2]]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   Spatial_Agent_Model,
                                   self.breeding_counts,
                                   incorrect_inputs[i],
                                   self.overpopulation_counts,
                                   self.hunt_success_probs,
                                   self.A0,
                                   self.B0,
                                   self.N_release,
                                   self.C_N,
                                   self.n_steps,
                                   self.m,
                                   self.n,
                                   self.animation_length)

    def test_overpopulation_counts(self):
        regex = ["overpopulation_counts should be a list of 3 numbers",
                 "Each count in list should be an integer",
                 "Each count should be non-negative"]
        error_types = [ValueError,
                       TypeError,
                       ValueError]
        incorrect_inputs = [[3, 5],
                            ["low", "high", "max"],
                            [-1, 0, 2]]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   Spatial_Agent_Model,
                                   self.breeding_counts,
                                   self.minimum_counts,
                                   incorrect_inputs[i],
                                   self.hunt_success_probs,
                                   self.A0,
                                   self.B0,
                                   self.N_release,
                                   self.C_N,
                                   self.n_steps,
                                   self.m,
                                   self.n,
                                   self.animation_length)

    def test_hunt_success_probs(self):
        regex = ["hunt_success_probs should be a list of 2 numbers",
                 "Each probability in hunt_success_probs should be between 0 and 1"]
        error_types = [ValueError,
                       ValueError]
        incorrect_inputs = [[0.005],
                            [-1.5, 2]]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   Spatial_Agent_Model,
                                   self.breeding_counts,
                                   self.minimum_counts,
                                   self.overpopulation_counts,
                                   incorrect_inputs[i],
                                   self.A0,
                                   self.B0,
                                   self.N_release,
                                   self.C_N,
                                   self.n_steps,
                                   self.m,
                                   self.n,
                                   self.animation_length)

    def test_A0(self):
        regex = ["The initial population of species A, A0, must be an integer",
                 "The initial population of species A, A0, must be non-negative"]
        error_types = [TypeError,
                       ValueError]
        incorrect_inputs = [72.5,
                            -100]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   Spatial_Agent_Model,
                                   self.breeding_counts,
                                   self.minimum_counts,
                                   self.overpopulation_counts,
                                   self.hunt_success_probs,
                                   incorrect_inputs[i],
                                   self.B0,
                                   self.N_release,
                                   self.C_N,
                                   self.n_steps,
                                   self.m,
                                   self.n,
                                   self.animation_length)

    def test_B0(self):
        regex = ["The initial population of species B, B0, must be an integer",
                 "The initial population of species B, B0, must be non-negative"]
        error_types = [TypeError,
                       ValueError]
        incorrect_inputs = [72.5,
                            -100]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   Spatial_Agent_Model,
                                   self.breeding_counts,
                                   self.minimum_counts,
                                   self.overpopulation_counts,
                                   self.hunt_success_probs,
                                   self.A0,
                                   incorrect_inputs[i],
                                   self.N_release,
                                   self.C_N,
                                   self.n_steps,
                                   self.m,
                                   self.n,
                                   self.animation_length)

    def test_N_release(self):
        regex = ["N_release must be an integer",
                 "N_release must be between 0 and n_steps"]
        error_types = [TypeError,
                       ValueError]
        incorrect_inputs = [5.5,
                            150]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   Spatial_Agent_Model,
                                   self.breeding_counts,
                                   self.minimum_counts,
                                   self.overpopulation_counts,
                                   self.hunt_success_probs,
                                   self.A0,
                                   self.B0,
                                   incorrect_inputs[i],
                                   self.C_N,
                                   self.n_steps,
                                   self.m,
                                   self.n,
                                   self.animation_length)

    def test_C_N(self):
        regex = ["The population of superpredators at release, C_N, must be an integer",
                 "The population of superpredators at release, C_N, must be non-negative"]
        error_types = [TypeError,
                       ValueError]
        incorrect_inputs = [1.5,
                            -1]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   Spatial_Agent_Model,
                                   self.breeding_counts,
                                   self.minimum_counts,
                                   self.overpopulation_counts,
                                   self.hunt_success_probs,
                                   self.A0,
                                   self.B0,
                                   self.N_release,
                                   incorrect_inputs[i],
                                   self.n_steps,
                                   self.m,
                                   self.n,
                                   self.animation_length)

    def test_n_steps(self):
        regex = ["n_steps must be an integer",
                 "n_steps must non negative"]
        error_types = [TypeError,
                       ValueError]
        incorrect_inputs = [55.5,
                            -100]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   Spatial_Agent_Model,
                                   self.breeding_counts,
                                   self.minimum_counts,
                                   self.overpopulation_counts,
                                   self.hunt_success_probs,
                                   self.A0,
                                   self.B0,
                                   self.N_release,
                                   self.C_N,
                                   incorrect_inputs[i],
                                   self.m,
                                   self.n,
                                   self.animation_length)

    def test_m(self):
        regex = ["The number of rows of the spatial grid must be an integer",
                 "The number of rows of the spatial grid must be non-negative"]
        error_types = [TypeError,
                       ValueError]
        incorrect_inputs = [12.5,
                            -10]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   Spatial_Agent_Model,
                                   self.breeding_counts,
                                   self.minimum_counts,
                                   self.overpopulation_counts,
                                   self.hunt_success_probs,
                                   self.A0,
                                   self.B0,
                                   self.N_release,
                                   self.C_N,
                                   self.n_steps,
                                   incorrect_inputs[i],
                                   self.n,
                                   self.animation_length)

    def test_n(self):
        regex = ["The number of columns of the spatial grid must be an integer",
                 "The number of columns of the spatial grid must be non-negative"]
        error_types = [TypeError,
                       ValueError]
        incorrect_inputs = [12.5,
                            -10]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   Spatial_Agent_Model,
                                   self.breeding_counts,
                                   self.minimum_counts,
                                   self.overpopulation_counts,
                                   self.hunt_success_probs,
                                   self.A0,
                                   self.B0,
                                   self.N_release,
                                   self.C_N,
                                   self.n_steps,
                                   self.m,
                                   incorrect_inputs[i],
                                   self.animation_length)
    def test_animation_length(self):
        regex = ["The animation length time in seconds must be a float or an integer",
                 "The animation length time in seconds must be non-negative"]
        error_types = [TypeError,
                       ValueError]
        incorrect_inputs = ["max",
                            -10]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   Spatial_Agent_Model,
                                   self.breeding_counts,
                                   self.minimum_counts,
                                   self.overpopulation_counts,
                                   self.hunt_success_probs,
                                   self.A0,
                                   self.B0,
                                   self.N_release,
                                   self.C_N,
                                   self.n_steps,
                                   self.m,
                                   self.n,
                                   incorrect_inputs[i])


class TestSpatialAgentModelFunctions(unittest.TestCase):
    def setUp(self):
        self.breeding_counts = [5, 5, 5]
        self.minimum_counts = [2, 2, 1]
        self.overpopulation_counts = [6, 4, 4]
        self.hunt_success_probs = [1, 1]
        self.A0 = 8
        self.B0 = 4
        self.N_release = 1
        self.C_N = 5
        self.n_steps = 20
        self.m = 5
        self.n = 5
        self.animation_length = 5

    def test_initialization(self):
        model = Spatial_Agent_Model(
            self.breeding_counts, self.minimum_counts,
            self.overpopulation_counts, self.hunt_success_probs,
            self.A0, self.B0, self.N_release, self.C_N,
            self.n_steps, self.m, self.n, self.animation_length
        )

        # Check if the grid has the correct shape
        self.assertEqual(model.grid.shape, (self.m, self.n))

    def test_add_new_agents(self):
        model = Spatial_Agent_Model(
            self.breeding_counts, self.minimum_counts,
            self.overpopulation_counts, self.hunt_success_probs,
            self.A0, self.B0, self.N_release, self.C_N,
            self.n_steps, self.m, self.n, self.animation_length
        )
        initial_count_A = np.sum(model.grid == 1)
        initial_count_B = np.sum(model.grid == 2)
        initial_count_C = np.sum(model.grid == 3)

        # Call the add_new_agents method to add more agents
        model.add_new_agents(5, 3, 2)

        # Check if the counts have increased accordingly
        self.assertEqual(
            np.sum(model.grid == 1) - initial_count_A, 5,
            "Population of species A did not increase correctly."
        )
        self.assertEqual(
            np.sum(model.grid == 2) - initial_count_B, 3,
            "Population of species B did not increase correctly."
        )
        self.assertEqual(
            np.sum(model.grid == 3) - initial_count_C, 2,
            "Population of species C did not increase correctly."
        )

    def test_iteration(self):
        model = Spatial_Agent_Model(
            self.breeding_counts, self.minimum_counts,
            self.overpopulation_counts, self.hunt_success_probs,
            self.A0, self.B0, self.N_release, self.C_N,
            self.n_steps, self.m, self.n, self.animation_length
        )

        # set grid to new initial starting point
        start_grid = np.array([[3, 3, 0, 1, 1],
                               [3, 3, 0, 1, 1],
                               [0, 0, 0, 1, 1],
                               [2, 2, 2, 0, 0],
                               [2, 1, 2, 3, 3]])
        model.grid = start_grid

        # iterate one frame and check result is expected
        im = model.iteration(model.n_wait + 1)
        # we expect 3 cells to become empty due to 4x huntings and 1x overpopulation
        one_step_grid = np.array([[3, 3, 0, 1, 1],
                                  [3, 3, 0, 1, 1],
                                  [0, 0, 0, 0, 1],
                                  [2, 0, 0, 0, 0],
                                  [2, 0, 0, 3, 3]])
        np.testing.assert_array_equal(model.grid, one_step_grid)

        # iterate one more frame
        im = model.iteration(model.n_wait + 1)
        # we expect 2 cells to become empty due to underpopulation
        final_grid = np.array([[3, 3, 0, 1, 1],
                               [3, 3, 0, 1, 1],
                               [0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 3, 3]])
        np.testing.assert_array_equal(model.grid, final_grid)

        # check convergence
        im = model.iteration(model.n_wait + 1)
        np.testing.assert_array_equal(model.grid, final_grid)

        # Check lists of A, B, and C populations have been updated
        self.assertEqual(np.sum(model.grid == 1), model.num_A_n[-1],
            "Population of species A did not update as expected.")
        self.assertEqual(np.sum(model.grid == 2), model.num_B_n[-1],
            "Population of species B did not update as expected.")
        self.assertEqual(np.sum(model.grid == 3), model.num_C_n[-1],
            "Population of species C did not update as expected.")

    def test_count_species(self):
        model = Spatial_Agent_Model(
            self.breeding_counts, self.minimum_counts,
            self.overpopulation_counts, self.hunt_success_probs,
            self.A0, self.B0, self.N_release, self.C_N,
            self.n_steps, self.m, self.n, self.animation_length
        )
        model.grid = np.array([[3, 3, 0, 1, 1],
                               [3, 3, 0, 1, 1],
                               [0, 0, 0, 1, 1],
                               [2, 2, 2, 0, 0],
                               [2, 1, 2, 3, 0]])

        # Test the count_species function (check cell in row 4, columnn 4)
        count_A = model.count_species(3, 3, 1)
        count_B = model.count_species(3, 3, 2)
        count_C = model.count_species(3, 3, 3)

        # Check if counts are correct
        self.assertEqual(count_A, 2, "Count of species A is incorrect.")
        self.assertEqual(count_B, 2, "Count of species B is incorrect.")
        self.assertEqual(count_C, 1, "Count of species C is incorrect.")

    def test_count_all(self):
        model = Spatial_Agent_Model(
            self.breeding_counts, self.minimum_counts,
            self.overpopulation_counts, self.hunt_success_probs,
            self.A0, self.B0, self.N_release, self.C_N,
            self.n_steps, self.m, self.n, self.animation_length
        )
        model.grid = np.array([[3, 3, 0, 1, 1],
                               [3, 3, 0, 1, 1],
                               [0, 0, 0, 1, 1],
                               [2, 2, 2, 0, 0],
                               [2, 1, 2, 3, 0]])

        # Test the count_all function for the cell in the 4th row, 4th column
        counts = model.count_all(3, 3)

        # Check if counts are correct
        self.assertEqual(counts, (2, 2, 1), "Counts are incorrect.")

    def test_empty_neighbors(self):
        model = Spatial_Agent_Model(
            self.breeding_counts, self.minimum_counts,
            self.overpopulation_counts, self.hunt_success_probs,
            self.A0, self.B0, self.N_release, self.C_N,
            self.n_steps, self.m, self.n, self.animation_length
        )
        model.grid = np.array([[3, 3, 0, 1, 1],
                               [3, 3, 0, 1, 1],
                               [0, 0, 0, 1, 1],
                               [2, 2, 2, 0, 0],
                               [2, 1, 2, 3, 0]])

        # Test the empty_neighbors function for the lower right corner cell
        empty_neighbors = model.empty_neighbors(4, 4)

        # Check if the empty neighbors are correct
        self.assertEqual(len(empty_neighbors), 2, "Number of empty neighbors is incorrect.")
        np.testing.assert_array_equal(empty_neighbors[0], (3, 3), "Empty neighbor position is incorrect.")

    @unittest.mock.patch("matplotlib.animation.FuncAnimation.save")
    def test_save_gif(self, mock_gif):
        model = Spatial_Agent_Model(
            self.breeding_counts, self.minimum_counts,
            self.overpopulation_counts, self.hunt_success_probs,
            self.A0, self.B0, self.N_release, self.C_N,
            self.n_steps, self.m, self.n, self.animation_length
        )
        filename = "test_animation.gif"
        model.save_gif(filename)

        # Test if gif has been saved
        mock_gif.assert_called_with(filename)

    @unittest.mock.patch("matplotlib.pyplot.savefig")
    def test_plot(self, mock_plot):
        model = Spatial_Agent_Model(
            self.breeding_counts, self.minimum_counts,
            self.overpopulation_counts, self.hunt_success_probs,
            self.A0, self.B0, self.N_release, self.C_N,
            self.n_steps, self.m, self.n, self.animation_length
        )
        title = "Population Dynamics Over Time"
        filename = "test_plot.png"

        # Note due to how FuncAnimation works, the animation must be viewed or saved first
        # Before the plot function can be called
        model.save_gif("test_animation.gif")
        # delete gif
        os.remove("test_animation.gif")

        # Test the plot has been saved
        model.plot(title, filename)
        mock_plot.assert_called_with(filename)

    def test_get_population_extremes(self):
        model = Spatial_Agent_Model(
            self.breeding_counts, self.minimum_counts,
            self.overpopulation_counts, self.hunt_success_probs,
            self.A0, self.B0, self.N_release, self.C_N,
            self.n_steps, self.m, self.n, self.animation_length
        )

        # set grid to new initial starting point
        start_grid = np.array([[3, 3, 0, 1, 1],
                               [3, 3, 0, 1, 1],
                               [0, 0, 0, 1, 1],
                               [2, 2, 2, 0, 0],
                               [2, 1, 2, 3, 3]])
        model.grid = start_grid

        # Reset population lists
        model.num_A_n = [7]
        model.num_B_n = [5]
        model.num_C_n = [6]

        # Run a few iterations until convergence
        for i in range(10):
            im = model.iteration(model.n_wait + 1)

        # Test the get_population_extremes function
        max_A, min_A, max_B, min_B, max_C, min_C = model.get_population_extremes()
        self.assertEqual(max_A, 7)
        self.assertEqual(min_A, 5)
        self.assertEqual(max_B, 5)
        self.assertEqual(min_B, 0)
        self.assertEqual(max_C, 6)
        self.assertEqual(min_C, 6)

class TestNeighborsFunction(unittest.TestCase):

    def test_neighbors(self):
        # Test the neighbors function for a given position
        m, n = 5, 5

        # Test middle of grid
        i, j = 2, 2
        # Get neighbors using the function
        result = list(neighbors(i, j, m, n))
        expected_neighbors = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]

        # Check if the result matches the expected neighbors
        self.assertEqual(result, expected_neighbors)

        # Test corner of grid
        i, j = 4, 4
        # Get neighbors using the function
        result = list(neighbors(i, j, m, n))
        expected_neighbors = [(3, 3), (3, 4), (4, 3)]

        # Check if the result matches the expected neighbors
        self.assertEqual(result, expected_neighbors)


if __name__ == '__main__':
    unittest.main()
