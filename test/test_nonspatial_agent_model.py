from src.nonspatial_agent_model import Nonspatial_Agent, Nonspatial_Agent_Model
import unittest
from unittest.mock import patch
import numpy as np

class TestInvalidInputs(unittest.TestCase):
    init_healths = [3, 5, 20]
    hunt_success_probs = [0.005, 0.002]
    food_values = [2, 5]
    reproduction_probs = [0.38, 0.05, 0]
    carrying_capacity_A = 100
    A0 = 100
    B0 = 10
    N_release = 10
    C_N = 1
    n_steps = 100

    def test_init_healths(self):
        regex = ["init_healths should be a list of 3 numbers",
                 "Each health in init_healths should be either a float or an integer"]
        error_types = [ValueError,
                       TypeError]
        incorrect_inputs = [[3, 5],
                            ["low", "high", "max"]]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   Nonspatial_Agent_Model,
                                   incorrect_inputs[i],
                                   self.hunt_success_probs,
                                   self.food_values,
                                   self.reproduction_probs,
                                   self.carrying_capacity_A,
                                   self.A0,
                                   self.B0,
                                   self.N_release,
                                   self.C_N,
                                   self.n_steps)

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
                                   Nonspatial_Agent_Model,
                                   self.init_healths,
                                   incorrect_inputs[i],
                                   self.food_values,
                                   self.reproduction_probs,
                                   self.carrying_capacity_A,
                                   self.A0,
                                   self.B0,
                                   self.N_release,
                                   self.C_N,
                                   self.n_steps)


    def test_food_values(self):
        regex = ["food_values should be a list of 2 numbers",
                 "Each value in food_values should be either a float or an integer",
                 "Each value in food_values should be a positive number"]
        error_types = [ValueError,
                       TypeError,
                       ValueError]
        incorrect_inputs = [[2, 10, 0],
                            [1, None],
                            [1, -1]]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   Nonspatial_Agent_Model,
                                   self.init_healths,
                                   self.hunt_success_probs,
                                   incorrect_inputs[i],
                                   self.reproduction_probs,
                                   self.carrying_capacity_A,
                                   self.A0,
                                   self.B0,
                                   self.N_release,
                                   self.C_N,
                                   self.n_steps)

    def test_reproduction_probs(self):
        regex = ["reproduction_probs should be a list of 3 numbers",
                 "Each probability in reproduction_probs should be between 0 and 1"]
        error_types = [ValueError,
                       ValueError]
        incorrect_inputs = [[0.38, 0.05],
                            [1.05, 0.05, 0.1]]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   Nonspatial_Agent_Model,
                                   self.init_healths,
                                   self.hunt_success_probs,
                                   self.food_values,
                                   incorrect_inputs[i],
                                   self.carrying_capacity_A,
                                   self.A0,
                                   self.B0,
                                   self.N_release,
                                   self.C_N,
                                   self.n_steps)

    def test_carrying_capacity_A(self):
        regex = ["The carrying capacity should be an integer",
                 "The carrying capacity should be non negative"]
        error_types = [TypeError,
                       ValueError]
        incorrect_inputs = ["low",
                            -10]
        for i in range(len(regex)):
            self.assertRaisesRegex(error_types[i],
                                   regex[i],
                                   Nonspatial_Agent_Model,
                                   self.init_healths,
                                   self.hunt_success_probs,
                                   self.food_values,
                                   self.reproduction_probs,
                                   incorrect_inputs[i],
                                   self.A0,
                                   self.B0,
                                   self.N_release,
                                   self.C_N,
                                   self.n_steps)


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
                                   Nonspatial_Agent_Model,
                                   self.init_healths,
                                   self.hunt_success_probs,
                                   self.food_values,
                                   self.reproduction_probs,
                                   self.carrying_capacity_A,
                                   incorrect_inputs[i],
                                   self.B0,
                                   self.N_release,
                                   self.C_N,
                                   self.n_steps)

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
                                   Nonspatial_Agent_Model,
                                   self.init_healths,
                                   self.hunt_success_probs,
                                   self.food_values,
                                   self.reproduction_probs,
                                   self.carrying_capacity_A,
                                   self.A0,
                                   incorrect_inputs[i],
                                   self.N_release,
                                   self.C_N,
                                   self.n_steps)

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
                                   Nonspatial_Agent_Model,
                                   self.init_healths,
                                   self.hunt_success_probs,
                                   self.food_values,
                                   self.reproduction_probs,
                                   self.carrying_capacity_A,
                                   self.A0,
                                   self.B0,
                                   incorrect_inputs[i],
                                   self.C_N,
                                   self.n_steps)

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
                                   Nonspatial_Agent_Model,
                                   self.init_healths,
                                   self.hunt_success_probs,
                                   self.food_values,
                                   self.reproduction_probs,
                                   self.carrying_capacity_A,
                                   self.A0,
                                   self.B0,
                                   self.N_release,
                                   incorrect_inputs[i],
                                   self.n_steps)

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
                                   Nonspatial_Agent_Model,
                                   self.init_healths,
                                   self.hunt_success_probs,
                                   self.food_values,
                                   self.reproduction_probs,
                                   self.carrying_capacity_A,
                                   self.A0,
                                   self.B0,
                                   self.N_release,
                                   self.C_N,
                                   incorrect_inputs[i])

class TestNonspatialAgent(unittest.TestCase):
    def setUp(self):
        self.health = 10
        self.hunt_success_prob = 1
        self.food_value = 5
        self.reproduction_prob = 0
        self.species = "C"

    def test_initialization(self):
        agent = Nonspatial_Agent(self.health,
                                 self.hunt_success_prob,
                                 self.food_value,
                                 self.reproduction_prob,
                                 self.species)

        self.assertTrue(agent.is_alive)

    def test_pass_day(self):
        agent = Nonspatial_Agent(self.health,
                                 self.hunt_success_prob,
                                 self.food_value,
                                 self.reproduction_prob,
                                 self.species)
        agent.pass_day()
        self.assertEqual(agent.health, 9)
        self.assertTrue(agent.is_alive)

    def test_hunt(self):
        agent = Nonspatial_Agent(self.health,
                                 self.hunt_success_prob,
                                 self.food_value,
                                 self.reproduction_prob,
                                 self.species)
        result = agent.hunt()
        self.assertTrue(result)  # As hunt_success_prob is 1, this should be True

    def test_feed(self):
        agent = Nonspatial_Agent(self.health,
                                 self.hunt_success_prob,
                                 self.food_value,
                                 self.reproduction_prob,
                                 self.species)
        agent.feed(2)
        self.assertEqual(agent.health, 12)

    def test_reproduce(self):
        agent = Nonspatial_Agent(self.health,
                                 self.hunt_success_prob,
                                 self.food_value,
                                 self.reproduction_prob,
                                 self.species)
        result = agent.reproduce()
        self.assertFalse(result)  # As reproduction_prob is 0, this should be False

    def test_die(self):
        agent = Nonspatial_Agent(self.health,
                                 self.hunt_success_prob,
                                 self.food_value,
                                 self.reproduction_prob,
                                 self.species)
        agent.die()
        self.assertEqual(agent.health, 0)
        self.assertFalse(agent.is_alive)

class TestNonspatialAgentModel(unittest.TestCase):
    def setUp(self):
        self.init_healths = [5, 10, 15]
        self.hunt_success_probs = [1, 1]
        self.food_values = [0.1, 0.5]
        self.reproduction_probs = [1, 0, 0]
        self.carrying_capacity_A = 100
        self.A0 = 50
        self.B0 = 20
        self.N_release = 20
        self.C_N = 5
        self.n_steps = 120

    def test_model_initialization(self):
        model = Nonspatial_Agent_Model(self.init_healths,
                                       self.hunt_success_probs,
                                       self.food_values,
                                       self.reproduction_probs,
                                       self.carrying_capacity_A,
                                       self.A0,
                                       self.B0,
                                       self.N_release,
                                       self.C_N,
                                       self.n_steps)

        self.assertEqual(len(model.num_A_n), 121)
        self.assertEqual(len(model.num_B_n), 121)
        self.assertEqual(len(model.num_C_n), 121)

    def test_add_new_agents(self):
        model = Nonspatial_Agent_Model(self.init_healths,
                                       self.hunt_success_probs,
                                       self.food_values,
                                       self.reproduction_probs,
                                       self.carrying_capacity_A,
                                       self.A0,
                                       self.B0,
                                       self.N_release,
                                       self.C_N,
                                       self.n_steps)

        # Reset lists of populations
        model.A_list = []
        model.B_list = []
        model.C_list = []

        # Add varying numbers of agents to each population
        model.add_new_agents(3, 2, 1)
        self.assertEqual(len(model.A_list), 3)
        self.assertEqual(len(model.B_list), 2)
        self.assertEqual(len(model.C_list), 1)

    def test_remove_dead_agents(self):
        model = Nonspatial_Agent_Model(self.init_healths,
                                       self.hunt_success_probs,
                                       self.food_values,
                                       self.reproduction_probs,
                                       self.carrying_capacity_A,
                                       self.A0,
                                       self.B0,
                                       self.N_release,
                                       self.C_N,
                                       self.n_steps)

        # Initialise population list of A with a new agent
        new_agent = Nonspatial_Agent(1, 0.8, 5, 0.5, 'A')
        model.A_list = [new_agent]
        self.assertTrue(model.A_list[0].is_alive)

        # Make agent die and test if it has been removed
        new_agent.die()
        model.remove_dead_agents()
        self.assertEqual(len(model.A_list), 0)

    def test_iteration(self):
        model = Nonspatial_Agent_Model(self.init_healths,
                                       self.hunt_success_probs,
                                       self.food_values,
                                       self.reproduction_probs,
                                       self.carrying_capacity_A,
                                       self.A0,
                                       self.B0,
                                       self.N_release,
                                       self.C_N,
                                       self.n_steps)

        # Reset values back to initials
        model.A_list = []
        model.B_list = []
        model.C_list = []
        model.num_A_n = [self.A0]
        model.num_B_n = [self.B0]
        model.num_C_n = [0]

        # construct starting populations
        model.add_new_agents(self.A0, self.B0, 0)

        # Set random seed for reproducibility
        np.random.seed(self.A0 + self.B0 + self.C_N)

        # Run 10 iterations and check values are as expected
        for i in range(10):
            model.iteration()
        self.assertEqual(len(model.A_list), 46)
        self.assertEqual(len(model.B_list), 10)
        self.assertEqual(len(model.C_list), 0)

        # Run 10 iterations until predator release
        for i in range(10):
            model.iteration()
        self.assertEqual(len(model.A_list), 55)
        self.assertEqual(len(model.B_list), 1)
        self.assertEqual(len(model.C_list), 0)

        # Run 10 iterations after predator release
        model.add_new_agents(0, 0, self.C_N)
        for i in range(10):
            model.iteration()
        self.assertEqual(len(model.A_list), 52)
        self.assertEqual(len(model.B_list), 0)
        self.assertEqual(len(model.C_list), 5)

    @unittest.mock.patch("matplotlib.pyplot.savefig")
    def test_plot(self, mock_plot):
        model = Nonspatial_Agent_Model(self.init_healths,
                                       self.hunt_success_probs,
                                       self.food_values,
                                       self.reproduction_probs,
                                       self.carrying_capacity_A,
                                       self.A0,
                                       self.B0,
                                       self.N_release,
                                       self.C_N,
                                       self.n_steps)
        title = "Population Dynamics Over Time"
        filename = "test_plot.png"

        # Test the plot has been saved
        model.plot(title, filename)
        mock_plot.assert_called_with(filename)

    def test_get_population_extremes(self):
        model = Nonspatial_Agent_Model(self.init_healths,
                                       self.hunt_success_probs,
                                       self.food_values,
                                       self.reproduction_probs,
                                       self.carrying_capacity_A,
                                       self.A0,
                                       self.B0,
                                       self.N_release,
                                       self.C_N,
                                       self.n_steps)

        # Test against expected results for this seed
        max_A, min_A, max_B, min_B, max_C, min_C = model.get_population_extremes()
        self.assertEqual(max_A, 67)
        self.assertEqual(min_A, 0)
        self.assertEqual(max_B, 20)
        self.assertEqual(min_B, 0)
        self.assertEqual(max_C, 5)
        self.assertEqual(min_C, 0)
