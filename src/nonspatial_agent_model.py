import numpy as np
import matplotlib.pyplot as plt
from src.tools import is_a_number


class Nonspatial_Agent:
    """
    Simple class used to represent one agent in the agent model.

    Instance Variables:
    *******************
    health = number of health "points" the agent has, or equivalently, the
        number of iterations until this agent dies, if no food is found
    hunt_success_prob = probability of successfully hunting some prey,
        float in (0,1)
    food_value = value of consuming this agent, i.e., health gained when hunt()
        is successful against this agent as prey.
    reproduction_prob = probability to reproduce at each iteration,
        float in (0,1). Reproduction is modeled to be asexual, only
        requiring one agent.
    species = String of one character, A, B or C, for prey, predator, and
        superpredator.
    is_alive = boolean, True if agent is alive, False otherwise.
    """
    def __init__(self, health, hunt_success_prob, food_value,
                reproduction_prob, species):
        """
        Initializes agent with given health, and sets the food_gain per kill,
            and reproduction_probability.
        """
        self.health = health
        self.hunt_success_prob = hunt_success_prob
        self.food_value = food_value
        self.reproduction_prob = reproduction_prob
        self.species = species
        self.is_alive = True

    def pass_day(self):
        """
        Decreases health by one to signify the passage of one day.
        """
        self.health = self.health - 1
        if self.health <= 0:
            self.is_alive = False

    def hunt(self):
        """
        Determines if agent was able to kill some prey, designed to be applied
            for every possible prey agent in a loop.

        :return: boolean, True if hunt is successful, False otherwise
        """
        return np.random.rand() < self.hunt_success_prob

    def feed(self, food_quantity):
        """
        Feeds the agent after a hunt, increasing health by food_quantity.
        :param food_quantity: integer, amount of health gained by feeding, may
            depend on species hunted.
        """
        self.health += food_quantity

    def reproduce(self):
        """
        Checks if the agent was able to reproduce in this iteration, depending
            on reproduction_prob.
        :return: True if Agent reproduced this iteration.
        """
        return np.random.rand() < self.reproduction_prob

    def die(self):
        """
        Kills the agent, setting health to zero and sets is_alive flag to False
        """
        self.health = 0
        self.is_alive = False


class Nonspatial_Agent_Model:
    """
    This class models the same superpredator release situation, but with an
        agent-based model instead. This class particularly does not spatially
        orient the agents, instead applying uniform probability of interaction
        of all agents at all discrete timesteps.

    Instance Variables (to determine agent behavior):
    *************************************************
    init_healths =  3-list of initial health values each species is
        born with. Species A,B,C initial healths at index 0,1,2 respectively
    hunt_success_probs =  2-list of probability that predator and
        superpredator, respectively, kill given prey during hunt-phase
    food_values = 2-list of food gained from killing prey and predator,
        repectively.
    reproduction_probs = 3-list of probability that each species reproduces.
        Species A,B,C probabilities at index 0,1,2 respectively
    carry_capacity_A = environmental carrying capacity for the prey population,
        so that reproduction of species A is 1.5 times more likely when the
        total population is below this rate.
    A0 = initial population of species A.
    B0 = initial population of species B.
    N_release = iteration number at which superpredators are released.
    C_N = population of superpredators to release at the (N_release)th
        iteration.
    n_steps = integer number of iteration steps to compute.

    Instance Variables (for simulation of interactions):
    ****************************************************
    A_list = list of prey Agents
    B_list = list of predator Agents
    C_list = list of superpredator Agents
    num_A_n = list of population of species A at each iteration
    num_B_n = list of population of species B at each iteration
    num_C_n = list of population of species B at each iteration
    """
    def __init__(self, init_healths, hunt_success_probs, food_values,
                 reproduction_probs, carrying_capacity_A, A0, B0, N_release,
                 C_N, n_steps):
        """
        Initalized the non-spatial model with given parameters and
            initial state

        :param init_healths: 3-list of initial health values each species is
            born with. Species A,B,C initial healths at index 0,1,2
            respectively
        :param hunt_success_probs: 2-list of probability that predator and
            superpredator kills given prey during hunt-phase
        :param food_values: 2-list of food gained from killing prey and predator,
            repectively
        :param reproduction_probs: 3-list of probability that each species
            reproduces. Species A,B,C probabilities at index 0,1,2 respectively
        :param carrying_capacity_A: environmental carrying capacity for the
            prey population, so that reproduction of species A is 1.5 times
            more likely when the total population is below this rate.
        :param A0: initial population of species A.
        :param B0: initial population of species B.
        :param N_release: interation number at which superpredators are
            released
        :param C_N: population of superpredators to release at the
            (N_release)th iteration
        :param n_steps: integer number of iterations steps to compute.
        """
        self.init_healths = init_healths
        self.hunt_success_probs = hunt_success_probs
        self.food_values = food_values
        self.reproduction_probs = reproduction_probs
        self.carrying_capacity_A = carrying_capacity_A
        self.A0 = A0
        self.B0 = B0
        self.n_steps = n_steps
        self.N_release = N_release
        self.C_N = C_N

        self.A_list = []
        self.B_list = []
        self.C_list = []
        self.num_A_n = [self.A0]
        self.num_B_n = [self.B0]
        self.num_C_n = [0]
        # construct starting populations
        self.add_new_agents(self.A0, self.B0, 0)
        # Seed randomness so simulation is repeatable.
        np.random.seed(self.A0 + self.B0 + self.C_N)

        # Run iterations until superpredator release
        for n in range(self.N_release):
            self.iteration()
        # Construct released superpredator agents
        self.add_new_agents(0, 0, self.C_N)
        # Finish remaining iterations
        for n in range(self.N_release, self.n_steps):
            self.iteration()

    def add_new_agents(self, num_A, num_B, num_C):
        """
        Constructs new agents of each type and adds them to the corresponding
            agent list.
        :param num_A: number of agents of species A to construct
        :param num_B: number of agents of species B to construct
        :param num_C: number of agents of species C to construct
        """
        for i in range(num_A):
            self.A_list.append(
                Nonspatial_Agent(health=self.init_healths[0],
                                 hunt_success_prob=0,
                                 food_value=self.food_values[0],
                                 reproduction_prob=self.reproduction_probs[0],
                                 species='A'))
        for j in range(num_B):
            self.B_list.append(
                Nonspatial_Agent(health=self.init_healths[1],
                                 hunt_success_prob=self.hunt_success_probs[0],
                                 food_value=self.food_values[1],
                                 reproduction_prob=self.reproduction_probs[1],
                                 species='B'))
        for k in range(num_C):
            self.C_list.append(
                Nonspatial_Agent(health=self.init_healths[2],
                                 hunt_success_prob=self.hunt_success_probs[1],
                                 food_value=0,
                                 reproduction_prob=self.reproduction_probs[2],
                                 species='C'))

    def iteration(self):
        # Define counters for how many new agents of each species are born
        new_C_count = 0
        new_B_count = 0
        new_A_count = 0
        # Model superpredator life
        for C in self.C_list:
            C.pass_day()
            if C.is_alive:
                if C.reproduce():
                    new_C_count += 1
                # Modeling hunting
                for B in self.B_list:
                    if B.is_alive and C.hunt():
                        C.feed(B.food_value)
                        B.die()
                for A in self.A_list:
                    if A.is_alive and C.hunt():
                        C.feed(A.food_value)
                        A.die()
        # Model predator life
        for B in self.B_list:
            B.pass_day()
            if B.is_alive:
                if B.reproduce():
                    new_B_count += 1
                # Modeling hunting
                for A in self.A_list:
                    if A.is_alive and B.hunt():
                        B.feed(A.food_value)
                        A.die()
        # Model prey life
        for A in self.A_list:
            A.pass_day()
            if A.is_alive and A.reproduce():
                new_A_count += 1
            if (np.random.rand() < 0.5
                    and len(self.A_list) < self.carrying_capacity_A
                    and A.reproduce()):
                new_A_count += 1
        # Remove dead agents, from all species lists
        self.remove_dead_agents()
        # Add newly birthed agents
        self.add_new_agents(new_A_count, new_B_count, new_C_count)
        # Add new population totals to tracking lists
        self.num_A_n.append(len(self.A_list))
        self.num_B_n.append(len(self.B_list))
        self.num_C_n.append(len(self.C_list))

    def remove_dead_agents(self):
        """
        Removes the dead agents from all three lists of agents.
        """
        for A in self.A_list:
            if not A.is_alive:
                self.A_list.remove(A)
        for B in self.B_list:
            if not B.is_alive:
                self.B_list.remove(B)
        for C in self.C_list:
            if not C.is_alive:
                self.C_list.remove(C)

    def plot(self, title, filename=None):
        """
        Creates a plot of populations of the three species of over the
            iterations, and saves the figure under the given name.
            With no given filename, this displays the plot.

        :param title: String title to give to created plot
        :param filename: String filename to save finished plot, as to be passed
            to matplotlib.pyplot.savefig(...). Must include the file-extension.
        """
        plt.clf()
        plt.figure()
        plt.title(title)
        plt.xlabel("Iteration number n")
        plt.ylabel("Population of Species A, B, C")
        ns = np.arange(0, self.n_steps+1, 1)  # array of iteration numbers
        plt.plot(ns, self.num_A_n, lw=2, ls="-", c='C0',
                 label="Pop. of Prey")
        plt.plot(ns, self.num_B_n, lw=2, ls="-", c='C1',
                 label="Pop. of Predator")
        plt.plot(ns, self.num_C_n, lw=2, ls="-", c='purple',
                 label="Pop. of Superpredator")
        plt.grid(True)
        # Set x-axis to only use integer ticks
        plt.gca().xaxis.get_major_locator().set_params(integer=True)
        plt.legend()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

    def get_population_extremes(self, print_output=False):
        """
        Returns the maximum and minimum values for each population series.
        Can also print results if print_output flag is True.

        :param print_output: boolean, if True, this method prings a string that
            reports the maximums and minimums of each species.
        :return: 6-tuple giving max, min of each species A, B, C respectively,
            and collated (max_A, min_A, max_B, min_B, max_C, min_C).
        """
        max_A = round(np.max(self.num_A_n), 0)
        min_A = round(np.min(self.num_A_n), 0)
        max_B = round(np.max(self.num_B_n), 0)
        min_B = round(np.min(self.num_B_n), 0)
        max_C = round(np.max(self.num_C_n), 0)
        min_C = round(np.min(self.num_C_n), 0)

        if print_output:
            print(f'''
                  Under the  nonspatial  agent model:
                  - Maximum population of Species A (prey) is {max_A}, and the minimum population is {min_A}
                  - Maximum population of Species B (predator) is {max_B}, and the minimum population is {min_B}
                  - Maximum population of Species C (superpredator) is {max_C}, and the minimum population is {min_C}
            ''')
        else:
            return max_A, min_A, max_B, min_B, max_C, min_C

    @property
    def arguments(self):
        """
        Returns the values of all parameters of our model in a
            formatted string for printing.
        """
        arg_str = f'''
        Initial Settings:
        There are two species: A (prey) and B (predator). After {self.N_release} iterations, species C (superpredator) will be introduced.
        Below are the details for each species:
        Species A (prey):
        - Initial population: {self.A0}, with a maximum population of {self.carrying_capacity_A}.
            - Increases by a reproduction rate of {round((self.reproduction_probs[0])*100)}% per iteration
            - Decreases due to:
                - Natural death when individual health equal to 0
                - Hunted by species B (predator) or species C (superpredator) with probabilities {round((self.hunt_success_probs[0])*100,2)}%, {round((self.hunt_success_probs[1])*100,2)}%
        - Each individual of species A starts with a health value of {self.init_healths[0]}
            - Decreases by each iteration until it is 0
        
        Species B (predator):
        - Initial population: {self.B0}
            - Increases by a reproduction rate of {round((self.reproduction_probs[1])*100)}% per iteration
            - Decreases due to:
                - Natural death when individual health equal to 0
                - Hunted by species C (superpredator) with a probability of {round((self.hunt_success_probs[1])*100,2)}%
        - Each individual of species B starts with a health value of {self.init_healths[1]}
            - Increases by hunting one species A, gaining a value of {self.food_values[0]}
            - Decreases by each iteration until it is 0
        
        Species C (superpredator):
        - Initial population: {self.C_N}
            - Increases by a reproduction rate of {round((self.reproduction_probs[2])*100)}% per iteration
            - Decreases due to natural death when individual health equal to 0
        - Each individual of species C starts with a health value of {self.init_healths[2]}
            - Increases by hunting species A and B, gaining values of {self.food_values[0]} for hunting one species A
                    and {self.food_values[1]} for hunting one species B
            - Decreases by each iteration until it is 0
        
        Observations within {self.n_steps} iterations will showcase interactions between these three species under different conditions.
        '''
        return arg_str
    
    @property
    def init_healths(self):
        return self._init_healths

    @property
    def hunt_success_probs(self):
        return self._hunt_success_probs

    @property
    def food_values(self):
        return self._food_values

    @property
    def reproduction_probs(self):
        return self._reproduction_probs

    @property
    def carrying_capacity_A(self):
        return self._carrying_capacity_A

    @property
    def A0(self):
        return self._A0

    @property
    def B0(self):
        return self._B0

    @property
    def N_release(self):
        return self._N_release

    @property
    def C_N(self):
        return self._C_N

    @property
    def n_steps(self):
        return self._n_steps

    @init_healths.setter
    def init_healths(self, initial_healths):
        if len(initial_healths) != 3:
            raise ValueError("init_healths should be a list of 3 numbers")
        for i in range(3):
            if not is_a_number(initial_healths[i]):
                raise TypeError("Each health in init_healths should be either a float or an integer")
        self._init_healths = initial_healths

    @hunt_success_probs.setter
    def hunt_success_probs(self, success_probs):
        if len(success_probs) != 2:
            raise ValueError("hunt_success_probs should be a list of 2 numbers")
        for i in range(2):
            if success_probs[i] < 0 or success_probs[i] > 1:
                raise ValueError("Each probability in hunt_success_probs should be between 0 and 1")
        self._hunt_success_probs = success_probs

    @food_values.setter
    def food_values(self, food_vals):
        if len(food_vals) != 2:
            raise ValueError("food_values should be a list of 2 numbers")
        for i in range(2):
            if not is_a_number(food_vals[i]):
                raise TypeError("Each value in food_values should be either a float or an integer")
            if food_vals[i] < 0:
                raise ValueError("Each value in food_values should be a positive number")
        self._food_values = food_vals

    @reproduction_probs.setter
    def reproduction_probs(self, repo_probs):
        if len(repo_probs) != 3:
            raise ValueError("reproduction_probs should be a list of 3 numbers")
        for i in range(3):
            if repo_probs[i] < 0 or repo_probs[i] > 1:
                raise ValueError("Each probability in reproduction_probs should be between 0 and 1")
        self._reproduction_probs = repo_probs
    @carrying_capacity_A.setter
    def carrying_capacity_A(self, k_A):
        if not isinstance(k_A, int):
            raise TypeError("The carrying capacity should be an integer")
        if k_A < 0:
            raise ValueError("The carrying capacity should be non negative")
        self._carrying_capacity_A = k_A

    @A0.setter
    def A0(self, A_pop):
        if not isinstance(A_pop, int):
            raise TypeError("The initial population of species A, A0, must be an integer")
        if A_pop < 0:
            raise ValueError("The initial population of species A, A0, must be non-negative")
        self._A0 = A_pop

    @B0.setter
    def B0(self, B_pop):
        if not isinstance(B_pop, int):
            raise TypeError("The initial population of species B, B0, must be an integer")
        if B_pop < 0:
            raise ValueError("The initial population of species B, B0, must be non-negative")
        self._B0 = B_pop

    @N_release.setter
    def N_release(self, iter_of_release):
        if not isinstance(iter_of_release, int):
            raise TypeError("N_release must be an integer")
        if iter_of_release < 0 or iter_of_release > self._n_steps:
            raise ValueError("N_release must be between 0 and n_steps")
        self._N_release = iter_of_release

    @C_N.setter
    def C_N(self, superpredator_pop):
        if not isinstance(superpredator_pop, int):
            raise TypeError("The population of superpredators at release, C_N, must be an integer")
        if superpredator_pop < 0:
            raise ValueError("The population of superpredators at release, C_N, must be non-negative")
        self._C_N = superpredator_pop

    @n_steps.setter
    def n_steps(self, n_iter):
        if not isinstance(n_iter, int):
            raise TypeError("n_steps must be an integer")
        if n_iter < 0:
            raise ValueError("n_steps must non negative")
        self._n_steps = n_iter
