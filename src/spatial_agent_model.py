import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from src.tools import is_a_number


class Spatial_Agent_Model:
    """
    This class models the same superpredator release situation, but with an
        spatial, agent-based model instead. Each agent is only able to interact
        (e.g. breed and hunt) with the agents in cells adjacent (orthogonally
        and diagonally) to them.

    Instance Variables (initialized by the user):
    *************************************************
    breeding_counts = 3-list of corresponding *exact* number of
        neighbors of the same species required for a new agent to be born.
    minimum_counts = 3-list of minimum population of neighbors of the
            same species for a cell to survive.
    overpopulation_counts = 3-list of corresponding number of
        neighbors of the same species that causes overpopulation and death
        for the center cell.
    hunt_success_probs = 2-list of probability that predator and
        superpredator kills given prey during hunt-phase.
    A0 = initial population of species A.
    B0 = initial population of species B.
    N_release = interation number at which superpredators are released
    C_N = population of superpredators to release at the
        (N_release)th iteration
    n_steps = integer number of iterations steps to compute.
    m = Number of rows in discretized spatial grid of agent positions
    n = Number of columns in discretized spatial grid of agent positions
    animation_length = length of the animation of the model to be
        created, in seconds. Default is 10 seconds.

    Instance Variables (created by model simulator):
    ****************************************************
    grid = (m x n) NumPy array of integers 0, 1, 2, 3, representing each agent
        as a single cell, where 0 indicates an empty cell, and 1,2,3 represent
        prey, predator, and superpredator respectively.
    im = output of plt.imshow(), used for animation tracking in FuncAnimation
    n_wait =
    animation =
    num_A_n = list of population of species A at each iteration
    num_B_n = list of population of species B at each iteration
    num_C_n = list of population of species B at each iteration
    """
    def __init__(self, breeding_counts, minimum_counts, overpopulation_counts,
                 hunt_success_probs, A0, B0, N_release,
                 C_N, n_steps, m, n, animation_length=10):
        """
        Initalized the non-spatial model with given parameters and
            initial state

        :param breeding_counts: 3-list of corresponding *exact* number of
            neighbors of the same species required for a new agent to be born.
        :param minimum_counts: 3-list of minimum population of neighbors of the
            same species for a cell to survive.
        :param overpopulation_counts: 3-list of corresponding number of
            neighbors of the same species that causes overpopulation and death
            for the center cell.
        :param hunt_success_probs: 2-list of probability that predator and
            superpredator kills given prey during hunt-phase.
        :param A0: initial population of species A.
        :param B0: initial population of species B.
        :param N_release: interation number at which superpredators are
            released
        :param C_N: population of superpredators to release at the
            (N_release)th iteration
        :param n_steps: integer number of iterations steps to compute.
        :param m: Number of rows in discretized spatial grid of agent positions
        :param n: Number of columns in discretized spatial grid of agent
            positions
        :param animation_length: length of the animation of the model to be
            created, in seconds. Default is 10 seconds.
        """
        # Store all results as instance variables
        self.breeding_counts = breeding_counts
        self.minimum_counts = minimum_counts
        self.overpopulation_counts = overpopulation_counts
        self.hunt_success_probs = hunt_success_probs
        self.A0 = A0
        self.B0 = B0
        self.n_steps = n_steps
        self.N_release = N_release
        self.C_N = C_N
        self.m = m
        self.n = n
        self.animation_length = animation_length

        '''
        We model the grid of agents in this case as a NumPy 2D array of
            integers 0, 1, 2, 3. Where empty cells are 0's, prey are 1's,
            predators are 2's, and superpredators are 3's.
        '''
        self.grid = np.zeros((m, n), dtype=np.uint8)
        self.num_A_n = [self.A0]
        self.num_B_n = [self.B0]
        self.num_C_n = [0]
        # Seed randomness so simulation is repeatable.
        np.random.seed(self.A0 + self.B0 + self.C_N)
        # construct starting populations
        self.add_new_agents(self.A0, self.B0, 0)

        # Run iterations through the animation creator
        plt.ioff()
        plt.clf()
        self.fig = plt.figure()
        plt.axis('off')
        cmap = ListedColormap(['lightgray', 'C0', 'C1', 'purple'], N=4)
        patches = [Patch(color='C0', label='Prey'),
                   Patch(color='C1', label='Predator'),
                   Patch(color='purple', label='Superpredator')]
        plt.legend(handles=patches, ncol=3, loc='lower center',
                   bbox_to_anchor=(0.5, -0.08))
        self.im = plt.imshow(self.grid, cmap=cmap, vmin=0, vmax=3,
                             interpolation='none', animated=True)
        # We want to ensure that the animation has some frozen time at
        #     either end.
        pad_length = 2.0  # length (in seconds) of frozen time at either end
        interval = (self.animation_length - 2.0*pad_length) / self.n_steps
        self.n_wait = int(pad_length // interval)
        interval = interval * 1000
        self.animation = FuncAnimation(self.fig, func=self.iteration,
                                       frames=self.n_steps + 2*self.n_wait,
                                       interval=interval, repeat=False)

    def add_new_agents(self, num_A, num_B, num_C):
        """
        Adds new agents to the grid-space to form random "blobs" of agents
            of the same species.

        :param num_A: number of agents of species A to add
        :param num_B: number of agents of species B to add
        :param num_C: number of agents of species C to add
        """
        empties = np.argwhere(self.grid == 0)  # find empty cells
        if len(empties) == 0:
            return
        # Counters for each species of agents created
        count_A = 0
        count_B = 0
        count_C = 0
        # Generate the first "seed" position to start generating agents at
        seed = empties[np.random.randint(len(empties))]
        # Start seed with the correct value, the first type to be added
        if num_A > 0:
            self.grid[tuple(seed)] = 1
            count_A += 1
        elif num_B > 0:
            self.grid[tuple(seed)] = 2
            count_B += 1
        elif num_C > 0:
            self.grid[tuple(seed)] = 3
            count_C += 1
        else:
            return
        # Assign the randomly ordered empty cells to correct number of
        #   new agents. We check that each number is not zero, as grid[()],
        #   (taking the slice of an empty tuple), would assign the whole grid.
        # for k in range(num_A):
        #     self.grid[tuple(empties_ij[k])] = 1
        # for k in range(num_A, num_A+num_B):
        #     self.grid[tuple(empties_ij[k])] = 2
        # for k in range(num_A+num_B, num_A+num_B+num_C):
        #     self.grid[tuple(empties_ij[k])] = 3
        while count_A < num_A:
            # randomly order the empty cells next to seed
            enb = np.random.permutation(self.empty_neighbors(seed[0], seed[1]))
            if len(enb) == 0:  # if we have no empty neighbors, get need seed
                empties = np.argwhere(self.grid == 0)
                if len(empties) == 0:
                    return
                seed = empties[np.random.randint(len(empties))]
                self.grid[tuple(seed)] = 1
                count_A += 1
            else:  # otherwise, make one random neighbor nonempty
                self.grid[tuple(enb[0])] = 1
                count_A += 1
                # Randomly choose to move seed
                # Decreasing this probability causes more "clumping" of agents
                if np.random.rand() < 0.5:
                    seed = enb[0]
        # Select a new seed for predator if needed
        if num_A > 0:
            empties = np.argwhere(self.grid == 0)  # find empty cells
            if len(empties) == 0:
                return
            seed = empties[np.random.randint(len(empties))]
            if num_B > 0:
                self.grid[tuple(seed)] = 2
                count_B += 1
            elif num_C > 0:
                self.grid[tuple(seed)] = 3
                count_C += 1
            else:
                return
        while count_B < num_B:
            # randomly order the empty cells next to seed
            enb = np.random.permutation(self.empty_neighbors(seed[0], seed[1]))
            if len(enb) == 0:  # if we have no empty neighbors, get need seed
                empties = np.argwhere(self.grid == 0)
                if len(empties) == 0:
                    return
                seed = empties[np.random.randint(len(empties))]
                self.grid[tuple(seed)] = 2
                count_B += 1
            else:  # otherwise, make one random neighbor nonempty
                self.grid[tuple(enb[0])] = 2
                count_B += 1
                # Randomly choose to move seed
                # Decreasing this probability causes more "clumping" of agents
                if np.random.rand() < 0.5:
                    seed = enb[0]
        # Select a new seed for predator if needed
        if num_A > 0 or num_B > 0:
            empties = np.argwhere(self.grid == 0)  # find empty cells
            if len(empties) == 0:
                return
            seed = empties[np.random.randint(len(empties))]
            if num_C > 0:
                self.grid[tuple(seed)] = 3
                count_C += 1
            else:
                return
        while count_C < num_C:
            # randomly order the empty cells next to seed
            enb = np.random.permutation(self.empty_neighbors(seed[0], seed[1]))
            if len(enb) == 0:  # if we have no empty neighbors, get need seed
                empties = np.argwhere(self.grid == 0)
                if len(empties) == 0:
                    return
                seed = empties[np.random.randint(len(empties))]
                self.grid[tuple(seed)] = 3
                count_C += 1
            else:  # otherwise, make one random neighbor nonempty
                self.grid[tuple(enb[0])] = 3
                count_C += 1
                # Randomly choose to move seed
                # Decreasing this probability causes more "clumping" of agents
                if np.random.rand() < 0.9:
                    seed = enb[0]

    def iteration(self, frame):
        """
        Function called implicitly by FuncAnimation to compute next step.

        :param frame: iteration number
        :return: plt.imshow() result after updating data in grid
        """
        if frame < self.n_wait or frame >= self.n_steps + self.n_wait:
            self.im.set_array(self.grid)
            return self.im
        # We are off-by-one since frame is zero indexed but N_release is
        #     1-indexed.
        if frame == self.n_wait + self.N_release - 1:
            # Construct released superpredator agents
            self.add_new_agents(0, 0, self.C_N)
        # Run the actual model update
        # Make a new grid so that we can take all calculations simultaneously,
        #   instead of having some cells updated before others.
        new_grid = np.zeros((self.m, self.n), dtype=np.uint8)
        # Next, compute the reproduction and overpopulation changes
        for i in range(self.m):
            for j in range(self.n):
                species = self.grid[i, j]
                if species == 0:
                    counts = self.count_all(i, j)
                    for k, count in enumerate(counts):
                        # Check if each species can reproduce at the empty cell
                        if count == self.breeding_counts[k]:
                            new_grid[i, j] = k+1
                        # We allow new_grid[i,j] to be overwritten, under the
                        # assumption that the superpredators would win out if
                        # multiple species had the opportunity to reproduce.
                else:
                    count = self.count_species(i, j, species)
                    # copy the agent to next grid if we are not overpopulated
                    if (self.minimum_counts[species-1] <= count <
                            self.overpopulation_counts[species-1]):
                        new_grid[i, j] = species

        # First, compute hunting behaviors
        C_coords = np.argwhere(self.grid == 3)
        B_coords = np.argwhere(self.grid == 2)
        for C_ij in C_coords:
            c_i, c_j = tuple(C_ij)
            # Iterate over neighbors
            for i2, j2 in neighbors(c_i, c_j, self.m, self.n):
                # if either of the available prey for C is a neighbor,
                if self.grid[i2, j2] == 1 or self.grid[i2, j2] == 2:
                    # Check if C kills that prey, and make its cell empty if so
                    if np.random.rand() < self.hunt_success_probs[1]:
                        new_grid[i2, j2] = 0
        for B_ij in B_coords:
            b_i, b_j = tuple(B_ij)
            # Iterate over neighbors
            for i2, j2 in neighbors(b_i, b_j, self.m, self.n):
                # if any neighbors are the prey A for B, kill that cell
                if (self.grid[i2, j2] == 1
                        and np.random.rand() < self.hunt_success_probs[
                            0]):
                    new_grid[i2, j2] = 0
        self.grid = new_grid  # copy over new grid
        # Append population counts
        self.num_A_n.append(np.sum(self.grid == 1))
        self.num_B_n.append(np.sum(self.grid == 2))
        self.num_C_n.append(np.sum(self.grid == 3))
        # Update animation
        self.im.set_array(self.grid)
        return self.im,

    def count_species(self, i, j, species):
        """
        Counts the number of adjacent cells that contain the desired species.

        :param i: Row-position of cell to focus around
        :param j: Column-position of cell to focus around
        :param species: Species to track occurrences of in neighbors,
            integer 1, 2, or 3.
        :return: Number of agents of species in the 8 (or fewer) cells
            surrounding [i, j]
        """
        count = 0
        for i2, j2 in neighbors(i, j, self.m, self.n):
            count += self.grid[i2, j2] == species
        return count

    def count_all(self, i, j):
        """
        Counts the number of adjacent cells that contain the desired species.

        :param i: Row-position of cell to focus around
        :param j: Column-position of cell to focus around
        :return: 3-tuple of integers, number of agents of species A, B, and C,
            respectively surrounding [i, j]
        """
        count_A = 0
        count_B = 0
        count_C = 0
        for i2, j2 in neighbors(i, j, self.m, self.n):
            count_A += self.grid[i2, j2] == 1
            count_B += self.grid[i2, j2] == 2
            count_C += self.grid[i2, j2] == 3
        return count_A, count_B, count_C

    def empty_neighbors(self, i, j):
        """
        Returns a NumPy array of coordinate pairs (tuples) that are empty cells
            that surround [i, j] in the spatial grid.
        """
        empties = []
        for i2, j2 in neighbors(i, j, self.m, self.n):
            if self.grid[i2, j2] == 0:
                empties.append((i2, j2))
        return np.array(empties)

    def save_gif(self, filename):
        """
        Saves a gif of the spatial model.

        :param filename: string, name of the file to create, must include the
            extension '.gif'
        """
        self.animation.save(filename)
        plt.close()

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
        plt.ion()
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
        max_A = max(self.num_A_n)
        min_A = min(self.num_A_n)
        max_B = max(self.num_B_n)
        min_B = min(self.num_B_n)
        max_C = max(self.num_C_n)
        min_C = min(self.num_C_n)
        if print_output:
            print(f'''
                  Under the spatial agent model:
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
        argument_str = f'''
        Initial Settings:
        There are two species, A (prey), and B (predator), in the {self.m} * {self.n} discretized spatial grid. After {self.N_release} iterations,
            species C (superpredator) will be introduced.
        In this model, breeding and hunting only occur when these species are in nearby cells.
        Below are the details of each species:
        Species A (prey):
        - Initial population: {self.A0}
            - Increases when the number of surrounding cells is up to {self.breeding_counts[0]}.
            - Decreases due to:
                - Exceeding environmental capacity: when the number of surrounding cells is up to {self.overpopulation_counts[0]}.
                - Predation by predators: hunted by species B (predator) with a probability of {round((self.hunt_success_probs[0])*100,2)}% when
                                                they are in adjacent cells,
                                          hunted by species C (superpredator) with a probability of {round((self.hunt_success_probs[1])*100,2)}% when
                                                they are in adjacent cells.
        
        Species B (predator):
        - Initial population: {self.B0}
            - Increases when the number of surrounding cells reaches {self.breeding_counts[1]}.
            - Decreases due to:
                - Exceeding environmental capacity: when the number of surrounding cells reaches {self.overpopulation_counts[1]}.
                - Predation by predators: hunted by species C (superpredator) with a probability of {round((self.hunt_success_probs[1])*100,2)}% when
                    they are in adjacent cells.
        
        Species C (superpredator):
        - Initial population: {self.C_N}, introduced at {self.N_release} iteration.
            - Increases when the number of surrounding cells reaches {self.breeding_counts[2]}.
            - Decreases when the number of surrounding cells is up to {self.overpopulation_counts[2]}.
    
        Observations within {self.n_steps} iterations will showcase interactions between these three species under the spatial model.
        '''
        
        return argument_str

    @property
    def breeding_counts(self):
        return self._breeding_counts

    @property
    def minimum_counts(self):
        return self._minimum_counts

    @property
    def overpopulation_counts(self):
        return self._overpopulation_counts

    @property
    def hunt_success_probs(self):
        return self._hunt_success_probs

    @property
    def A0(self):
        return self._A0

    @property
    def B0(self):
        return self._B0

    @property
    def n_steps(self):
        return self._n_steps

    @property
    def N_release(self):
        return self._N_release

    @property
    def C_N(self):
        return self._C_N

    @property
    def m(self):
        return self._m

    @property
    def n(self):
        return self._n

    @property
    def animation_length(self):
        return self._animation_length

    @breeding_counts.setter
    def breeding_counts(self, breeding_cts):
        if len(breeding_cts) != 3:
            # check if list has length 3
            raise ValueError("breeding_counts should be a list of 3 numbers")
        for i in range(3):
            if not isinstance(breeding_cts[i], int):
                raise TypeError("Each count in list should be an integer")
            if breeding_cts[i] < 0:
                raise ValueError("Each count should be non-negative")
        self._breeding_counts = breeding_cts

    @minimum_counts.setter
    def minimum_counts(self, min_cts):
        if len(min_cts) != 3:
            # check if list has length 3
            raise ValueError("minimum_counts should be a list of 3 numbers")
        for i in range(3):
            if not isinstance(min_cts[i], int):
                raise TypeError("Each count in list should be an integer")
            if min_cts[i] < 0:
                raise ValueError("Each count should be non-negative")
        self._minimum_counts = min_cts

    @overpopulation_counts.setter
    def overpopulation_counts(self, overpop_cts):
        if len(overpop_cts) != 3:
            # check if list has length 3
            raise ValueError("overpopulation_counts should be a list of 3 numbers")
        for i in range(3):
            if not isinstance(overpop_cts[i], int):
                raise TypeError("Each count in list should be an integer")
            if overpop_cts[i] < 0:
                raise ValueError("Each count should be non-negative")
        self._overpopulation_counts = overpop_cts

    @hunt_success_probs.setter
    def hunt_success_probs(self, success_probs):
        if len(success_probs) != 2:
            raise ValueError("hunt_success_probs should be a list of 2 numbers")
        for i in range(2):
            if success_probs[i] < 0 or success_probs[i] > 1:
                raise ValueError("Each probability in hunt_success_probs should be between 0 and 1")
        self._hunt_success_probs = success_probs

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

    @n_steps.setter
    def n_steps(self, n_iter):
        if not isinstance(n_iter, int):
            raise TypeError("n_steps must be an integer")
        if n_iter < 0:
            raise ValueError("n_steps must non negative")
        self._n_steps = n_iter
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

    @m.setter
    def m(self, num_spatial_rows):
        if not isinstance(num_spatial_rows, int):
            raise TypeError("The number of rows of the spatial grid must be an integer")
        if num_spatial_rows < 0:
            raise ValueError("The number of rows of the spatial grid must be non-negative")
        self._m = num_spatial_rows

    @n.setter
    def n(self, num_spatial_cols):
        if not isinstance(num_spatial_cols, int):
            raise TypeError("The number of columns of the spatial grid must be an integer")
        if num_spatial_cols < 0:
            raise ValueError("The number of columns of the spatial grid must be non-negative")
        self._n = num_spatial_cols

    @animation_length.setter
    def animation_length(self, t):
        if not is_a_number(t):
            raise TypeError("The animation length time in seconds must be a float or an integer")
        if t < 0:
            raise ValueError("The animation length time in seconds must be non-negative")
        self._animation_length = t


def neighbors(i, j, m, n):
    """
    Generator function to yield coordinates of the neighbors of a given
        position [i, j] in an mxn grid. This includes the diagonals, and
        orthogonal cells, but does not include the center cell.
    This code is adapted from the course notes:
    https://uchi-compy23.github.io/notes/09_computing/agent_based_models.html

    :param i, j: Position of cell to find neighbors of, [i, j]. This uses array
        indexing as coordinates, so [0, 0] is the top-left cell
    :param m, n: Size of grid to find neighbors on, for edge consideration
        assumed to be an array of np.shape -> (m, n)
        Assumes m>1, n>1.
    :return: Generator for use in for loop with two outputs -- i2, j2 --
        that iterate through the 8 (or fewer neighbors of a cell at [i, j].
    """
    inbrs = [-1, 0, 1]
    if i == 0:  # checking left-and-right edge conditions
        inbrs.remove(-1)
    elif i == m-1:
        inbrs.remove(1)
    jnbrs = [-1, 0, 1]
    if j == 0:  # checking top-and-bottom edge conditions
        jnbrs.remove(-1)
    elif j == n-1:
        jnbrs.remove(1)
    # Looping over valid neighbors and yielding coordinate pairs
    for delta_i in inbrs:
        for delta_j in jnbrs:
            if delta_i == delta_j == 0:
                continue  # do not include the cell itself
            # otherwise, return the pair that gives index-pair of one neighbor
            yield i + delta_i, j + delta_j
