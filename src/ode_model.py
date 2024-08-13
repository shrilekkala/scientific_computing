import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from src.tools import is_a_number


class ODE_Superpredator_Release:
    """
    This class models a situation in which we have a predator and prey
    species that exist for some time in the environment, before we introduce
    a population of superpredators that hunt both existing species.

    In this file, we look into modelling this system using ODEs,
    with modified Lotka-Volterra equations:
    Let A be our prey species, B be the predator species,
        and C be the super predator species.
    Our equations thus become:
    dA/dt = r[0]*A*(1-A/K) - mu[0]*A*B - nu[0]*A*C
    dB/dt = -r[1]*B + mu[1]*A*B - eta[0]*B*C
    dC/dt = -r[2]*C + nu[1]*A*C + eta[1]*B*C

    Instance Variables (for the Lotka-Volterra equations):
    ***************************************************************************
    r = 3-list of absolute growth rate for A, and absolute death rates of B
        and C. All values should be positive, representing the number each
        species that are born or die in some timestep, disregarding
        interactions between species.
    K = carrying capacity of the environment for species A, i.e., the maximum
        population of A that still has a positive growth rate (ignoring
        interspecies dynamics)
    mu = 2-list of rate at which A is hunted by B (in first coordinate)
        and the growth rate in B from feeding on A (in the second coordinate).
    nu = 2-list of rate at which A is hunted by C (in first coordinate)
        and the growth rate in C from feeding on A (in the second coordinate).
    eta = 2-list of rate at which B is hunted by C (in first coordinate)
        and the growth rate in C from feeding on B (in the second coordinate).
    t_bounds = length 2 list, giving start time and ending time on which to
        solve the ODE.
    A0 = initial population of species A.
    B0 = initial population of species B.
    T_release = time at which predator population is released.
    C_T = population of superpredators to release at T_release.

    Instance Variables (for the solver):
    ***************************************************************************
    f = vector-valued function that computes [dA/dt, dB/dt, dC/dt]
    t_arr = NumPy array of times the population is given by the ODE solver.
    A_t = NumPy array, timeseries giving population of A at each time step
    B_t = NumPy array, timeseries giving population of B at each time step
    C_t = NumPy array, timeseries giving population of C at each time step
    """
    def __init__(self, r, K, mu, nu, eta, t_bounds, A0, B0, T_release, C_T):
        """
        Initializes a new ODE model with given parameters and timespan.
            This object can later be used for plotting, accessing the
            population over time, etc.

        :param r: 3-list of absolute growth rate for A, and absolute death
            rates of B and C. All values should be positive, representing the
            number each species that are born or die in some timestep,
            disregarding interactions between species.
        :param K: carrying capacity of the environment for species A, i.e., the
            maximum population of A that still has a positive growth rate
            (ignoring interspecies dynamics)
        :param mu: 2-list of rate at which A is hunted by B (in first
            coordinate) and the growth rate in B from feeding on A (in the
            second coordinate).
        :param nu: 2-list of rate at which A is hunted by C (in first
            coordinate) and the growth rate in C from feeding on A (in the
            second coordinate).
        :param eta: 2-list of rate at which B is hunted by C (in first
            coordinate) and the growth rate in C from feeding on B (in the
            second coordinate).
        :param t_bounds: length 2 list, giving start time and ending time on
            which to solve the ODE.
        :param A0: initial population of species A.
        :param B0: initial population of species B.
        :param T_release: time at which predator population is released.
        :param C_T: population of superpredators to release at T_release.
        """

        # Assigning class variables
        self.r = r
        self.K = K
        self.mu = mu
        self.nu = nu
        self.eta = eta
        self.t_bounds = t_bounds
        self.A0 = A0
        self.B0 = B0
        self.T_release = T_release
        self.C_T = C_T

        # Create and save function that computes [dA/dt, dB/dt, dC/dt]
        def f(t, y, r, K, mu, nu, eta):
            A, B, C = y
            dAdt = r[0]*A*(1-A/K) - mu[0]*A*B - nu[0]*A*C
            dBdt = -r[1]*B + mu[1]*A*B - eta[0]*B*C
            dCdt = -r[2]*C + nu[1]*A*C + eta[1]*B*C
            # Slight improvement to Lotka-Volterra behavior, so that we avoid
            #    "resurgence" where almost extinct species grow in population
            #    from a population of less than 1.
            if A < 1:
                dAdt = -r[0]*A
            if B < 1:
                dBdt = -r[1]*B
            if C < 1:
                dCdt = -r[2]*C
            return [dAdt, dBdt, dCdt]
        self.f = f
        # Run the ODE solver from the initial time up until superpredators
        # would be released.
        t_span0 = [self.t_bounds[0], T_release]
        max_step0 = np.diff(t_span0)[0] / 100
        arg_tuple = (self.r, self.K, self.mu, self.nu, self.eta)
        sol_before_release = solve_ivp(self.f,
                                       t_span0,
                                       [self.A0, self.B0, 0],
                                       args=arg_tuple,
                                       max_step=max_step0)
        # Take this population at T_release and add the
        # superpredator population
        population_at_release = sol_before_release.y[:, -1]
        population_at_release[2] = population_at_release[2] + C_T
        # Run the solver from T_release
        t_span1 = [T_release, self.t_bounds[1]]
        max_step1 = np.diff(t_span1)[0] / 100
        sol_after_release = solve_ivp(self.f,
                                      t_span1,
                                      population_at_release,
                                      args=arg_tuple,
                                      max_step=max_step1)
        self.t_arr = np.append(sol_before_release.t, sol_after_release.t)
        self.A_t = np.append(sol_before_release.y[0, :], sol_after_release.y[0, :])
        self.B_t = np.append(sol_before_release.y[1, :], sol_after_release.y[1, :])
        self.C_t = np.append(sol_before_release.y[2, :], sol_after_release.y[2, :])

    def plot(self, title, filename=None):
        """
        Creates a time-plot of populations of the three species of overtime,
            and saves the figure under the given name.
            With no given filename, this displays the plot.

        :param title: String title to give to created plot
        :param filename: String filename to save finished plot, as to be passed
            to matplotlib.pyplot.savefig(...). Must include the file-extension.
        """
        plt.clf()
        plt.figure()
        plt.title(title)
        plt.xlabel("Time t")
        plt.ylabel("Population of Species A, B, C")
        plt.plot(self.t_arr, self.A_t, lw=2, ls="-", c='C0',
                 label="Pop. of Prey")
        plt.plot(self.t_arr, self.B_t, lw=2, ls="-", c='C1',
                 label="Pop. of Predator")
        plt.plot(self.t_arr, self.C_t, lw=2, ls="-", c='purple',
                 label="Pop. of Superpredator")
        plt.grid(True)
        plt.legend()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

    @property
    def timespan(self):
        """
        Returns a NumPy array of all time values that populations are
            computed at in our numerical ODE solution.
        """
        return self.t_arr

    @property
    def population_timeseries(self):
        """
        Returns a tuple of 3 NumPy arrays, containing the populations of each
            species over the timespan, as found by the numerical ODE solution.
        """
        return self.A_t, self.B_t, self.C_t

    @property
    def final_populations(self):
        """
        Returns a 3-tuple, giving the final populations at
            the conclusion of the timespan.
        """
        return self.A_t[-1], self.B_t[-1], self.C_t[-1]

    @property
    def arguments(self):
        """
        Returns the values of all parameters of our model in a
            formatted string for printing.
        """
        t_bounds = self.t_bounds
        arg_str = f'''
        Initial Settings:
        From time {t_bounds[0]} until time {t_bounds[1]}, we have two species: A and B, representing prey and predator respectively.
        Below are their details:
        Species A (prey):
            - Initial population of species A (prey): {self.A0}, with a maximum population reaching {self.K}.
            - Growth rate: {round(self.r[0]*100,2)}%.
            - Death rate from being preyed upon by other species: {round((self.mu[0] + self.nu[0])*100,2)}%, can be divided into two parts:
                - Death rate from being preyed upon by predator B: {round(self.mu[0]*100,2)}%.
                - Death rate from being preyed upon by predator C: {round(self.nu[0]*100,2)}%, which is 0 before time {self.T_release}.
        Species B (predator):
            - Initial population of species B (predator): {self.B0}, with a natural death rate of {round(self.r[1]*100,2)}%.
            - Growth rate from feeding on prey A: {round(self.mu[1]*100,2)}%.
            - Death rate from being preyed upon by other species: {round(self.eta[0]*100,2)}%.
        
        After time {self.T_release}, superpredators are released into the environment. Here is the related information about superpredator C:
            - Number of species C (superpredators) released: {self.C_T}, which is 0 before time {self.T_release}.
            - Death rate: {round(self.r[2]*100,2)}%.
            - Growth rate from feeding on other species: {round((self.nu[1] + self.eta[1])*100,2)}%, can be divided into two parts:
                - Growth rate of species C (superpredators) from feeding on species A: {round(self.nu[1]*100,2)}%.
                - Growth rate of species C (superpredators) from feeding on species B: {round(self.eta[1]*100,2)}%.
        
        In the end, the population of each species is:
        Species A (prey): {round(self.A_t[-1],2)}
        Species B (predator): {round(self.B_t[-1],2)}
        Species C (superpredators): {round(self.C_t[-1],2)}
        '''
        return arg_str

    @property
    def equation_string(self):
        """
        Returns a string representation of them Lotka-Volterra equations, to
            be displayed.
        """
        return '''
            dA/dt = r[0]*A*(1-A/K) - mu[0]*A*B - nu[0]*A*C
            dB/dt = -r[1]*B + mu[1]*A*B - eta[0]*B*C
            dC/dt = -r[2]*C + nu[1]*A*C + eta[1]*B*C
        '''

    def get_population_extremes(self, print_output=False):
        """
        Returns the maximum and minimum values for each population series.
        Can also print results if print_output flag is True.

        :param print_output: boolean, if True, this method prings a string that
            reports the maximums and minimums of each species.
        :return: 6-tuple giving max, min of each species A, B, C respectively,
            and collated (max_A, min_A, max_B, min_B, max_C, min_C).
        """
        max_A = round(np.max(self.A_t),0) 
        min_A = round(np.min(self.A_t),0)
        max_B = round(np.max(self.B_t),0)
        min_B = round(np.min(self.B_t),0)
        max_C = round(np.max(self.C_t),0)
        min_C = round(np.min(self.C_t),0)

        if print_output:
            print(f'''
                  Under the ODE model:
                  - Maximum population of Species A (prey) can reach is {max_A}, and the minimum population is {min_A}
                  - Maximum population of Species B (predator) can reach is {max_B}, and the minimum population is {min_B}
                  - Maximum population of Species C (superpredators) can reach is {max_C}, and the minimum population is {min_C}
            ''')
        else:
            return max_A, min_A, max_B, min_B, max_C, min_C

    @property
    def r(self):
        return self._r

    @property
    def K(self):
        return self._K

    @property
    def mu(self):
        return self._mu

    @property
    def nu(self):
        return self._nu

    @property
    def eta(self):
        return self._eta

    @property
    def t_bounds(self):
        return self._t_bounds

    @property
    def A0(self):
        return self._A0

    @property
    def B0(self):
        return self._B0

    @property
    def T_release(self):
        return self._T_release

    @property
    def C_T(self):
        return self._C_T

    @r.setter
    def r(self, rates):
        if len(rates) != 3:
            # check if list has length 3
            raise ValueError("r should be a list of 3 numbers")
        for i in range(3):
            if not is_a_number(rates[i]):
                raise TypeError("Each rate in r should be either a float or an integer")
        self._r = rates

    @K.setter
    def K(self, capacity):
        if not isinstance(capacity, int):
            raise TypeError("The carrying capacity, K, should be an integer")
        if capacity < 0:
            raise ValueError("The carrying capacity, K, should be non negative")
        self._K = capacity

    @mu.setter
    def mu(self, mu_rate):
        if len(mu_rate) != 2:
            # check if list has length 2
            raise ValueError("mu should be a list of 2 numbers")
        for i in range(2):
            if not is_a_number(mu_rate[i]):
                raise TypeError("Each rate in mu should be either a float or an integer")
        self._mu = mu_rate

    @nu.setter
    def nu(self, nu_rate):
        if len(nu_rate) != 2:
            # check if list has length 2
            raise ValueError("nu should be a list of 2 numbers")
        for i in range(2):
            if not is_a_number(nu_rate[i]):
                raise TypeError("Each rate in nu should be either a float or an integer")
        self._nu = nu_rate

    @eta.setter
    def eta(self, eta_rate):
        if len(eta_rate) != 2:
            # check if list has length 2
            raise ValueError("eta should be a list of 2 numbers")
        for i in range(2):
            if not is_a_number(eta_rate[i]):
                raise TypeError("Each rate in eta should be either a float or an integer")
        self._eta = eta_rate

    @t_bounds.setter
    def t_bounds(self, time_bounds):
        if len(time_bounds) != 2:
            # check if list has length 2
            raise ValueError("t_bounds should be a list of 2 numbers")
        for i in range(2):
            if not is_a_number(time_bounds[i]):
                raise TypeError("Each time in t_bounds should be either a float or an integer")
        self._t_bounds = time_bounds

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

    @T_release.setter
    def T_release(self, release_time):
        if not is_a_number(release_time):
            raise TypeError("T_release must be a float or an integer")
        if release_time < self.t_bounds[0] or release_time > self.t_bounds[1]:
            raise ValueError("T_release must be between the start and end time of the ODE")
        self._T_release = release_time

    @C_T.setter
    def C_T(self, super_pop):
        if not isinstance(super_pop, int):
            raise TypeError("The initial population of superpredators, C_T, must be an integer")
        if super_pop < 0:
            raise ValueError("The initial population of superpredators, C_T, must be non-negative")
        self._C_T = super_pop
