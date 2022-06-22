r"""Uses phenotype to determine fitness"""

class Fitness:
    r"""Class to determine fitness from phenotype for a sequence

    Args:
        delta_log_KD: predicted ∆log10_KD produced from torchdms model
    """

    def __init__(self, delta_log_KD: float):
        self.delta_log_KD = delta_log_KD
        self.antigen_bound = None
        self.fitness = None


    def frac_antigen_bound(self, log10_naive_KD: float, concentration_antigen: float):
        """Uses the KD and concentration of antigen to calculate the fraction antigen bound via the Hill equation"""
        log10_KD  = self.delta_log_KD + log10_naive_KD
        KD = 10**log10_KD
        # Hill equation with n = 1:
        theta = concentration_antigen/(KD + concentration_antigen)
        self.antigen_bound = theta

    def linear_fitness(self, k: float, c: float, log10_naive_KD: float, concentration_antigen: float):
        """Combines methods to get the antigen bound, Tfh help, and fitness from the KD using a linear model"""
        self.antigen_bound = self.frac_antigen_bound(log10_naive_KD, concentration_antigen)
        Tfh_help = self.antigen_bound_Tfh_help_linear(k)
        self.fitness = self.fitness_from_Tfh_help(Tfh_help, c)

    def sigmoidal_fitness(self, k: float, α: float, β: float, log10_naive_KD: float, concentration_antigen: float):
        """Combines methods to get the antigen bound, Tfh help, and fitness from the KD using a sigmoidal model"""
        self.antigen_bound = self.frac_antigen_bound(log10_naive_KD, concentration_antigen)
        Tfh_help = self.antigen_bound_Tfh_help_sigmoid(k, α, β)
        self.fitness = self.fitness_from_Tfh_help(Tfh_help, c)

    def fitness(self, mapping_type: str = 'linear'):
        r"""Maps evaluation to unnormalized fitness
        Args:
            mapping_type: type of mapping function (defaults to linear)
        """
        # TODO: variable-len parameters, finish calculating fitness
        if mapping_type == 'linear':
            self.fitness = None
        elif mapping_type == 'sigmoid':
            self.fitness = None
        else:
            raise Exception('Only linear and sigmoid are acceptable mapping types')

    def antigen_bound_Tfh_help_sigmoid(self, k: float, α: float, β: float):
        """Produce a transformation from antigen bound to Tfh help using parameters k, alpha, and beta"""
        x = α * (antigen_bound - β)
        Tfh = k/(1 + exp(-1 * x))
        return Tfh

    def antigen_bound_Tfh_help_linear(self, k: float):
        """Produce a transformation from antigen bound to Tfh help using slope k"""
        Tfh = antigen_bound * k
        return Tfh


    def fitness_from_Tfh_help(Tfh_help: float, c: float):
        """Produce a linear transformation from antigen bound to fitness using coefficient c"""
        return c*Tfh_help


    def map_antigen_bound(antigen_bound_fracs: list[float]):
        """Map a list of antigen bound values to fitnesses using the sigmoidal transformation"""
        fitnesses = []
        for antigen_bound_frac in antigen_bound_fracs:
            Tfh_help = antigen_bound_Tfh_help_sigmoid(antigen_bound_frac, k, α, β)
            fitnesses.append(fitness_from_Tfh_help(Tfh_help, c))
        return fitnesses

    def normalize_fitness(fitness_df):
        """Normalize fitness from a dataframe with a fitness column using a min-max approach"""
        min_fitness = fitness_df['fitness'].min()
        max_fitness = fitness_df['fitness'].max()
        normalized_fitness_df = fitness_df.copy()
        normalized_fitness_df['normalized_fitness'] = (fitness_df['fitness'] - min_fitness) / (max_fitness - min_fitness)
        return normalized_fitness_df


    def map_cell_divisions(normalized_fitness_df, m: float):
        """Map fitness linearly to the number of cell divisions using coefficient m"""
        cell_divisions_df = normalized_fitness_df.copy()
        cell_divisions_df['cell_divisions'] = normalized_fitness_df['normalized_fitness'] * m
        return cell_divisions_df
