import numpy as np
import random
import utils

class GA:
    def __init__(self, parameters, server_pair, function_name, ci_avg, cur_ci, cur_interval):
        self.max_delta_ci = 0.1
        self.prev_ci = cur_ci
        self.max_delta_fn = 0.1
        self.prev_fn = len(cur_interval)
        self.w = 1
        self.iteration = 1
        self.size = parameters[0]   
        self.var_num = 2
        self.var_1 = [0, 1]  # kat
        self.var_2 = parameters[1]  # choices of kat
        self.lam = parameters[2]  # lambda to control service time
        self.server_pair = server_pair
        self.pop_x = np.zeros((self.size, self.var_num))  # particle loc
        self.pop_v = np.zeros((self.size, self.var_num))  # particle v
        self.p_best = np.zeros((self.size, self.var_num))  # best particle loc
        self.g_best = np.zeros((1, self.var_num))  # global particle loc
        self.function_name = function_name
        # compute max
        old_cold, _ = utils.get_st(function_name, server_pair[0])
        new_cold, _ = utils.get_st(function_name, server_pair[1])
        cold_carbon_max, _ = utils.compute_exe(function_name, server_pair, ci_avg)
        self.max_st = max(old_cold, new_cold)
        self.max_carbon_st = max(cold_carbon_max)
        self.max_carbon_kat = max(utils.compute_kat(function_name, server_pair[0], 7, ci_avg),
                                  utils.compute_kat(function_name, server_pair[1], 7, ci_avg))
        self.bound = [[0, 0], [1, max(self.var_2)]]  # setting bound
        temp = np.inf
        self.st_score = []
        self.carbon_score = []
        for i in range(self.size):
            self.pop_x[i][0] = int(random.choice(self.var_1))
            self.pop_x[i][1] = int(random.choice(self.var_2))
            for j in range(self.var_num):
                self.pop_v[i][j] = random.uniform(0, 1)
            self.p_best[i] = self.pop_x[i]
            fit = self.fitness(self.p_best[i], cur_ci, cur_interval)
            if fit < temp:
                self.g_best = self.p_best[i]
                temp = fit
        self.temp = temp

    def prob_cold(self, cur_interval, kat):
        if len(cur_interval) == 0:
            return 0.5, 0.5
        else:
            cold, warm = 0, 0
            for interval in cur_interval:
                if interval <= kat:
                    warm += 1
                else:
                    cold += 1
            return cold / (cold + warm), warm / (cold + warm)

    def fitness(self, var, ci, past_interval):
        var = var.astype(int)
        ka_loc = var[0]
        kat = var[1]
        score = 0
        old_kat_carbon = utils.compute_kat(self.function_name, self.server_pair[0], kat, ci)
        new_kat_carbon = utils.compute_kat(self.function_name, self.server_pair[1], kat, ci)
        cold_carbon, warm_carbon = utils.compute_exe(self.function_name, self.server_pair, ci)
        old_st = utils.get_st(self.function_name, self.server_pair[0])
        new_st = utils.get_st(self.function_name, self.server_pair[1])
        score += (1 - self.lam) * (((1 - ka_loc) * old_kat_carbon + ka_loc * new_kat_carbon) / self.max_carbon_kat)
        cold_prob, warm_prob = self.prob_cold(past_interval, kat)
        part_time_prob = cold_prob * ((1 - ka_loc) * old_st[0] + ka_loc * new_st[0]) + \
                         warm_prob * ((1 - ka_loc) * old_st[1] + ka_loc * new_st[1])
        part_carbon_prob = cold_prob * ((1 - ka_loc) * cold_carbon[0] + ka_loc * cold_carbon[1]) + \
                           warm_prob * ((1 - ka_loc) * warm_carbon[0] + ka_loc * warm_carbon[1])
        score += self.lam * (part_time_prob) / self.max_st
        score += (1 - self.lam) * (part_carbon_prob) / self.max_carbon_st
        return score

    def next_generation(self):
        # Create a new population using selection, crossover, and mutation
        new_population = np.zeros_like(self.pop_x)

        population_size = self.size if self.size % 2 == 0 else self.size - 1

        for i in range(0, population_size, 2):
            parent1, parent2 = self.selection()

            # Crossover
            child1, child2 = self.crossover(self.pop_x[parent1], self.pop_x[parent2])

            # Mutation
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)

            new_population[i] = child1
            new_population[i + 1] = child2

        if self.size % 2 != 0:
            new_population[-1] = self.pop_x[-1]

        self.pop_x = new_population

    def selection(self):
        # Dummy selection method (e.g., tournament or roulette selection)
        return random.sample(range(self.size), 2)

    def crossover(self, parent1, parent2):
        # Dummy crossover method (e.g., single-point crossover)
        point = random.randint(0, self.var_num - 1)
        child1 = np.copy(parent1)
        child2 = np.copy(parent2)
        child1[point:], child2[point:] = parent2[point:], parent1[point:]
        return child1, child2

    def mutation(self, individual):
        # Dummy mutation method (e.g., random change to a gene)
        mutation_point = random.randint(0, self.var_num - 1)
        individual[mutation_point] = random.choice(self.var_1)
        return individual

    def main(self, ci, past_interval):
        diff_ci = abs(ci - self.prev_ci)
        diff_fn = abs(len(past_interval) - self.prev_fn)

        # Randomly reinitialize half of the population if needed
        half_indices = np.random.choice(self.size // 2, self.size // 2, replace=False)
        if diff_fn / self.max_delta_fn or diff_ci / self.max_delta_ci > 0:
            for index in half_indices:
                self.pop_x[index][0] = int(random.choice(self.var_1))
                self.pop_x[index][1] = int(random.choice(self.var_2))

        # Update velocity and position of each particle
        for i in range(self.size):
            for j in range(self.var_num):
                self.pop_v[i][j] = random.uniform(0, 1)

        for _ in range(1):
            self.next_generation()

        if diff_ci > self.max_delta_ci:
            self.max_delta_ci = diff_ci
        if diff_fn > self.max_delta_fn:
            self.max_delta_fn = diff_fn
        self.prev_ci = ci
        self.prev_fn = len(past_interval)

        return self.g_best, self.p_best
