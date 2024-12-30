import numpy as np
import random
import utils

class MOParticle:
    def __init__(self, var_num):
        self.position = np.zeros(var_num)
        self.velocity = np.zeros(var_num)
        self.p_best = np.zeros(var_num)
        self.p_best_fitness = np.inf
        self.score = np.zeros(2)  # Objective scores

class MOPSO:
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
        self.lam = parameters[2]  # lamda to control service time
        self.server_pair = server_pair
        self.swarm = [MOParticle(self.var_num) for _ in range(self.size)]
        self.g_best = MOParticle(self.var_num)  # global best particle
        self.function_name = function_name
        old_cold, _ = utils.get_st(function_name, server_pair[0])
        new_cold, _ = utils.get_st(function_name, server_pair[1])
        cold_carbon_max, _ = utils.compute_exe(function_name, server_pair, ci_avg)
        self.max_st = max(old_cold, new_cold)
        self.max_carbon_st = max(cold_carbon_max)
        self.max_carbon_kat = max(
            utils.compute_kat(function_name, server_pair[0], 7, ci_avg),
            utils.compute_kat(function_name, server_pair[1], 7, ci_avg)
        )
        self.bound = [[0, 0], [1, max(self.var_2)]]  # setting bound
        temp = np.inf
        self.temp = 0
        self.st_score = []
        self.carbon_score = []
        for particle in self.swarm:
            particle.position[0] = int(random.choice(self.var_1))
            particle.position[1] = int(random.choice(self.var_2))
            for j in range(self.var_num):
                particle.velocity[j] = random.uniform(0, 1)
            particle.p_best = particle.position
            fit = self.fitness(particle.position, cur_ci, cur_interval)
            if np.all(fit < temp):
                self.g_best = particle
                temp = fit
        self.temp = temp

    def prob_cold(self, cur_interval, kat):
        if len(cur_interval) == 0:
            return 0.5, 0.5
        else:
            cold = 0
            warm = 0
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
        score = np.zeros(2)
        old_kat_carbon = utils.compute_kat(self.function_name, self.server_pair[0], kat, ci)
        new_kat_carbon = utils.compute_kat(self.function_name, self.server_pair[1], kat, ci)
        cold_carbon, warm_carbon = utils.compute_exe(self.function_name, self.server_pair, ci)
        old_st = utils.get_st(self.function_name, self.server_pair[0])
        new_st = utils.get_st(self.function_name, self.server_pair[1])
        score[0] = (1 - self.lam) * (((1 - ka_loc) * old_kat_carbon + ka_loc * new_kat_carbon) / self.max_carbon_kat)
        cold_prob, warm_prob = self.prob_cold(past_interval, kat)
        part_time_prob = cold_prob * ((1 - ka_loc) * old_st[0] + ka_loc * new_st[0]) + warm_prob * (
                    (1 - ka_loc) * old_st[1] + ka_loc * new_st[1])
        part_carbon_prob = cold_prob * ((1 - ka_loc) * cold_carbon[0] + ka_loc * cold_carbon[1]) + warm_prob * (
                    (1 - ka_loc) * warm_carbon[0] + ka_loc * warm_carbon[1])
        score[1] = self.lam * (part_time_prob) / self.max_st
        score[1] += (1 - self.lam) * (part_carbon_prob) / (self.max_carbon_st)
        return score

    def update_operator(self, ci, past_interval, diff_ci, diff_fn):
        w_max = 1
        w_min = 0.5
        c1_min = 0.3
        c1_max = 1
        c2_min = 0.3
        c2_max = 1
        self.w = w_max * (diff_fn / self.max_delta_fn + diff_ci / self.max_delta_ci)
        c1 = c1_max * (1 - (diff_fn / self.max_delta_fn + diff_ci / self.max_delta_ci))
        c2 = c2_max * (1 - (diff_fn / self.max_delta_fn + diff_ci / self.max_delta_ci))
        if self.w < w_min:
            self.w = w_min
        elif self.w > w_max:
            self.w = w_max
        if c1 < c1_min:
            c1 = c1_min
        elif c1 > c1_max:
            c1 = c1_max
        if c2 < c2_min:
            c2 = c2_min
        elif c2 > c2_max:
            c2 = c2_max
        for particle in self.swarm:
            particle.velocity = self.w * particle.velocity + c1 * random.uniform(0, 1) * (
                        particle.p_best - particle.position) + c2 * random.uniform(0, 1) * (self.g_best.position - particle.position)
            particle.position = particle.position + particle.velocity
            particle.position = particle.position.astype(int)
            for j in range(self.var_num):
                if particle.position[j] < self.bound[0][j]:
                    particle.position[j] = self.bound[0][j]
                if particle.position[j] > self.bound[1][j]:
                    particle.position[j] = self.bound[1][j]
            particle.score = self.fitness(particle.position, ci, past_interval)
            if np.all(particle.score < particle.p_best_fitness):
                particle.p_best = particle.position
                particle.p_best_fitness = particle.score
            if np.all(particle.score < self.g_best.p_best_fitness):
                self.g_best = particle

    def main(self, ci, past_interval):
        diff_ci = abs(ci - self.prev_ci)
        diff_fn = abs(len(past_interval) - self.prev_fn)
        half_indices = np.random.choice(int(self.size / 2), int(self.size / 2), replace=False)
        if diff_fn / self.max_delta_fn or diff_ci / self.max_delta_ci > 0:
            for index in half_indices:
                i = half_indices[index]
                self.swarm[i].position[0] = int(random.choice(self.var_1))
                self.swarm[i].position[1] = int(random.choice(self.var_2))
        for particle in self.swarm:
            for j in range(self.var_num):
                particle.velocity[j] = random.uniform(0, 1)
        for _ in range(1):
            self.update_operator(ci, past_interval, diff_ci, diff_fn)
        if diff_ci > self.max_delta_ci:
            self.max_delta_ci = diff_ci
        if diff_fn > self.max_delta_fn:
            self.max_delta_fn = diff_fn
        self.prev_ci = ci
        self.prev_fn = len(past_interval)
        return self.g_best.position, self.g_best.p_best
