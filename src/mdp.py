# MDP, for handling a queuing problem, where i have N users with their max buffer size B that have to share M resources

import numpy as np
from scipy.stats import norm, poisson
from itertools import product

DEBUG = 1

class MDP:
    def __init__(self, n_users, n_rb, buffer_size, cqi_values, lambdas=[3,1]):
        self.N = n_users
        self.M = n_rb
        self.B = buffer_size
        self.cqi = cqi_values
        self.lambdas = lambdas
        self.generate_actions()

        overflow_states = 2
        self.state_space = list(product(range(self.B + 1 + overflow_states), repeat=self.N))
        self.state_space_size = len(self.state_space)
        self.action_space_size = len(self.actions)

        self.generate_transition_matrix()

    def generate_actions(self):
        # Generate all possible resource allocation combinations
        all_combinations = list(product(range(self.M + 1), repeat=self.N))
        # Filter valid combinations with total allocation <= M
        valid_combinations = [comb for comb in all_combinations if sum(comb) <= self.M]
        
        self.actions = np.array(valid_combinations)
        
    def generate_transition_matrix(self):
        self.transition_matrix = np.zeros((self.action_space_size,
                                            self.state_space_size,
                                            self.state_space_size))
        for a_idx, a in enumerate(self.actions):
            for b_idx, b in enumerate(self.state_space):
                for b_prime_idx, b_prime in enumerate(self.state_space):
                    self.transition_matrix[a_idx, b_idx, b_prime_idx] += self.transition_probability(b_prime, b, a, self.lambdas)

                    print("a_idx ", a_idx, " b_idx ", b_idx, " b_prime_idx ", b_prime_idx, " prob ", self.transition_matrix[a_idx, b_idx, b_prime_idx])
                    # normalize the transition matrix
                    self.transition_matrix[a_idx, b, :] /= np.sum(self.transition_matrix[a_idx, b, :])
                    
    def transition_probability(self, b_prime, b, a, L):
        # Calcular el parámetro lambda para la distribución de Poisson
        lambda_val = L
        
        prob_total = 1.0

        if DEBUG:
            print("\n------------------------------------")
            
        # Calcular la probabilidad de transición para cada elemento del vector
        for i in range(len(b_prime)):
            # Calcular la probabilidad de Poisson para el elemento i
            prob_i = poisson.pmf(b_prime[i] - b[i] + a[i], lambda_val[i])
            if DEBUG:
                print("b_prime[i] ", b_prime[i], " b[i] ", b[i], " a[i] ", a[i], " prob_i ", prob_i)
            # Multiplicar la probabilidad al total
            prob_total *= prob_i
        
        if DEBUG:
            if prob_total != -1:
                print("b_prime ", b_prime, " b ", b, " a ", a, " prob ", prob_total)
        return prob_total

    # calculate the reward for a given state and action as the sum of the rewards of each user
    def reward(self, state, state_prime, action):
        return sum([self.reward_user(state[i], state_prime[i], action[i], self.cqi[i], self.lambdas[i]) for i in range(self.N)])

    # calculate the reward for a given user as max(0, max(0, b - a*h) + l - self.B) + max(0, - a*h + b)
    # where b is the buffer size, a is the allocated resources, h is the cqi value and l is a value from the poisson distribution
    def reward_user(self, b, b_prime, a, cqi, l):
        # if b_prime > self.B: the user cost will be the overflow plus
        # if b_prime <= self.B: the user cost will be b_prime 
        reward_user = b_prime #+ max(0, b_prime - self.B)
        return reward_user
        #return max(0, max(0, b - a*cqi) + np.random.poisson(l) - self.B) + max(0, - a*cqi + b)