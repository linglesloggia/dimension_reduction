import numpy as np

# Value iteration algorithm for the MDP

# Loop:
#     \Delta <- 0
#     Loop for each s \in S:
#         v <- V(s)
#         V(s) <- max_a \sum_{s', r} p(s', r | s, a) [r + \gamma V(s')]
#         \Delta <- max(\delta, |v - V(s)|)
# until \Delta < \theta

# output a deterministic policy, \pi \sim \pi_*, such that
# \pi(s) = argmax_a \sum_{s', r} p(s', r | s, a) [r + \gamma V(s')]

DEBUGV = 0
def value_iteration(mdp, gamma=0.8, epsilon=0.001):
    # Initialize the value function
    V = np.zeros(mdp.state_space_size)
    V_prev = np.zeros(mdp.state_space_size)
    # Initialize the policy
    policy = np.zeros(mdp.state_space_size)

    # Initialize the value function with random values
    #V = np.random.rand(mdp.state_space_size)
    # Initialize the value function with all -10
    #V = np.full(mdp.state_space_size, -10)

    
    # Initialize the list of Vs for each iteration
    Vs = []
    Vo = []
    # Initialize the iteration counter
    iteration = 0
    
    rewards = []
    difs = [] 

    ###################################
    # Value Iteration
    ###################################

    while True:
        # Initialize delta
        delta = 1e-2

        # max value will be a vector of size N, where N is the number of users an each entry is -np.inf
        max_value = np.full(mdp.state_space_size, -np.inf)
        max_action = np.full(mdp.state_space_size, None)
        
        V_prev = V.copy()

        # For each state
        for s_idx, s in enumerate(mdp.state_space):
            # Initialize the max value and the max action
            
            value = np.zeros((mdp.state_space_size, mdp.action_space_size))

            for a_idx, a in enumerate(mdp.actions):
                # Initialize the value
                total_reward = 0
                # For each possible next state
                for s_prime_idx, s_prime in enumerate(mdp.state_space):
                    # Accumulate the expected value
                    value[s_idx, a_idx] += mdp.transition_matrix[a_idx, s_idx, s_prime_idx] * (mdp.reward(s, s_prime, a) + gamma * V[s_prime_idx])
                    #print("State ", s, " Action ", a, " Next state ", s_prime, " Reward ", mdp.reward(s, s_prime, a), " Value ", value[s_idx, a_idx])

                #if value[s_idx, a_idx] > max_value[s_idx]:
                    #max_value[s_idx] = value[s_idx, a_idx]
                    #max_action[s_idx] = a_idx

            max_value[s_idx] = np.max(value[s_idx, :])
            #max_action[s_idx] = np.argmax(value[s_idx, :])

            V[s_idx] = max_value[s_idx]

            #policy[s_idx] = max_action[s_idx]
            rewards.append(total_reward)
        
        dif_norm = np.linalg.norm(V_prev - V)
        print(dif_norm)
        
       
        difs.append(dif_norm)
                
        Vs.append(V.copy())
        Vo.append(V[0])

        # Increment the iteration counter
        iteration += 1
        
        # \pi(s) = argmax_a \sum_{s', r} p(s', r | s, a) [r + \gamma V(s')]


        if iteration == 50:


            ###################################
            # obtain the policy for each state
            ###################################

            for s_idx, s in enumerate(mdp.state_space):

                value = np.zeros((mdp.state_space_size, mdp.action_space_size))

                for a_idx, a in enumerate(mdp.actions):
                    # For each possible next state
                    for s_prime_idx, s_prime in enumerate(mdp.state_space):
                        # Accumulate the expected value
                        value[s_idx, a_idx] += mdp.transition_matrix[a_idx, s_idx, s_prime_idx] * (mdp.reward(s, s_prime, a) + gamma * V[s_prime_idx])

                max_action[s_idx] = np.argmax(value[s_idx, :])

                policy[s_idx] = max_action[s_idx]

            break

    
    return Vs, policy, rewards, difs, Vo