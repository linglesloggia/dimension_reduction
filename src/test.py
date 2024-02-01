# test the algorithm

Vs, policy, rewards, difs, Vo = value_iteration(mdp)

print("Value function:")
print(Vs[-1])

# print (State[i], action taken) for each state
for i in range(mdp.state_space_size):
    print("Estado: ", mdp.state_space[i], " Politica " , policy[i], " Acción ", mdp.actions[int(policy[i])])

# plot difs

plt.plot(difs)
plt.ylabel('Diferencia entre iteraciones')
plt.xlabel('Iteraciones')
plt.show()

# plot Vo

plt.plot(Vo)
plt.ylabel('Valor función de valor')
plt.xlabel('Iteraciones')
plt.show()