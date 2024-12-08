import Solver_general
import importlib
importlib.reload(Solver_general)
from Solver_general import *


m=4
Horizon=5
X_space=[]
for i in range(m+1):
    for j in range(m+1):
        for k in range(m+1):
            if i+j+k<=2*Horizon:
                X_space.append((i,j,k))

sandwich=(3,3,2)
soup=(3,2,3)
chiken=(4,1,0)
theta_space=[sandwich,soup, chiken]

A_R=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
A_H=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]

def T(x,a_H,a_R,xp):
    ns = tuple(np.array(x)+np.array(a_H)+np.array(a_R))
    return 1.0 if ns==xp else 0.0

def R(x,θ):
    return 1.0 if x == θ else 0.0

gamma=0.9
P0={((0,0,0),chiken):1.0}

initial_belief=np.array([1.0,0.5, 0.5])
initial_belief=initial_belief/initial_belief.sum()

game = CIRLGame(X_space,theta_space,A_H,A_R,T,R,gamma,P0,Horizon,initial_belief,verbose=True)

alpha_vectors=game.solve()

game.print("Final alpha-vectors:")
for av in alpha_vectors:
    game.print(av)

X_seq, A_H_seq, A_R_seq, rewards = game.forward_simulation(alpha_vectors)
game.print("\nForward Simulation Result:")
game.print("X_seq:",X_seq)
game.print("A_R_seq:",A_R_seq)
game.print("A_H_seq:",A_H_seq)
game.print("Rewards:", rewards)

# Plot the obtained rewards
plt.figure()
plt.plot(range(len(rewards)), rewards, marker='o')
plt.title("Reward per Step in Forward Simulation")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.grid(True)