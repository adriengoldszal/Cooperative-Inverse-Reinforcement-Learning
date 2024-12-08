import Solver_general
import importlib
importlib.reload(Solver_general)
from Solver_general import *


####### Parameters of the problem #######
L=4# taille de la grille
N_phi=2 # nombre de features
centroids=np.array([[0,1],[1,4]]) # les cenroides de la feature function
sigmas=np.array([0.5,0.5,0.5,0.5]) # les sigmas de la feature function
m=2 # nombre de valeurs de theta possibles -1
gamma=0.9
Horizon=4

####### Définition des ensmbles######
A_H=[(0,0), (-1, 0), (1, 0), (0, -1), (0, 1)] # les actions de l'humain
A_R=[(0,0), (-1, 0), (1, 0), (0, -1), (0, 1)] # les actions du robot
X_space=[(i,j) for i in range(-1,L+1) for j in range(-1,L+1)] # les états, on crée des états fictifs pour les bords pour rendre les zones interidtes
theta_space_uni=[-1+2*i/m for i in range(m+1)] # les valeurs de theta
Theta_space=[(theta1, theta2) for theta1 in theta_space_uni for theta2 in theta_space_uni] # les valeurs de theta

n_theta=len(Theta_space)
initial_belief=[1/n_theta for i in range(n_theta)] # la distribution initiale de theta est a priori uniforme


####### Définition des fonctions de transition et de récompense #######

def phi(x):
    """
    Fonction de features
    """
    return np.array([np.exp(-np.linalg.norm(np.array(x)-c)**2/sigma) for c,sigma in zip(centroids,sigmas)])


def T(x,a_H,a_R,xp):
    """
    Fonction de transition
    """
    new_x = tuple(np.array(x)+np.array(a_H)+np.array(a_R))
    return 1.0 if new_x==xp else 0.0

def R(x,theta):
    """
    Fonction de récompense
    """
    theta1=np.array(theta)
    if x[0]==-1 or x[0]==L or x[1]==-1 or x[1]==L:
        return -1000
    else:
        return np.dot(phi(x),theta1)

P0={((L//2,L//2),Theta_space[5]):1.0}

######## Run du jeu ########

game=CIRLGame(X_space,Theta_space,A_H,A_R,T,R,gamma,P0,Horizon,initial_belief,verbose=True)
alpha_vectors=game.solve()
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