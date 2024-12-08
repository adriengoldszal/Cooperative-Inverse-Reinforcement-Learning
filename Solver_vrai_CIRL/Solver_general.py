import itertools
import numpy as np
from functools import lru_cache
import matplotlib.pyplot as plt

 # Set a fixed seed for reproducibility

##########################################
# CIRL with Belief States over Theta
##########################################

class Plan:
    def __init__(self, a_R, alpha_vector):
        self.a_R = a_R
        self.alpha_vector = alpha_vector

    def __hash__(self):
        return hash((self.a_R, tuple(np.round(self.alpha_vector,8))))

    def __eq__(self, other):
        return isinstance(other, Plan) and self.a_R == other.a_R and np.allclose(self.alpha_vector, other.alpha_vector, atol=1e-8)

    def __repr__(self):
        return f"Plan(a_R={self.a_R}, alpha={np.round(self.alpha_vector,4)})"

########## CIRL Game ##########

class CIRLGame:
    def __init__(self, X_space, theta_space, A_H, A_R, T, R, gamma, P0, Horizon, initial_belief, verbose=True):
        self.X_space = X_space
        self.theta_space = theta_space
        self.A_H = A_H
        self.A_R = A_R
        self.T = T
        self.R = R
        self.gamma = gamma
        self.P0 = P0
        self.Horizon = Horizon
        self.verbose = verbose

        self.S_space = list(itertools.product(X_space, theta_space))
        self.initial_belief = initial_belief / np.sum(initial_belief)

    def print(self, *args):
        if self.verbose:
            print(*args)

    def alpha_backup(self, depth, alpha_vectors_next):
        idx_map = {s:i for i,s in enumerate(self.S_space)}
        if len(alpha_vectors_next)==0:
            def V_next(xp,thetap):
                return 0.0
        else:
            max_vals = np.max([p.alpha_vector for p in alpha_vectors_next], axis=0)
            def V_next(xp,thetap):
                return max_vals[idx_map[(xp,thetap)]]

        new_alpha_vectors = []
        for a_R in self.A_R:
            alpha_array = np.zeros(len(self.S_space))
            for i,(x,θ) in enumerate(self.S_space):
                q_values = []
                for a_H in self.A_H:
                    val=0.0
                    for x_prime in self.X_space:
                        p = self.T(x,a_H,a_R,x_prime)
                        if p>0:
                            val += p*(self.R(x_prime,θ)+self.gamma*V_next(x_prime,θ))
                    q_values.append((a_H,val))
                q_values.sort(key=lambda x:x[1], reverse=True)
                alpha_array[i] = q_values[0][1]
            new_alpha_vectors.append(Plan(a_R, alpha_array))
        return new_alpha_vectors

    def prune(self, alpha_vectors):
        all_alpha = [(p, p.alpha_vector) for p in alpha_vectors]
        pruned = []
        seen = set()
        for i,(pi,alphai) in enumerate(all_alpha):
            alpha_tuple = tuple(np.round(alphai,8))
            if alpha_tuple in seen:
                continue
            dominated=False
            for j,(pj,alphaj) in enumerate(all_alpha):
                if pj!=pi:
                    if np.all(alphaj>=alphai) and np.any(alphaj>alphai):
                        dominated=True
                        break
            if not dominated:
                pruned.append(pi)
                seen.add(alpha_tuple)
        return pruned

    def solve(self):
        base_alpha = Plan(None, np.array([self.R(*s) for s in self.S_space]))
        alpha_vectors = [base_alpha]

        for d in range(1, self.Horizon+1):
            alpha_vectors = self.alpha_backup(d, alpha_vectors)
            alpha_vectors = self.prune(alpha_vectors)

        return alpha_vectors

    def value_of_belief(self, alpha_vectors, b, x):
        idx_map = {s:i for i,s in enumerate(self.S_space)}
        vals = []
        for p in alpha_vectors:
            val=0.0
            for i,θ in enumerate(self.theta_space):
                val+=b[i]*p.alpha_vector[idx_map[(x,θ)]]
            vals.append(val)
        return max(vals) if len(vals)>0 else 0.0

    def forward_simulation(self, alpha_vectors):
        states = list(self.P0.keys())
        probs = list(self.P0.values())
        s_index = np.random.choice(len(states), p=probs)
        s = states[s_index]
        x_init, θ_true = s  # True θ chosen once at start

        b = self.initial_belief.copy()
        trajectory=[(x_init, θ_true)]
        A_R_seq=[]
        A_H_seq=[]
        X_seq=[x_init]
        rewards = [self.R(x_init, θ_true)]  # use θ_true

        idx_map = {st:i for i,st in enumerate(self.S_space)}

        for t in range(self.Horizon):
            # pick best a_R
            best_val = -np.inf
            best_action = None
            for p in alpha_vectors:
                if p.a_R is None:
                    continue
                val=0.0
                for i_,θ_ in enumerate(self.theta_space):
                    val+=b[i_]*p.alpha_vector[idx_map[(X_seq[-1],θ_)]]
                if val>best_val:
                    best_val=val
                    best_action=p.a_R

            if best_action is None:
                break

            a_R = best_action
            A_R_seq.append(a_R)

            # Human action chosen based on θ_true
            max_vals = np.max([p.alpha_vector for p in alpha_vectors], axis=0)
            def V_next(xp,thetap):
                return max_vals[idx_map[(xp,thetap)]]

            q_values=[]
            for a_H_cand in self.A_H:
                val=0.0
                for xp in self.X_space:
                    p_ = self.T(X_seq[-1], a_H_cand, a_R, xp)
                    if p_>0:
                        val += p_*(self.R(xp, θ_true) + self.gamma*V_next(xp, θ_true))
                q_values.append((a_H_cand,val))
            q_values.sort(key=lambda x:x[1], reverse=True)
            a_H = q_values[0][0]
            A_H_seq.append(a_H)

            # Transition
            ps = [self.T(X_seq[-1],a_H,a_R,xp) for xp in self.X_space]
            ps = np.array(ps)
            if ps.sum()==0:
                break
            ps=ps/ps.sum()
            xp_index=np.random.choice(len(self.X_space),p=ps)
            x_prime=self.X_space[xp_index]
            X_seq.append(x_prime)

            # Belief update as before, using a_H and next state x_prime
            # (No change in belief update logic)
            new_b = np.zeros_like(b)
            for i_,θ_ in enumerate(self.theta_space):
                qv=[]
                for aH_c in self.A_H:
                    val=0.0
                    for xpp in self.X_space:
                        p__ = self.T(X_seq[-2],aH_c,a_R,xpp)
                        if p__>0:
                            val+=p__*(self.R(xpp,θ_)+self.gamma*V_next(xpp,θ_))
                    qv.append((aH_c,val))
                qv.sort(key=lambda xx:xx[1], reverse=True)
                best_aH_θ = qv[0][0]
                p_aH = 1.0 if best_aH_θ==a_H else 0.0
                new_b[i_]=b[i_]*p_aH
            if new_b.sum()>0:
                b=new_b/new_b.sum()

            # Record reward at new state with θ_true
            rewards.append(self.R(x_prime, θ_true))
            trajectory.append((x_prime, θ_true))

        return X_seq, A_H_seq, A_R_seq, rewards