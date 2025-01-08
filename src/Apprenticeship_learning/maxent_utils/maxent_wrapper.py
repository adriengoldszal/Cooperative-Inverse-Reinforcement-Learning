from typing import List, Tuple, Dict 
import numpy as np 
from dataclasses import dataclass

@dataclass 
class Trajectory:
    """Wrapper class to make trajectories compatible with maxent package"""
    states_actions: List[Tuple[np.ndarray, str]]
    
    def states(self):
        """Return list of state indices"""
        state_indices = [self._state_to_idx(s) for s, _ in self.states_actions]
        return state_indices
    
    def transitions(self):
        """Return list of (state, next_state, action) indices"""
        transitions = []
        for i in range(len(self.states_actions) - 1):
            curr_state, action = self.states_actions[i]
            next_state, _ = self.states_actions[i + 1]
            transition = (
                self._state_to_idx(curr_state),
                self._state_to_idx(next_state),
                self._action_to_idx(action)
            )
            transitions.append(transition)
        return transitions
    
    def _state_to_idx(self, state: np.ndarray) -> int:
        """Convert 2D state to flat index"""
        idx = state[0] * self.grid_size + state[1]
        if not hasattr(self, '_printed_size'):
            self._printed_size = True
        return idx
    
    def _action_to_idx(self, action: str) -> int:
        """Convert action string to index"""
        action_map = {'N': 0, 'S': 1, 'E': 2, 'W': 3, 'NOOP': 4}
        return action_map[action]

class Optimizer:
    """
    Optimizer base-class.

    Note:
        Before use of any optimizer, its `reset` function must be called.

    Attributes:
        parameters: The parameters to be optimized. This should only be set
            via the `reset` method of this optimizer.
    """
    def __init__(self):
        self.parameters = None

    def reset(self, parameters):
        """
        Reset this optimizer.

        Args:
            parameters: The parameters to optimize.
        """
        self.parameters = np.array(parameters, dtype=np.float64)

    def step(self, grad, *args, **kwargs):
        """
        Perform a single optimization step.

        Args:
            grad: The gradient used for the optimization step.

            Other arguments are optimizer-specific.
        """
        raise NotImplementedError

    def normalize_grad(self, ord=None):
        """
        Create a new wrapper for this optimizer which normalizes the
        gradient before each step.

        Returns:
            An Optimizer instance wrapping this Optimizer, normalizing the
            gradient before each step.

        See also:
            `class NormalizeGrad`
        """
        return NormalizeGrad(self, ord)


class Sga(Optimizer):
    """
    Basic stochastic gradient ascent.

    Args:
        lr: The learning-rate. This may either be a float for a constant
            learning-rate or a function
            `(k: Integer) -> learning_rate: Float`
            taking the step number as parameter and returning a learning
            rate as result.
    """
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.k = 0

    def reset(self, parameters):
        super().reset(parameters)
        self.k = 0

    def step(self, grad, *args, **kwargs):
        lr = self.lr if not callable(self.lr) else self.lr(self.k)
        self.k += 1
        self.parameters += lr * grad
        return self.parameters


class ExpSga(Optimizer):
    """
    Exponentiated stochastic gradient ascent.

    Args:
        lr: The learning-rate. This may either be a float for a constant
            learning-rate or a function
            `(k: Integer) -> learning_rate: Float`.
        normalize: A boolean specifying if the parameters should be
            normalized after each step.
    """
    def __init__(self, lr, normalize=False):
        super().__init__()
        self.lr = lr
        self.normalize = normalize
        self.k = 0

    def reset(self, parameters):
        super().reset(parameters)
        self.k = 0

    def step(self, grad, *args, **kwargs):
        lr = self.lr if not callable(self.lr) else self.lr(self.k)
        self.k += 1
        self.parameters *= np.exp(lr * grad)
        if self.normalize:
            self.parameters /= np.sum(self.parameters)
        return self.parameters


class NormalizeGrad(Optimizer):
    """
    A wrapper wrapping another Optimizer, normalizing the gradient before
    each step.

    Args:
        opt: The underlying optimizer to be used.
        ord: The order of the norm to be used for normalizing.
    """
    def __init__(self, opt, ord=None):
        super().__init__()
        self.opt = opt
        self.ord = ord

    def reset(self, parameters):
        super().reset(parameters)
        self.opt.reset(parameters)

    def step(self, grad, *args, **kwargs):
        norm = np.linalg.norm(grad, ord=self.ord)
        if norm == 0:
            raise ValueError("Gradient norm is zero; cannot normalize.")
        normalized_grad = grad / norm
        return self.opt.step(normalized_grad, *args, **kwargs)


def linear_decay(lr0=0.2, decay_rate=1.0, decay_steps=1):
    def _lr(k):
        return lr0 / (1.0 + decay_rate * np.floor(k / decay_steps))
    return _lr


def power_decay(lr0=0.2, decay_rate=1.0, decay_steps=1, power=2):
    def _lr(k):
        return lr0 / (1.0 + decay_rate * np.floor(k / decay_steps)) ** power
    return _lr


def exponential_decay(lr0=0.2, decay_rate=0.5, decay_steps=1):
    def _lr(k):
        return lr0 * np.exp(-decay_rate * np.floor(k / decay_steps))
    return _lr

class Initializer:
    """Uniform random initializer for maxent package"""
    def __init__(self, low=-1, high=1):
        self.low = low
        self.high = high
    
    def __call__(self, n_features):
        params = np.random.uniform(self.low, self.high, n_features)
        return params