import itertools
import torch
import torch.distributions as dist
import numpy as np
from typing import Dict, List, Tuple, Iterable


def generate_hparam_configs(base_config:Dict, hparam_ranges:Dict) -> Tuple[List[Dict], List[str]]:
    """
    Generate a list of hyperparameter configurations for hparam sweeping

    :param base_config (Dict): base configuration dictionary
    :param hparam_ranges (Dict): dictionary mapping hyperparameter names to lists of values to sweep over
    :return (Tuple[List[Dict], List[str]]): list of hyperparameter configurations and swept parameter names
    """

    # hparam_ranges = {}
    # hparam_ranges["policy_learning_rate"]= [1e-4, 1e-3, 1e-2, 1e-1]
    # hparam_ranges["critic_learning_rate"] = [1e-4, 1e-3, 1e-2, 1e-1]
    # hparam_ranges["critic_hidden_size"] = [[300, 250], [400, 300], [450, 400]]
    # hparam_ranges["policy_hidden_size"] = [[300, 250], [400, 300], [450, 400]]
    # hparam_ranges["gamma"] = [0.99]
    # hparam_ranges[ "batch_size"] = [32, 64, 100, 200]
    # hparam_ranges["buffer_capacity"] = [1e4, 1e5, 1e6, 1e7]
    # hparam_ranges["tau"] = [0.001, 0.01, 0.1, 0.11, 0.12]
   
    keys, values = zip(*hparam_ranges.items())
    hparam_configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    swept_params = list(hparam_ranges.keys())

    new_configs = []
    for hparam_config in hparam_configurations:
        new_config = base_config.copy()
        new_config.update(hparam_config)
        new_configs.append(new_config)

    return new_configs, swept_params


def grid_search(num_samples: int, min: float = None, max: float = None, **kwargs)->Iterable:
    """ Implement this method to set hparam range over a grid of hyperparameters.
    :param num_samples (int): number of samples making up the grid
    :param min (float): minimum value for the allowed range to sweep over
    :param max (float): maximum value for the allowed range to sweep over
    :param kwargs: additional keyword arguments to parametrise the grid.
    :return (Iterable): tensor/array/list/etc... of values to sweep over

    Example use: hparam_ranges['batch_size'] = grid_search(64, 512, 6, log=True)

    **YOU MAY IMPLEMENT THIS FUNCTION FOR Q5**

    """
    values = np.zeros(num_samples)
    values = np.linspace(min, max, num_samples)
    
    return values


def random_search(num_samples: int, distribution: str, min: float=None, max: float=None, **kwargs) -> Iterable:
    """ Implement this method to sweep via random search, sampling from a given distribution.
    :param num_samples (int): number of samples to take from the distribution
    :param distribution (str): name of the distribution to sample from
        (you can instantiate the distribution using torch.distributions, numpy.random, or else).
    :param min (float): minimum value for the allowed range to sweep over (for continuous distributions)
    :param max (float): maximum value for the allowed range to sweep over (for continuous distributions)
    :param kwargs: additional keyword arguments to parametrise the distribution.

    Example use: hparam_ranges['lr'] = random_search(1e-6, 1e-1, 10, distribution='exponential', lambda=0.1)

    **YOU MAY IMPLEMENT THIS FUNCTION FOR Q5**

    """
    values = torch.zeros(num_samples)
    if distribution == 'exponential':
        rate = kwargs.get('rate', 0.1)
        values = torch.empty(num_samples)
        for i in range(num_samples):
            values[i] = dist.Exponential(rate=rate).sample()
        values = (max - min) * values / values.max() + min
        return values.detach().numpy()
    
    if distribution=='discrete':
        return np.random.randint(min, max, size=(num_samples,2))
    if distribution=='uniform':
        return np.random.uniform(min, max, size=num_samples)
    if distribution=='normal':
        return np.random.normal(min, max, size=num_samples)


