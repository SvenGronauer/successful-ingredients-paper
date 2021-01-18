import os
import json
import numpy as np
from collections import namedtuple, OrderedDict
import matplotlib.pyplot as plt
from sipga.algs import core
import atexit
import warnings
import os
import gym
import torch


# def evaluate_returns(actor_critic_model, env, log_dir):
#     ee = EnvironmentEvaluator(env=env,
#                               actor_critic_model=actor_critic_model)
#     ret, _ = ee.eval(num_evaluations=32)
#
#     file_name_path = os.path.join(log_dir, 'returns.csv')
#     np.savetxt(file_name_path, ret, delimiter=',')
#     print('SAVED to:', file_name_path)
#     print('Mean return:', np.mean(ret))


def get_file_contents(file_path: str,
                      skip_header: bool = False):
    """Open the file with given path and return Python object."""
    assert os.path.isfile(file_path), 'No file exists at: {}'.format(file_path)

    if file_path.endswith('.json'):  # return dict
        with open(file_path, 'r') as fp:
            data = json.load(fp)

    elif file_path.endswith('.csv'):
        if skip_header:
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        else:
            data = np.loadtxt(file_path, delimiter=",")

    else:
        raise NotImplementedError

    return data


def get_experiment_paths(path: str
                         ) -> tuple:
    """ Walk through path recursively and find experiment log files.

        Note:
            In a directory must exist a config.json and metrics.json file, such
            that path is detected.

    Parameters
    ----------
    path
        Path that is walked through recursively.

    Raises
    ------
    AssertionError
        If no experiment runs where found.

    Returns
    -------
    list
        Holding path names to directories.
    """
    experiment_paths = []
    for root, dirs, files in os.walk(path):  # walk recursively trough basedir
        config_json_in_dir = False
        metrics_json_in_dir = False
        for file in files:
            if file.endswith("config.json"):
                config_json_in_dir = True
            if file.endswith("progress.csv"):
                metrics_json_in_dir = True
        if config_json_in_dir and metrics_json_in_dir:
            experiment_paths.append(root)

    assert experiment_paths, f'No experiments found at: {path}'

    return tuple(experiment_paths)


class EnvironmentEvaluator(object):
    def __init__(self, log_dir):

        self.log_dir = log_dir
        self.env = None
        self.ac = None

        # open returns.csv file at the beginning to avoid disk access errors
        # on our HPC servers...
        os.makedirs(log_dir, exist_ok=True)
        self.output_file_name = 'returns.csv'
        self.output_file = open(os.path.join(log_dir, self.output_file_name),
                                'w')
        # Register close function to be executed upon normal program termination
        atexit.register(self.output_file.close)

    def eval(self, env, ac, num_evaluations):
        """ Evaluate actor critic module for given number of evaluations.
        """
        assert isinstance(ac, core.ActorCritic)
        self.ac = ac

        if isinstance(env, gym.Env):
            self.env = env
        elif isinstance(env, str):
            self.env = gym.make(env)
        else:
            raise TypeError('Env is not of type: str, gym.Env')

        self.ac.eval()  # disable exploration noise
        returns = list()
        ep_lengths = list()
        for i in range(num_evaluations):
            ret, ep_length = self.eval_once()
            returns.append(ret)
            ep_lengths.append(ep_length)

        # now write returns as column into output file...
        self.write_to_output_file(returns)
        print('Saved to:', os.path.join(self.log_dir, self.output_file_name))
        print(f'Mean Ret: { np.mean(returns)} \t'
              f'Mean EpLen: {np.mean(ep_lengths)}')

        return np.array(returns), np.array(ep_lengths)

    def eval_once(self):
        assert not self.ac.training, 'Call actor_critic.eval() beforehand.'
        done = False
        x = self.env.reset()
        ret = 0.
        episode_length = 0

        while not done:
            obs = torch.as_tensor(x, dtype=torch.float32)
            action, value, *_ = self.ac(obs)
            x, r, done, info = self.env.step(action)
            ret += r
            episode_length += 1

        return ret, episode_length

    def write_to_output_file(self, xs: list):
        column = [str(x) for x in xs]
        self.output_file.write("\n".join(column) + "\n")
        self.output_file.flush()


class ParamaterContainer:
    def __init__(self):
        self.parameter_settings = dict()

    @classmethod
    def _parameters_to_string(
            cls,
            params: tuple
    ) -> str:
        return '/'.join([str(x) for x in params])

    def __contains__(self, items):
        if isinstance(items, list):
            item_as_string = self._parameters_to_string(items)
            return item_as_string in self.parameter_settings
        elif isinstance(items, str):
            return items in self.parameter_settings
        else:
            raise NotImplementedError

    def add(self,
            items: tuple,
            values: np.ndarray
            ) -> None:
        items_string = self._parameters_to_string(items)
        if items_string in self:
            self.parameter_settings[items_string].append(values)
        else:
            self.parameter_settings[items_string] = [values]

    def all_items(self):
        """ returns all stored data."""
        return self.parameter_settings.items()

    def clear(self):
        self.parameter_settings = dict()

    def get_data(self):
        return self.parameter_settings


class ExperimentAnalyzer(object):
    def __init__(self, base_dir, data_file_name):
        self.base_dir = base_dir
        self.data_file_name = data_file_name
        self.param_container = ParamaterContainer()
        self.exp_paths = get_experiment_paths(base_dir)
        print(f'Found {len(self.exp_paths)} files.')
        self.filtered_paths = list()

    def _find_nested_item(self, obj, key):
        if key in obj:
            return obj[key]
        for (k, v) in obj.items():
            if isinstance(v, dict):
                return self._find_nested_item(v, key)  # added return statement

    def _search_nested_key(self, dic, key, default=None):
        """Return a value corresponding to the specified key in the (possibly
        nested) dictionary d. If there is no item with that key, return
        default.
        """
        stack = [iter(dic.items())]
        while stack:
            for k, v in stack[-1]:
                if isinstance(v, dict):
                    stack.append(iter(v.items()))
                    break
                elif k == key:
                    return v
            else:
                stack.pop()
        return default

    def _fill_param_container(self,
                              params: tuple,
                              filter: dict
                              ) -> None:
        """ Fill up the internal data container with the data created in the
            experiments.
            If filter dict is provided, only those keys are processed.
         """
        for path in self.exp_paths:
            # fetch config.json first and determine parameter values#
            config_file_path = os.path.join(path, 'config.json')
            config = get_file_contents(config_file_path)

            # Check if filter matches to current config
            if filter is not None:  # iterates when filter is not empty
                skip_path = False
                for key, v in filter.items():
                    found_value = self._search_nested_key(config, key)
                    # skip if filter does not match
                    if found_value is None:
                        skip_path = True
                        warnings.warn(
                            f'Filter {filter} did not apply at: {path}')
                    if found_value != v:
                        skip_path = True  # skip if filter does not match
                if skip_path:
                    continue

            fetched_config_values = OrderedDict()

            for param in params:
                # config typically holds nested dictionaries...
                is_present = self._search_nested_key(config, param)

                if is_present:
                    fetched_config_values[param] = is_present
                # if parameter is not found, return Not A Number
                else:
                    fetched_config_values[param] = np.NaN
                # assert is_present is not None, \
                #     f'Param: {param} not found in config {config_file_path}'
                # fetched_config_values[param] = is_present
            vals = fetched_config_values.values()

            data_file_path = os.path.join(path, self.data_file_name)
            try:
                data = get_file_contents(data_file_path)
            except ValueError:
                data = get_file_contents(data_file_path, skip_header=True)
            except AssertionError:
                print(f'WARNING: nothing found at: {data_file_path}')
                continue
            assert isinstance(data, np.ndarray)
            # only add data if file is not empty
            if data.any():
                self.param_container.add(vals, data)

    def get_data(self,
                 params: tuple,
                 filter: dict
                 ) -> dict:
        """fetch data from the experiment directories and merge
        runs with same parameters"""

        self.param_container.clear()
        # fill internal data container first
        self._fill_param_container(params, filter=filter)

        return self.param_container.get_data()

    def get_mean_return(self,
                        params: tuple,
                        filter: dict
                        ):
        data_dic = self.get_data(params, filter=filter)
        return_scores = []
        for values in data_dic.values():
            for vs in values:  # iterate over individual runs
                # print(vs)
                mean_return = np.mean(vs)  # build mean of return.csv
                return_scores.append(mean_return)
        return np.mean(return_scores)