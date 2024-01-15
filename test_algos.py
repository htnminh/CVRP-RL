from delivery import Delivery

import warnings
from pprint import pprint

import numpy as np

import pandas as pd

from stable_baselines3 import DQN, A2C, PPO

def test_delivery(delivery: Delivery, model_class, model_name: str,
                  total_timesteps_list: list, run_times_each=10):
    """Train and run the model for run_times_each times for each total_timesteps in total_timesteps_list"""
    N_MOVE_THRESHOLD = delivery.n_stops * 100
    
    _full_result = list()
    result = list()

    for total_timesteps in total_timesteps_list:
        observation, info = delivery.reset()
        model = model_class("MultiInputPolicy", delivery, verbose=1)
        model.learn(total_timesteps=total_timesteps, progress_bar=False)
        last_solution_info = delivery.last_solution_info.copy()
        model.save(f"models/{model_name}_{total_timesteps}")
        model = model_class.load(f"models/{model_name}_{total_timesteps}")
        
        for _ in range(run_times_each):
            fallback = False
            observation, info = delivery.reset()
            while True:
                action, _states = model.predict(observation, deterministic=False)
                observation, reward, terminated, truncated, info = delivery.step(action)
                # pprint(delivery.observation)
                if terminated or truncated:
                    break
                elif info["n_routes_redundant"] > N_MOVE_THRESHOLD:
                    warnings.warn(f'The algorithm took too long to give a solution, '
                                'falling back to the last found solution during training.')
                    fallback = True
                    break

            if not fallback:
                final_info = delivery.pre_reset_info.copy()
            else:
                final_info = last_solution_info.copy()
            _full_result.append(final_info)
            
        result.append(
            np.min([info['ortools_format_solution']['objective'] for info in _full_result[-run_times_each:]])
        )

    return result

if __name__ == '__main__':
    # constants
    models = (
        (A2C, 'a2c'),
        (DQN, 'dqn'),
        (PPO, 'ppo'),
    )
    model_names = [model_name for _, model_name in models]

    TOTAL_TIMESTEPS_LIST = [250, 500, 1_000, 5_000, 10_000, 50_000]
    # TOTAL_TIMESTEPS_LIST = [250, 500]  # test
    N_STOPS_LIST = [5, 6, 7, 8, 9, 10, 12, 15, 20]
    # N_STOPS_LIST = [5, 10]  # test

    # train and save results of each model
    for model, model_name in models:
        df = pd.DataFrame(columns=['n_stops'] + TOTAL_TIMESTEPS_LIST).set_index('n_stops')
        for n_stops in N_STOPS_LIST:
            delivery = Delivery(n_stops=n_stops, gen_seed=n_stops, gym_seed=n_stops, max_env_size=1000, print_terminated=False)
            test_result_row = [n_stops] + test_delivery(delivery, model, model_name, TOTAL_TIMESTEPS_LIST)
            # print(test_result_row)
            _new_row = pd.DataFrame([test_result_row],
                                    index=[n_stops], columns=['n_stops'] + TOTAL_TIMESTEPS_LIST).set_index('n_stops')
            df = pd.concat([df, _new_row])
            # print(df)
            df.to_csv(f"results/{model_name}.csv")

    # concat best results + ortools results 
    df_best = pd.concat([
        pd.read_csv(f"results/{model_name}.csv", index_col='n_stops').min(axis=1) for model_name in model_names
    ], axis=1).rename(columns=dict(zip(range(len(models)), [model_name for model_name in model_names])))

    ortools_result = []
    for n_stops in N_STOPS_LIST:
        delivery = Delivery(n_stops=n_stops, gen_seed=n_stops, gym_seed=n_stops, max_env_size=1000, print_terminated=False)
        ortools_solution = delivery.get_ortools_solution(time_limit_seconds=10)
        ortools_result.append([n_stops, ortools_solution['objective']])
    
    df_best = pd.concat([
        df_best,
        pd.DataFrame(ortools_result, columns=['n_stops', 'ortools']).set_index('n_stops')
    ], axis=1)

    # save results
    df_best.to_csv(f"results/best.csv")

