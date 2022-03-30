from os import path, listdir, mkdir
import numpy as np
import sys
import math
import gym
from Base.bp2DPlot import plot_packing_state
from Base.bpReadWrite import ReadWrite
import torch
from BoxPlacementEnvironment import BoxPlacementEnvironment
from itertools import count
from TrainRL import policy_net, select_action, device, EPS_START, EPS_END, EPS_DECAY, steps_done

input_folder = './our_tests'
output_folder = './results'

policy_net.load_state_dict(torch.load('./policy_net_13200.pytorch_model'))
policy_net.eval()

EPS_START = 0.05
EPS_END = 0.05
EPS_DECAY = 100

def solve(test_state):
    with torch.no_grad():
        env = gym.make('BoxPlacementEnvironment-v0', bpState=test_state).unwrapped
        state = torch.from_numpy(env._next_observation().astype(np.float32)).unsqueeze(0)
        for t in count():
            # Select and perform an action
            action = select_action(state)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            if reward == 0: # placement success
                print('count ', t, ' action ', action.item(), ' reward ', reward.item(), ' done ', done, ' ', env.nr_remaining_boxes, ' eps ', EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY))
            if not done:
                next_state = torch.from_numpy(env._next_observation().astype(np.float32)).unsqueeze(0)
            else:
                next_state = None

            # Move to the next state
            state = next_state

            if done:
                # plot_packing_state(env.bpState, fname='./vis/{}'.format(test_instance))
                break
        return env.bpState

def main():
    cases = []
    for f in listdir(input_folder):
        cases.append(f)
    if not path.exists(output_folder):
        mkdir(output_folder)
    

    for case in cases:
        test_case = ReadWrite.read_state(path=path.join(input_folder, case))
        print('Solving case ', case)
        solution = solve(test_case)
        print(solution)
        test_case = ReadWrite.read_state(path=path.join(input_folder, case))
        plot_packing_state(solution, fname='./vis/{}'.format('test'))
        
        if not solution.is_valid(test_case):
            print(f'alarm, wrong solution!!! on {case}')
            break
        
        ReadWrite.write_state(path=path.join(output_folder, case), state=solution)

if __name__ == '__main__':
    sys.exit(main())