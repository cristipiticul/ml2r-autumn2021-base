# Inspired from https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
# Defines the Environment in the OpenAI gym interface
# State:
# - 2 matrices: 
#   - the current bin occupation (10x10 matrix, 0->free, 1->occupied) 
#   - the box counts (10x10 matrix: for each box size, count how many open boxes are)
# Action: a number between 0 and 101
# - 0: close the current bin and open a new one
# - 1-101: place one box of that size in the current bin
# Step function: how an action modifies the current state
# - action 0 
#   => bin occupation matrix is reset to 0
#   => reward is 50-number_of_free_spaces (if the bin is 50% full, reward is 0, if it's empty, reward is -50)
# - action 1-101
#   => try to place the box using bpState.place_box_in_bin
#   => if it succeeds, update the bin occupation and reward is 0
#   => if it fails, reward is -10
import gym
from gym import spaces
from numpy import dtype
import numpy
from Base.bp2DPnt import Point

from Base.bp2DState import State

NEW_BOX_REWARD = lambda bin: 90 - 3 * len(bin.pnts_open) # reward depending on bin fill
TRY_TO_PUT_INEXISTENT_BOX_REWARD = -150
TRY_TO_PUT_BOX_INVALID_POSITION_REWARD = -10
def BOX_PLACED_SUCCESSFULLY_REWARD(bin, box): 
    initial_free = len(bin.pnts_open) + box.a
    if initial_free > 50:
        return 0
    return box.a * 15 / initial_free

class BoxPlacementEnvironment(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, bpState: State):
        super(BoxPlacementEnvironment, self).__init__()
        # needed for self.reset()
        self.initialBpState = bpState 
        
        # Will be modified when actions occur, so we create a copy
        self.bin_w = bpState.bin_size[0]
        self.bin_h = bpState.bin_size[1]

        self.reset()
        
        # Actions:
        # - close this bin and open a new one (1 action)
        # - place box of size (box_w, box_h) 
        self.action_space = spaces.Discrete(1 + self.bin_w * self.bin_h)
        # State:
        # - bin occupation (matrix of 1s and 0s)
        # - open boxes (matrix of counts of boxes)
        self.observation_space = spaces.Tuple(
            [spaces.Discrete(1) for i in range(self.bin_w * self.bin_h)] + 
            [spaces.Discrete(1000) for i in range(self.bin_w * self.bin_h)]
        )

    def step(self, action):
        reward = None
        # if action[0] == 1:
        if action == 0:
            reward = NEW_BOX_REWARD(self.bpState.bins[-1])
            self.bpState.open_new_bin()
            self.bin_occupation = numpy.zeros(self.bpState.bin_size, dtype=numpy.int8)
        else:
            box_w = (action - 1) // self.bin_h
            box_h = (action - 1) % self.bin_h
            if self.box_counts[box_w][box_h] == 0:
                reward = TRY_TO_PUT_INEXISTENT_BOX_REWARD
            else:
                last_box = self.boxes_by_size[box_w][box_h][-1]
                
                # Place the box in the current (last) bin
                last_bin_index = len(self.bpState.bins) - 1
                success = self.bpState.place_box_in_bin(last_box, last_bin_index)
                if success:
                    reward = BOX_PLACED_SUCCESSFULLY_REWARD(self.bpState.bins[last_bin_index], last_box)
                    self.nr_remaining_boxes -= 1
                    self.box_counts[box_w][box_h] -= 1
                    self.boxes_by_size[box_w][box_h].pop()
                    last_box_index = [i for (i, box) in enumerate(self.bpState.boxes_open) if box.n==last_box.n][0]
                    self.bpState.boxes_open.pop(last_box_index)
                    self.bin_occupation = numpy.ones((self.bin_w, self.bin_h), dtype=numpy.int8)
                    for p in self.bpState.bins[last_bin_index].pnts_open:
                        self.bin_occupation[p.coord[0]][p.coord[1]] = 0
                else:
                    reward = TRY_TO_PUT_BOX_INVALID_POSITION_REWARD
        done = self.nr_remaining_boxes == 0
        return self._next_observation(), reward, done, {}
    
    # Returns the initial state (empty bin, all open boxes counts)
    def reset(self, bpState = None):
        if bpState != None:
            self.initialBpState = bpState

        # Reset also the bpState
        self.bpState = State(len(self.initialBpState.bins), self.initialBpState.bin_size, self.initialBpState.boxes_open.copy())
        
        self.bin_occupation = numpy.zeros((self.bin_w, self.bin_h), dtype=numpy.int8)

        self.nr_remaining_boxes = len(self.bpState.boxes_open)
        self.box_counts = numpy.zeros((self.bin_w, self.bin_h), dtype=numpy.int32)
        self.boxes_by_size = [[[] for _ in range(self.bin_h)] for _ in range(self.bin_w)]
        for box in self.bpState.boxes_open:
            self.box_counts[box.w-1][box.h-1] += 1
            self.boxes_by_size[box.w-1][box.h-1].append(box)
        
        return self._next_observation()

    def _next_observation(self):
        return numpy.array([self.bin_occupation, self.box_counts / self.nr_remaining_boxes]).flatten()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass