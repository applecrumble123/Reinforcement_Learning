import sys
from contextlib import closing
from io import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np



MAP = [
    "+-------------------+",
    "| :A| : :B: : | :C| |",
    "| : | : | : | : | : |",
    "| : | : | : | : | : |",
    "| : : : | : : : : : |",
    "| : | : : : | : | : |",
    "| : | : | : | : | : |",
    "| : | : : : | : | : |",
    "| | : : | : : | : : |",
    "| :D| : :E: : : |F| |",
    "| | : : | : | | : : |",
    "+-------------------+",
]

class TaxiEnv(discrete.DiscreteEnv):
    """
    Description:
    There are 8 designated locations in the grid world indicated by A, B, C, D, E, F . When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off, the episode ends.

    Observations:
    There are 4200 discrete states since there are 100 taxi positions, 7 possible locations of the passenger (including the case when the passenger is in the taxi), and 6 destination locations. 

    Passenger locations:
    - 0: A
    - 1: B
    - 2: C
    - 3: D
    - 4: E
    - 5: F
    - 6: in taxi
    
    Destinations:
    - 0: A
    - 1: B
    - 2: C
    - 3: D
    - 4: E
    - 5: F
    
    
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move up
    - 1: move down
    - 2: move left
    - 3: move right
    - 4: pickup passenger
    - 5: drop off passenger

    Rewards:
    There is a default per-step reward of -1,
    except for delivering the passenger, which is +10,
    or executing "pickup" and "drop-off" actions illegally, which is -5.

    Rendering:
    - blue: passenger
    - magenta: destination
    - red: empty taxi
    - green: full taxi
    - other letters (A, B, C, D, E, F, G, H): locations for passengers and destinations
    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype='c')

        self.locs = locs = [(0, 1), (0, 4), (0, 8), (8, 1), (8, 4), (8, 8)]

        num_states = 4200
        num_rows = 10
        num_columns = 10
        max_row = num_rows - 1
        max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        num_actions = 6
        P = {state: {action: []
                     for action in range(num_actions)} for state in range(num_states)}
        for row in range(num_rows):

            for col in range(num_columns):

                for pass_idx in range(len(locs) + 1):  # +1 for being inside taxi

                    for dest_idx in range(len(locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)

                        if pass_idx < 6 and pass_idx != dest_idx:
                            initial_state_distrib[state] += 1

                        for action in range(num_actions):
                            # defaults
                            new_row, new_col, new_pass_idx = row, col, pass_idx
                            reward = -1  # default reward when there is no pickup/dropoff
                            done = False
                            taxi_loc = (row, col)

                            if action == 0:
                                new_row = min(row + 1, max_row)

                            elif action == 1:
                                new_row = max(row - 1, 0)

                            if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                                new_col = min(col + 1, max_col)

                            elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                                new_col = max(col - 1, 0)

                            elif action == 4:  # pickup

                                if (pass_idx < 6 and taxi_loc == locs[pass_idx]):
                                    new_pass_idx = 6

                                else:  # passenger not at location
                                    reward = -3

                            elif action == 5:  # dropoff

                                # if the taxi decides to drop off and it reaches the correct destination after exploring all the destination, reward is 10 and the experiment stop
                                if (taxi_loc == locs[dest_idx] and pass_idx == 6):
                                    new_pass_idx = dest_idx
                                    done = True
                                    reward = 10

                                # if the taxi decides to drop off and hasnt reach the correct destination, reward is -5
                                elif (taxi_loc in locs):
                                    new_pass_idx = locs.index(taxi_loc)

                                    if (new_pass_idx != dest_idx):
                                        done = False
                                        reward = -5

                              

                            new_state = self.encode(new_row, new_col, new_pass_idx, dest_idx)

                            P[state][action].append((1.0, new_state, reward, done))

        initial_state_distrib /= initial_state_distrib.sum()

        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib)

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        # (10) 10, 7, 6
        i = taxi_row
        i *= 10
        i += taxi_col
        i *= 7
        i += pass_loc
        i *= 6
        i += dest_idx
        return i

    def decode(self, i):
        out = []
        out.append(i % 6)
        i = i // 6
        out.append(i % 7)
        i = i // 7
        out.append(i % 10)
        i = i // 10
        out.append(i)
        assert 0 <= i < 10
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)
        

        #print("taxi_row:{}, taxi_col: {}, pass_idx: {}, dest_idx: {}".format(taxi_row, taxi_col, pass_idx, dest_idx))
        

        def ul(x): return "_" if x == " " else x

        if pass_idx < 8:
            #print("pass_idx: {}".format(pass_idx))
            #print("[1 + taxi_row]: {}, [2 * taxi_col + 1]: {}".format([1 + taxi_row],[2 * taxi_col + 1]))
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], 'red', highlight=True)
            #print("out[1 + taxi_row][2 * taxi_col + 1]: {}".format(out[1 + taxi_row][2 * taxi_col + 1]))
            
            pi, pj = self.locs[pass_idx]
            #print("\npi: {}, pj: {}".format(pi, pj))
            #print("[1 + pi]: {}, [2 * pj + 1]: {}".format([1 + pi],[2 * pj + 1]))
            out[1 + pi][2 * pj + 1] = utils.colorize(
                out[1 + pi][2 * pj + 1], 'blue', bold=True)
            #print("out[1 + pi][2 * pj + 1]: {}".format(out[1 + pi][2 * pj + 1]))
            
        else:  # passenger in taxi
            #print("[1 + taxi_row]: {}, [2 * taxi_col + 1]: {}".format([1 + taxi_row],[2 * taxi_col + 1]))
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), 'green', highlight=True)
            #print("out[1 + taxi_row][2 * taxi_col + 1]: {}".format(out[1 + taxi_row][2 * taxi_col + 1]))

        di, dj = self.locs[dest_idx]
        #print("\ndi: {}, dj: {}".format(di, dj))
        #print("[1 + di]: {}, [2 * dj + 1]: {}".format([1 + di],[2 * dj + 1]))
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')
        #print("out[1 + di][2 * dj + 1]: {}".format(out[1 + di][2 * dj + 1]))
        
        
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Down", "Up", "Right", "Left", "Pickup", "Dropoff"][self.lastaction]))
        else:
            outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
