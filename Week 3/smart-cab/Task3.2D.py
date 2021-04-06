#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gym.envs.registration import register

register(
    id='smart_cab-v2',
    entry_point='smart_cab.envs:TaxiEnv')


# In[2]:


import gym

# core gym interface is env
env = gym.make('smart_cab:smart_cab-v2')

env.render()


# In[3]:


# env.reset(): Resets the environment and returns a random initial state.
env.reset() 

# env.render(): Renders one frame of the environment (helpful in visualizing the environment)
env.render()

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

text = """
The filled square represents the taxi, which is yellow without a passenger and green with a passenger.

The pipe ("|") represents a wall which the taxi cannot cross.

A, B, C, D, E, F are the possible pickup and destination locations. 

The blue letter represents the current passenger pick-up location.

The pink letter is the current drop-off location.
"""

print(text)


# In[4]:


# (taxi row, taxi column, passenger location index, drop-off location index)
# Pick-up/Drop-off --> A - 0, B - 1, C - 2, D - 3, E - 4, F - 5
# Manually set the state and  give it to the environment
state = env.encode(0, 1, 2, 3) 
print("State:", state)

# A number is generated corresponding to a state between 0 and 4200, which turns out to be 57.

env.s = state
env.render()


# In[5]:


# Reward Table

text = """
Output is default reward values assigned to each state.

This dictionary has the structure {action: [(probability, nextstate, reward, done)]}.

The 0-5 corresponds to the actions (south, north, east, west, pickup, dropoff) the taxi can perform at our current state in the illustration.

Probability of 1.0 for taking an action to reach a state and 0.0 if the action nvr reach the state

The nextstate is the state we would be in if we take the action at this index of the dict

All the movement actions have a -1 reward, -3 for a wrong pickup and -10 for a wrong dropoff.

If we are in a state where the taxi has a passenger and is on top of the right destination, we would see a reward of 20 at the dropoff action (5)

""done"" is used to tell us when we have successfully dropped off a passenger in the right location. Each successfull dropoff is the end of an episode

If the taxi hits the wall, it will accumulate a -1 as well and this will affect a the long-term reward.
"""

print(text)

env.P[57]


# In[6]:


#print(type(env.P[57]))

from mdp import MDP
"""
for state_num in range(500):
    print(env.P[state_num])
    print()
"""

"""
print(env.P[57])
print()
print(env.P[57][0])
print()
print(env.P[57][0][0])
print()
print(env.P[57][0][0][0])
print()
print(env.P[57][0][0][1])
print()
print(env.P[57][0][0][2])
print()
"""

states = []


for state_num in range(108):
    states.append("s{}".format(state_num))

state_trans_prob = []

for i in range(108):
    dict1 = {}
    dict1['s{}'.format(i)] = env.P[i][0][0][0]
    state_trans_prob.append(dict1)


action_key_list = ['a0', 'a1', 'a2', 'a4', 'a5', 'a6']


transition_probs = {}
for state_num in range(108):
    
    per_state_dict = {}
    
    for action_key in range(6):
        per_state_dict[action_key_list[action_key]] = state_trans_prob[env.P[state_num][action_key][0][1]]
    transition_probs[states[state_num]] = per_state_dict

#print(transition_probs)

rewards = {}
    
for state_num in range(108):
    
    per_state_dict = {}
    
    for action_key in range(6):
        per_action_dict = {}
        per_action_dict['s{}'.format(env.P[state_num][action_key][0][1])] = env.P[state_num][action_key][0][2]
        per_state_dict[action_key_list[action_key]] = per_action_dict
    rewards[states[state_num]] = per_state_dict

#print(rewards)

mdp = MDP(transition_probs, rewards, initial_state='s0')
    


# In[7]:


from mdp import has_graphviz
from IPython.display import display
print("Graphviz available:", has_graphviz)


# In[8]:


if has_graphviz:
    from mdp import plot_graph, plot_graph_with_state_values, plot_graph_optimal_strategy_and_state_values
    display(plot_graph(mdp))


# In[25]:


import numpy as np


# initialise a random policy where for each action for all states is initialised with a random value
random_policy = np.ones([env.nS, env.nA]) / env.nA

total_epochs, total_rewards = 0, 0

all_epochs = []
all_reward = []

total_episodes = 100

for i in range(total_episodes):
    state = env.reset()
    epoch_per_episode = 0
    reward_per_episode = 0
    done = False
    for i in range(1000):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)  
        epoch_per_episode = epoch_per_episode + 1
        reward_per_episode = reward_per_episode + reward
        
        if done:
            break
    
    total_epochs = total_epochs + epoch_per_episode
    total_rewards = total_rewards + reward_per_episode
    
    all_epochs.append(epoch_per_episode)
    all_reward.append(reward_per_episode)

mean_steps_per_episode = total_epochs/total_episodes
mean_reward_per_episode = total_rewards/total_episodes

print("The average steps per episode is {}".format(mean_steps_per_episode))
print("The average reward per episode is {}".format(mean_reward_per_episode))
   
        
    


# In[14]:


"""
Evaluate a policy given an environment and a full description of the environment's dynamics.

Args:
    policy: [S, A] shaped matrix representing the policy.
    
    env: OpenAI env. env.P represents the transition probabilities of the environment.
        env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
        env.nS is a number of states in the environment. 
        env.nA is a number of actions in the environment.
    
    theta: We stop evaluation once our value function change is less than theta for all states.
    
    discount_factor: Gamma discount factor.

Returns:
    Vector of length env.nS representing the value function.
    
"""
# Initialise the state-value with 0
V = np.zeros(env.nS)
# initialise a random policy for which the value of each action in all state is the probability of taking an action
random_policy = np.ones([env.nS, env.nA]) / env.nA

# policy evaluation
def policy_eval(policy, discount_factor=1.0, theta=0.00001):
    
    for i in range(1000):

        #delta = change in value of state from one iteration to next
        # there is no change so initialise with 0
        delta = 0  

        #for all states
        for state in range(env.nS):  
            #print(state)

            #initiate value of the state as 0
            val = 0  

             #for all actions/action probabilities
            for action, action_probability in enumerate(random_policy[state]):
                #print(action, act_prob)

                #transition probabilities,state,rewards of each action
                for trans_prob, next_state, reward, done in env.P[state][action]:
                    #print(prob, next_state, reward, done)

                    # equation to calculate the value of the state
                    # action_probability = probability of taking action a in state s under policy π
                    val = val + (action_probability * trans_prob) * (reward + discount_factor * V[next_state])

                    # the change would be the max value between the initial change and the current change in value
                    delta = max(delta, np.abs(val-V[state]))

                    # the current state would have that value
                    V[state] = val

        #break if the change in value is less than the threshold (theta)
        if delta < theta: 
                break
    
    return np.array(V)
    


# In[23]:


# policy iteration


def action_value(state, A):
    discount_factor = 0.95
    
    """
    Helper function to calculate the value for all action in a given state.

    Args:
        state: The state to consider (int)
        V: The value to use as an estimator, Vector of length env.nS

    Returns:
        A vector of length env.nA containing the expected value of each action.
    """
    
    A = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[state][a]:
            A[a] =  A[a] + prob * (reward + discount_factor * V[next_state])
    return A



"""
 Policy Improvement Algorithm. Iteratively evaluates and improves a policy until an optimal policy is found.
    
   
    env: The OpenAI envrionment.

    policy_eval_fn: Policy Evaluation function that takes 3 arguments:
        policy, env, discount_factor.

    discount_factor: gamma discount factor.
    
        
    Returns:
        A tuple (policy, V). 
        
        Policy is the optimal policy, a matrix of shape [S, A] where each state s contains a valid 
        probability distribution over actions.
        
        V is the value function for the optimal policy.

"""

for i in range(1000):
    
    # evaluate current policy
    curr_pol_val = policy_eval(policy = random_policy, discount_factor=0.95, theta=0.00001)
    
    # Check if policy did improve (Set it as True first)
    policy_stable = True  
    
    # for each states
    for state in range(env.nS):
        
        # best action (Highest prob) under current policy
        chosen_act = np.argmax(random_policy[state])
        
        # find action values in that given state
        act_values = action_value(state, curr_pol_val) 
        
        # policy improvement
        #find best action
        best_act = np.argmax(act_values) 
        
        if chosen_act != best_act:
            #Greedily find best action
            policy_stable = False  
        
        #update     
        random_policy[state] = np.eye(env.nA)[best_act]  
    
    if policy_stable:
        
        print(random_policy)
        
        print(curr_pol_val)
        break
        


# In[28]:


def view_policy(policy):
    curr_state = env.reset()
    counter = 0
    reward = None
    while reward != 20:
        state, reward, done, info = env.step(np.argmax(policy[curr_state])) 
        curr_state = state
        counter += 1
        env.s = curr_state
        env.render()


# In[ ]:


view_policy(random_policy)


# In[ ]:




