import gym
import envs
import numpy as np
import random
import time
import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from rl.agents import SARSAAgent
from rl.policy import EpsGreedyQPolicy


env = gym.make("BallHoverEnv-v0")


"""
states = env.observation_space.shape[0]
print("States: " + str(states))

actions = env.action_space.n
print("Actions: " + str(actions))
"""



##----- MAIN -----##

mode = "random"

if(mode == "random"):

    totalEpisodes = 5


    print("begin")

    for episode in range(1, totalEpisodes+1):

        #print("episode: " + str(episode))

        state = env.reset() # reset environment
        done = False # make sure the game is not over
        score = 0 # reset the score
    
        while not done:
            env.render() # visualise each step
        
            time.sleep(0.05)
        				 		
            action = random.choice([0,1]) # take a random action
            n_state, reward, done, info = env.step(action) # execute action
            score += reward	# keep track of rewards

        print("episode {} score {}".format(episode, score))






if(mode == "learning"):

    def agent(states, actions):
        model = Sequential()
        model.add(Flatten(input_shape=(1, states)))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(actions, activation="linear"))
        return model

    model = agent(env.observation_space.shape[0], env.action_space.n)



    policy = EpsGreedyQPolicy()

    sarsa = SARSAAgent(model=model, policy=policy, nb_actions=env.action_space.n)

    sarsa.compile("adam", metrics=["mse"])

    sarsa.fit(env, nb_steps=10000, visualize=False, verbose=0)




    scores = sarsa.test(env, nb_episodes=50, visualize=True)

    print('Average score over 100 test games:{}'.format(np.mean(scores.history['episode_reward'])))

    #sarsa.save_weights('sarsa_weights.h5f', overwrite=True) # save trained weights

    # sarsa.load_weights('sarsa_weights.h5f') # can be used to load trained weights

    _ = sarsa.test(env, nb_episodes = 50, visualize= False)

