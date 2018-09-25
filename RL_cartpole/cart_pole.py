import gym
import numpy as np
from flask import Flask
import json

app = Flask(__name__, static_folder='.')

def play(env, policy):
    observation = env.reset()
    done = False
    score = 0
    observations = []

    for _ in range(5000):
        observations += [observation.tolist()] # Record the observations for normalization and replay
    
        if done: # If the simulation was over last iteration, exit loop
            break
        
        # Pick an action according to the policy matrix
        outcome = np.dot(policy, observation)
        action = 1 if outcome < 0 else 0
        
        # Make the action, record reward
        observation, reward, done, info = env.step(action)
        score += reward

    return score, observations

env = gym.make('CartPole-v1')
max = (0, [], [])
for _ in range(10):
  policy = np.random.rand(1,4)
  score, observations = play(env, policy)
  
  if score > max[0]:
    max = (score, observations, policy)

print('Max Score', max[0])

@app.route("/data")
def data():
    return json.dumps(max[1])

@app.route('/')
def root():
    return app.send_static_file('./index.html')
    
app.run(host='0.0.0.0')