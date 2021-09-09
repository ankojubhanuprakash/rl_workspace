#from _typeshed import Self
import gym
#import torch
import random 

class ActionWr(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        super(ActionWr,self).__init__(env)
        self.epsilon =epsilon
    def action(self,action):
        if random.random()>self.epsilon:
            print("random")
            return self.env.action_space.sample()
        return action
    def pp(self):
        print(self.epsilon)    
if __name__ == "__main__":
    env = ActionWr(gym.make("CartPole-v0"))  
    obs = env.reset()
    total_reward = 0
    while True:
        obs, reward, done, _ = env.step(0)
        total_reward += reward
        env.render() 
        if done:
            break
    print(total_reward)    
             