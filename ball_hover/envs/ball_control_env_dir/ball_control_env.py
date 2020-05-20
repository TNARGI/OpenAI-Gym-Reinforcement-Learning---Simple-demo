import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class BallControlEnv(gym.Env):
   
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    


    def __init__(self):

        self.debug = ["!method_calls", "!misc", "!kinematics", "!action"]

        if("method_calls" in self.debug): print("ball_hover_env: __INIT__")


        self.gravity = -9.8
        self.cartmass = 1.0
        self.force_mag = 20.0

        self.tau = 0.02 # seconds between updates

        self.y_threshold = 0.8
        self.x_threshold = 0.8

        high = np.array([self.y_threshold*2, np.finfo(np.float32).max, self.x_threshold*2, np.finfo(np.float32).max], dtype=np.float32)
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None





    def seed(self, seed=None):

        if("method_calls" in self.debug): print("ball_hover_env: SEED")

        self.np_random, seed = seeding.np_random(seed)
        return[seed]




    def step(self, action):

        if("method_calls" in self.debug): print("ball_hover_env: STEP")
        
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        state = self.state
        y, y_vel, x, x_vel = state

        if ("action" in self.debug): print("action: " + str(action))

        if(action==1):
            y_force = self.force_mag
            x_force = 0

        elif(action==2):
            y_force = 0
            x_force = self.force_mag

        elif(action==3):
            y_force = 0
            x_force = -self.force_mag
        else:
            y_force = 0
            x_force = 0

        if("kinematics" in self.debug): print("y-pos: " + str(y) + ", y-vel: " + str(y_vel))
        if("kinematics" in self.debug): print("x-pos: " + str(x) + ", x-vel: " + str(x_vel))

        y_acc = (y_force/self.cartmass) + self.gravity
        x_acc = (x_force/self.cartmass)

        y_vel = y_vel + (y_acc * self.tau)
        y = y + (y_vel * self.tau)

        x_vel = x_vel + (x_acc * self.tau)
        x = x + (x_vel * self.tau)

        self.state = (y, y_vel, x, x_vel)

        done = y < -self.y_threshold \
                or y > self.y_threshold \
                or x < -self.x_threshold \
                or x > self.x_threshold

        done = bool(done)


        if not done:
            if("misc" in self.debug): print("not done")
            reward = 1.0
        elif self.steps_beyond_done is None:
            if("misc" in self.debug): print("not done and steps beyond done is none")
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if("misc" in self.debug): print("done")
            if self.steps_beyond_done == 0:
                if("misc" in self.debug): print("done and steps beyond done is none")
                logger.warn("env has returned 'done'. no more steps to take. please call 'reset()'")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}




    def reset(self):

        if("method_calls" in self.debug): print("ball_hover_env: RESET")

        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)




    def render(self, mode="human"):

        if("method_calls" in self.debug): print("ball_hover_env: RENDER")

        screen_width = 600
        screen_height = 600

        #world_width = self.y_threshold*2
        world_width = 6
        #world_width = 60
        
        scale = screen_width/world_width
        
        cart_x = screen_width/2
        cart_y = 100 # top of cart
        cart_width = 40
        cart_height = 40

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l,r,t,b = -cart_width/2, cart_width/2, cart_height/2, -cart_height/2

            cart = rendering.FilledPolygon([
                (l,b),
                (l,t),
                (r,t),
                (r,b)
                ])

            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            #self.track = rendering.Line((screen_width/2, 0), (screen_width/2, screen_height))
            #self.track.set_color = (0,0,0)
            #self.viewer.add_geom(self.track)

            self.top_bound = rendering.Line((0, (screen_height/2 + (self.y_threshold*scale))),(screen_width, (screen_height/2 + (self.y_threshold*scale))))
            self.top_bound.set_color = (1,0,0)
            self.viewer.add_geom(self.top_bound)

            self.bot_bound = rendering.Line((0, (screen_height/2 - (self.y_threshold*scale))),(screen_width, (screen_height/2 - (self.y_threshold*scale))))
            self.bot_bound.set_color = (1,0,0)
            self.viewer.add_geom(self.bot_bound)

            self.left_bound = rendering.Line((screen_width/2-(self.x_threshold*scale), 0),(screen_width/2-(self.x_threshold*scale), screen_height))

            self.left_bound.set_color = (1,0,0)
            self.viewer.add_geom(self.left_bound)

            self.right_bound = rendering.Line((screen_width/2+(self.x_threshold*scale), 0), (screen_width/2+(self.x_threshold*scale), screen_height))
            self.right_bound.set_color = (1,0,0)
            self.viewer.add_geom(self.right_bound)

        if self.state is None: return None
        
        alpha = self.state

        cart_y = alpha[0] * scale + screen_height/2.0
        cart_x = alpha[2] * scale + screen_width/2.0

       
        self.carttrans.set_translation(cart_x, cart_y)

        return self.viewer.render(return_rgb_array = mode=="rgb_array")




    def close(self):

        if("method_calls" in self.debug): print("ball_hover_env: CLOSE")

        if self.viewer:
            self.viewer.close()
            self.viewer = None


