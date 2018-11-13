from actor import Actor
from critic import Critic

import numpy as np
from numpy.random import choice
import random

from collections import namedtuple, deque

# Create a ReplayBuffer class that adds, samples, and evaluates a buffer
class ReplayBuffer:
    # Fixed sized buffer to stay experience tuples

    def __init__(self, buffer_size, batch_size):
        # Initialize a replay buffer object.

        # parameters

        # buffer_size: maximum size of buffer. Batch size: size of each batch

        self.memory = deque(maxlen=buffer_size)  # memory size of replay bufferooks.

        self.batch_size = batch_size  # Training batch size for Neural nets
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state",
                                                                "done"])  # Tuple containing experienced replay

    # Add a new experience to the replay buffer memory:

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    # randomly sample states from a memory buffer.We do this so that the states
    # that we feed to the model are not temporally correlated.This will reduce overfitting
    “

    def sample(self, batch_size=32):
        return random.sample(self.memory, k=self.batch_size)

    # Return the current size of the buffer memory, as follows:
    def __len__(self):
        return len(self.memory)

    # The reinforcement learning agent that learns using the actor-critic net work is as follows:

    class Agent:
        def __init__(self, state_size, batch_size, is_eval=False):
            self.state_size = state_size

    The number of ctions are defined as 3: sit, buy, sell
    self.action_size = 3

    # Define the replay memory size
    self.buffer_size = 1000000
    self.batch_size = batch_size
    self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
    self.inventory = []

    # Define whether or not training is ongoing
    self.is_eval = is_eval

    Discount factor in Bellman equation:
    self.gamma = 0.99

    # “A soft update of the actor and critic networks can be done as follows:

    self.tau = 0.001

    # “The actor policy model maps states to actions and instantiates the
    # “actor networks (local and target models, for soft updates of parameters
    self.actor_local = Actor(self.state_size, self.action_size)
    self.actor_target = Actor(self.state_size, self.action_size)

    # “The critic (value) model that maps the state-action pairs to Q_values is
    self.critic_local = Critic(self.state_size, self.action_size)

    # Instantiate the critic model (the local and target models are utilized to allow for soft updates), as follows:
    self.critic_target = Critic(self.state_size, self.action_size)
    self.critic_target.model.set_weights(self.critic_local.model.get_weights())

    # The following code sets the target model parameters to local model parameters:
    self.actor_target.model.set_weights(self.actor_local.model.get_weights()

    # Returns an action, given a state, using the actor (policy network) and the output of the softmax layer of the actor-network, returning the
    # probability for each action. An action method that returns an action, given a state, using the actor (policy network) is as follows:
    def act(self, state):
        options = self.actor_local.model.predict(state)
        self.last_state = state
        if not self.is_eval:
            return choice(range(3), p = options[0])
        return np.argmax(options[0])

    # Returns a stochastic policy, based on the action probabilities in the training model and
    # a deterministic action corresponding to the maximum probability during testing. There is
    # a set of actions to be carried out by the agent at every step of the episode. A method
    # (step) that returns the set of actions to be carried out by the agent at every step of the
    # episode is as follows:

    def step(self, action, reward, next_state, done):

        # The following code adds a new experience to the memory:
        self.memory.add(self.last_state, action, reward, next_state, done):

        # The following code asserts that enough experiences are present in the memory to train:
            if len(self.memory) > self.batch_size:

                # The following code samples a random batch from the memory to train:
                experiences = self.memory.sample(self.batch_size)

                # Learn from the sampled experiences, as follows:
                self.learn(experiences)

                # The following code updates the state to the next state:
                self.last_state = next_state

    # Learning from the sampled experiences through the actor and the critic. 
    # Create a method to learn from the sampled experiences through the actor and the critic, as follows:
    def learn(self, experiences):
        states = np.vstack([e.state for e in experiences if e is not None]).astype(np.float32).reshape(-1,self.state_size)
        actions = np.vstack([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1,self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1,1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.float32).reshape(-1,1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None]).astype(np.float32).reshape(-1,self.state_size)

        # Return a separate array for each experience in the replay component and predict actions based on the next states, as follows:
        actions_next = self.actor_target.model.predict_on_batch(next_states)

        # Predict the Q_value of the actor output for the next state, as follows:
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # target the Q_value to serve as a label for the critic network, based on the temporal difference, as follows:
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)

        # Fit the critic model to the time difference of the target, as follows:
        self.critic_local.model.train_on_batch(x = [states, actions], y = Q_targets)

        # Train the actor model (local) using the gradient of the critic network output with
        # respect to the action probabilities fed from the actor-network:
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]),
                                      (-1, self.action_size))

        # Next, define a custom training function, as follows:
        self.actor_local.train_fn([states, action_gradients, 1])

        # Next, initiate a soft update of the parameters of both networks, as follows:
        self.soft_update(self.actor_local.model, self.actor_target.model)

    # This performs soft updates on the model parameters, based on the parameter tau
    # to avoid drastic model changes. A method that updates the model by performing soft
    # updates on the model parameters, based on the parameter tau (to avoid drastic model changes), is as follows:
    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        assert len(local_weights) == len(target_weights)
        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

