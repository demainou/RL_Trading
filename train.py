from agent import Agent
from helper import getStockData, getState
import sys

# Next, define the number of market days to consider as the window size, and define
# the batch size with which the neural network will be trained, as follows:
window_size = 100
batch_size = 32

# Instantiate the stock agent with the window size and batch size, as follows:
agent = Agent(window_size, batch_size)

# Next, read the training data from the CSV file, using the helper function:
data = getStockData("train500")
l = len(data) - 1

# Next, the episode count is defined as 300. The agent will look at the
# data for so many numbers of times. An episode represents a complete pass over the data:
episode_count = 300

# Next, we can start to iterate through the episodes, as follows:
for e in range(episode_count):
    print("Episode " + str(e) + "/" + str(episode_count))

# Each episode has to be started with a state based on the data and window size. The inventory of stocks is initialized before going through the data:
    state = getState(data, 0, window_size + 1)
    agent.inventory = []
    total_profit = 0
    done = False

# Next, start to iterate over every day of the stock data. The action probability is predicted by the agent,
    #  based on the state:
    for t in range(l):
        action = agent.act(state)
        action_prob = agent.actor_local.model.predict(state)

        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

# The action can be held, if the agent decides not to do anything with the stock.
        #  Another possible action is to buy (hence, the stock will be added to the inventory), as follows:
        if action == 1:
            agent.inventory.append(data[t])
            print("Buy:" + formatPrice(data[t]))

# If the action is 2, the agent sells the stocks and removes it from the inventory.
        #  Based on the sale, the profit (or loss) is calculated:
        elif action == 2 and len(agent.inventory) > 0:  # sell
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)