# Importing libraries
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np
import random
from collections import deque
import streamlit as st

# Loading pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initializing memory and Q-learning hyperparameters
memory = {}
conversation_history = []

# Hyperparameters for Q-learning
l_r = 0.001  # Learning rate
df = 0.9     # Discount factor
eet = 0.1    # Exploration-exploitation tradeoff

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the policy network
input_size = model.config.hidden_size  # Size of the hidden state
hidden_size = 128 #No of neurons in hidden state
output_size = 2  # Number of actions
policy_network = PolicyNetwork(input_size, hidden_size, output_size)
optimizer = optim.Adam(policy_network.parameters(), lr=0.001)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Initializing replay buffer
replay_buffer = ReplayBuffer(capacity=1000)

# Reward Function using sentiment analysis
sentiment_analyzer = pipeline("sentiment-analysis")

def get_reward(user_input, bot_response):
    # Sentiment analysis of the bot response
    sentiment = sentiment_analyzer(bot_response)[0]
    if sentiment["label"] == "POSITIVE":
        return 1
    elif sentiment["label"] == "NEGATIVE":
        return -1
    return 0

# State Representation
def get_state(chat_history_ids):
    # Using last hidden state of the model as state representation
    with torch.no_grad():
        outputs = model(chat_history_ids, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
    return last_hidden_state.cpu().numpy()

# Choosing Action eet greedy strategy
def choose_action(state, eet):
    if random.uniform(0, 1) < eet:
        return random.choice([0, 1])
    else:
        q_values = policy_network(torch.FloatTensor(state))
        return torch.argmax(q_values).item()

# Train Policy Network
def train_policy_network(batch_size):
    if len(replay_buffer) < batch_size:
        return

    # Sample a batch of experiences
    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states = zip(*batch)

    # Converting it to tensors
    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(np.array(actions))
    rewards = torch.FloatTensor(np.array(rewards))
    next_states = torch.FloatTensor(np.array(next_states))

    # Computing Q-values
    q_values = policy_network(states)
    next_q_values = policy_network(next_states)

    # Computing target Q-values
    targets = rewards + df * torch.max(next_q_values, dim=1)[0]

    # Computing MSE loss
    loss = nn.MSELoss()(q_values.gather(1, actions.unsqueeze(1)), targets.unsqueeze(1))

    # Updating policy_network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Streamlit
st.title("Chatbot")
st.write("This is simple chatbot using Reinforcement Learning")

# Initializing session state for chat history
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# User input
user_input = st.text_input("You:", key="user_input")

if user_input:
    if user_input.lower() == "exit":
        st.write("Chatbot: Goodbye!")
    else:
        # Generate response
        if st.session_state.chat_history_ids is None:
            st.session_state.chat_history_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
        else:
            new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
            st.session_state.chat_history_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1)

        # Getting current state
        state = get_state(st.session_state.chat_history_ids)

        # Choosing action
        action = choose_action(state, eet)

        # Generating the model response based on chosen action
        response_ids = model.generate(
            st.session_state.chat_history_ids,
            max_length=st.session_state.chat_history_ids.shape[-1] + 50,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            top_k=50,
            temperature=0.7,
        )
        response = tokenizer.decode(response_ids[:, st.session_state.chat_history_ids.shape[-1]:][0], skip_special_tokens=True)
        st.write(f"Chatbot: {response}")

        # Getting reward
        reward = get_reward(user_input, response)

        # Getting next state
        next_state = get_state(st.session_state.chat_history_ids)

        # Storing experience in the replay buffer
        replay_buffer.push(state, action, reward, next_state)

        # Training policy network
        train_policy_network(batch_size=32)

        # Updating conversation history
        st.session_state.conversation_history.append((user_input, response))

# Display conversation history
st.write("### Conversation History")
for user_msg, bot_msg in st.session_state.conversation_history:
    st.write(f"You: {user_msg}")
    st.write(f"Chatbot: {bot_msg}")
    st.write("---")

