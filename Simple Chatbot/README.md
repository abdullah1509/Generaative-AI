# Reinforcement Learning Chatbot with DialoGPT

### Screenshot 1
![Screenshot (21)](https://github.com/user-attachments/assets/acc0982a-6046-4590-8b9f-9023f9b88228)

### Screenshot 2
![Screenshot (22)](https://github.com/user-attachments/assets/c273059c-1541-44c6-8a8b-fc05dbafedc0)

---

## Project Overview
This project implements a Reinforcement Learning (RL)-based chatbot using the DialoGPT-medium model from Microsoft. The chatbot learns from user interactions by using a Q-learning algorithm with a policy network. The reward function is based on sentiment analysis, which provides feedback on the quality of the bot's responses.

The chatbot is built using:
* **PyTorch** for the RL components.
* **Hugging Face** Transformers for the pre-trained DialoGPT model.
* **Streamlit** for the user interface.

---

## Features
* **Interactive Chatbot:** Users can interact with the chatbot in real-time via a web interface.
* **Reinforcement Learning:** The chatbot learns from user interactions using Q-learning.
* **Sentiment-Based Rewards:** The reward function uses sentiment analysis to evaluate the quality of the bot's responses.
* **Conversation History:** The chatbot maintains a history of the conversation for context-aware responses.
---

## Installation

### Prerequisites
* Python 3.8 or higher
* pip (Python package manager)

### Steps

### 1. Clone the repository:
```
git clone https://github.com/abdullah1509/Generative-AI.git
cd Simple Chatbot
```

### 2. Install the required dependencies:
```
pip install -r requirements.txt
```

### 3. Run the Streamlit app:
```
streamlit run simplechat.py
```

### 4. Access the chatbot:
You'll be redicrted to web browser accessing the chatbot. If not then open your browser and go to ```http://localhost:8501``` to interact with the chatbot.


---

## How It Works
* **User Input:** The user types a message in the Streamlit interface.
* **State Representation:** The chatbot encodes the conversation history and extracts the last hidden state from the DialoGPT model as the state representation.
* **Action Selection:** The chatbot uses an epsilon-greedy strategy to choose an action (e.g., continue or stop the conversation).
* **Response Generation:** The chatbot generates a response using the DialoGPT model.
* **Reward Calculation:** The reward is calculated based on the sentiment of the bot's response (positive, negative, or neutral).
* **Training:** The chatbot updates its policy network using Q-learning and stores the experience in a replay buffer.
* **Conversation History:** The conversation history is updated and displayed in the Streamlit interface.
___

## Dependencies
The project uses the following Python libraries:
* ```torch:``` For building and training the policy network.
* ```transformers:``` For loading the DialoGPT model and tokenizer
* ```streamlit:``` For building the web interface.
* ```numpy:``` For numerical operations.
* ```scipy:``` For sentiment analysis.

You can install all dependencies using:
```
pip install -r requirements.txt
```

