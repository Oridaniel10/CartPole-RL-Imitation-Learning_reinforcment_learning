# 🚀 CartPole: Reinforcement Learning & Imitation Learning

**Link to a summary report with the results and visuals:**  
[Download the file from Google Drive](https://drive.google.com/file/d/1P6yHktBXlVbEodyGWL8F6D9rwpNlT0pY/view)

This repository contains implementations of **reinforcement learning (RL)** and **imitation learning** to solve the CartPole-v1 environment using deep learning techniques.

## 🌜 Project Overview

The goal is to train an agent to balance a pole on a moving cart. The project consists of:
1. **Reinforcement Learning (RL)** – Training an agent using the **REINFORCE policy gradient algorithm**.
2. **Data Collection** – Using the trained RL agent to collect (state, action) pairs.
3. **Imitation Learning (Supervised Learning)** – Training a neural network to imitate the RL agent’s policy.
4. **Evaluation** – Comparing models trained on different dataset sizes and analyzing their performance.

## 📂 Project Structure
- `ATDL303g_PG_reinforce_cartpole(STEP_1).ipynb` - **Training the RL agent** using policy gradients and collection data for next step.
- `imitation_learning+challenge.ipynb` - **Training the imitation learning models** with different dataset sizes.
- `imitation_learning_running_inference_model100.ipynb` - **Running inference on the trained imitation learning model**.
- `Report.pdf` - **Comprehensive project report** detailing implementation, results, and conclusions.

## 📊 Key Results
| Dataset Size | Accuracy | Best Steps Achieved |
|-------------|----------|--------------------|
| 50          | 65%      | 106               |
| 50 (Filtered) | 65%    | 123               |
| 100         | 80%      | 500 (Max)         |
| 500         | 85%      | 500 (Max)         |
| 1000        | 90%      | 500 (Max)         |
| 7000        | 95%      | 500 (Max)         |

### 🎮 Performance Videos
- Videos showing trained models playing CartPole can be found in the report and Google Drive links.

## 🛠 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/CartPole-RL-Imitation-Learning.git
   cd CartPole-RL-Imitation-Learning
   ```
2. Open the notebooks in **Google Colab** or **Jupyter Notebook**.
3. Run the cells to train the models and evaluate their performance.

---

## 🏋️ Reinforcement Learning: Training CartPole Agent

This notebook trains a reinforcement learning agent to solve the CartPole-v1 environment using **policy gradients (REINFORCE algorithm)**.

### 🔹 Key Steps:
1. **Setup Environment** – Install libraries and initialize CartPole-v1.
2. **Define REINFORCE Algorithm** – Build the neural network policy.
3. **Train the RL Agent** – Optimize the policy to maximize episode rewards.
4. **Evaluate Performance** – Run trained models to measure stability.

### 🎯 Training Goals:
- The RL agent is considered **successful** if it reaches an average reward of **480 over 80 episodes** (96% success rate).

---

## 🤖 Imitation Learning: Training from RL Agent

This notebook trains a neural network to **imitate a pre-trained RL agent** using supervised learning.

### 🔹 Key Steps:
1. **Load Data** – Import datasets of (state, action) pairs collected from the RL agent.
2. **Train Neural Network** – Test multiple dataset sizes (50, 100, 500, etc.).
3. **Evaluate Models** – Compare results and determine the best dataset for learning.

### 📊 Key Observations:
- The **100-example dataset** was the most efficient, achieving a **max score of 500**.

---

## 🎥 Running Inference on Model 100

This notebook loads a trained imitation learning model (`model_100`) and runs inference in the CartPole environment.

### 🔹 Key Features:
- Loads a **pre-trained neural network** from `cartpole_imitation_100.h5`.
- Runs **multiple test episodes** and records gameplay videos.
- Compares the agent’s performance with different dataset sizes.

### 🎮 Video Output:
- Generated videos (`gameplay_1.mp4`, `gameplay_2.mp4`, etc.) show how well the trained model balances the pole.

---

## 📚 References
- [Gymnasium Documentation](https://gymnasium.farama.org)
- [Keras API](https://keras.io)
- [TensorFlow Documentation](https://www.tensorflow.org)

