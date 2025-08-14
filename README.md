# Assetto Corsa AI Bot

An AI-powered racing bot for Assetto Corsa that uses Proximal Policy Optimization (PPO) reinforcement learning to learn how to drive cars autonomously around race tracks.

## 🏎️ Project Overview

This project creates an AI agent that can drive cars in Assetto Corsa by:
- Reading real-time telemetry data from the game through shared memory
- Processing this data to understand the car's state (position, velocity, wheel slip, etc.)
- Using a PPO neural network to make driving decisions (gas, brake, steering)
- Sending control inputs back to the game through virtual gamepad emulation
- Learning from experience to improve driving performance over time

## 🧠 How It Works

### 1. **Data Collection** (`sim_info.py`)
- Connects to Assetto Corsa's shared memory to read real-time telemetry
- Extracts physics data (speed, wheel slip, tire wear, position) and graphics data (car coordinates, lap progress)
- Provides a clean interface for other components to access game state

### 2. **State Processing** (`utils.py`)
- Converts raw telemetry into a normalized state vector for the AI
- Handles game control through virtual gamepad emulation (using ViGEm)
- Manages episode lifecycle (restart, step, reward calculation)
- Implements reward function based on:
  - Progress along the track (normalized position)
  - Speed maintenance
  - Lap completion bonuses
  - Penalties for going off-track or getting stuck

### 3. **AI Architecture** (`PPO.py`)
- **Actor-Critic Network**: Neural network with separate actor (policy) and critic (value) components
- **Continuous Action Space**: Outputs gas (0-1), brake (0-1), and steering (-1 to 1)
- **PPO Algorithm**: Proximal Policy Optimization for stable policy learning
- **Experience Buffer**: Stores trajectories for batch learning updates

### 4. **Training Loop** (`train.py`)
- Orchestrates the complete training process
- Manages episodes, timesteps, and model updates
- Handles logging, checkpointing, and hyperparameter tuning
- Implements action standard deviation decay for exploration vs exploitation balance

## 🚀 Getting Started

### Prerequisites
- **Assetto Corsa** (with a track and car loaded)
- **Python 3.8+** with PyTorch
- **ViGEm Bus Driver** for virtual gamepad emulation
- **vgamepad** Python library for gamepad control

### Installation
1. Install ViGEm Bus Driver from [ViGEm's GitHub](https://github.com/ViGEm/ViGEmBus)
2. Install Python dependencies:
   ```bash
   pip install torch numpy vgamepad
   ```
3. Ensure Assetto Corsa is running with a track loaded

### Running the Bot

#### Training Mode
```bash
python train.py
```
This will start the PPO training process, where the AI learns to drive through trial and error.

#### Testing Individual Components
- **Data Reading Test**: `python data_read_test.py` - Verifies telemetry data collection
- **Joystick Test**: `python joystick_test.py` - Tests virtual gamepad functionality
- **Model Test**: `python model.py` - Tests the LSTM model (experimental)

## ⚙️ Configuration

### PPO Hyperparameters (in `train.py`)
- **Learning Rates**: Actor: 0.0003, Critic: 0.001
- **Update Frequency**: Every 4 episodes
- **K Epochs**: 80 policy updates per batch
- **Epsilon Clip**: 0.2 for PPO clipping
- **Discount Factor**: 0.99 for future reward consideration

### Reward System (in `utils.py`)
- **Progress Reward**: Based on track position advancement
- **Speed Multiplier**: 0.5x speed contribution to reward
- **Lap Completion**: +100 reward for finishing a lap
- **Failure Penalties**: -10 for going off-track or getting stuck

## 📊 Training Output

### Logs
- Training progress is logged to `PPO_logs/AssettoCorsa/` directory
- CSV files track episode number, timestep, and average reward
- Console output shows real-time training statistics

### Model Checkpoints
- Trained models are saved to `PPO_preTrained/AssettoCorsa/` directory
- Models can be loaded to continue training or for inference

## 🔧 Technical Details

### State Vector Components
The AI receives a 24-dimensional state vector containing:
- **Wheel Slip** (4 values): Tire grip information
- **Velocity** (3 values): 3D velocity vector
- **Position** (3 values): 3D car coordinates
- **Progress** (1 value): Normalized track position (0-1)
- **Wheels Out** (1 value): Number of wheels off-track

### Action Space
- **Gas**: 0.0 (no throttle) to 1.0 (full throttle)
- **Brake**: 0.0 (no braking) to 1.0 (full braking)
- **Steering**: -1.0 (full left) to 1.0 (full right)

### Neural Network Architecture
- **Actor Network**: 64 → 64 → action_dim with Tanh activation
- **Critic Network**: 64 → 64 → 1 with Tanh activation
- **Input Layer**: 24 neurons (state vector size)
- **Hidden Layers**: 64 neurons each

## 🎯 Current Status & Limitations

### What Works
- ✅ Real-time telemetry data collection from Assetto Corsa
- ✅ Virtual gamepad control integration
- ✅ PPO reinforcement learning framework
- ✅ Basic reward system for track progress
- ✅ Training loop with logging and checkpointing

### Known Limitations
- ⚠️ Assetto Corsa has no official API hooks (workaround using shared memory)
- ⚠️ Requires specific DLL files in the project directory
- ⚠️ Training can be unstable due to game physics complexity
- ⚠️ Reward function may need tuning for optimal performance

### Experimental Features
- LSTM-based model in `model.py` (not fully implemented)
- Alternative joystick control methods in `joystick_test.py`

## 🤝 Contributing

This project is actively being developed. Areas for improvement include:
- More sophisticated reward functions
- Better state representation
- Hyperparameter optimization
- Multi-track training support
- Performance benchmarking

## 📝 License

This project is open source. Feel free to modify and improve upon it for your own Assetto Corsa AI experiments!

## 🙏 Acknowledgments

- **Rombik** for the original Assetto Corsa shared memory implementation
- **ViGEm** for virtual gamepad emulation
- **PyTorch** for the deep learning framework
- **Assetto Corsa** developers for creating such a detailed racing simulation

---

*Note: This bot learns through trial and error. Expect crashes, spins, and off-track excursions during early training phases. The AI will gradually improve its driving skills over time!*

