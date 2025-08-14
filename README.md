# Assetto Corsa AI Bot

An AI-powered racing bot for Assetto Corsa that uses Proximal Policy Optimization (PPO) reinforcement learning to learn how to drive cars autonomously around race tracks.

## 📁 Project Structure

```
AssettoCorsaBot/
├── 📄 Core AI Files
│   ├── PPO.py                    # PPO algorithm implementation with Actor-Critic networks
│   ├── train.py                  # Main training loop and orchestration
│   ├── model.py                  # Experimental LSTM model (incomplete)
│   └── utils.py                  # Game control, reward system, and episode management
│
├── 🎮 Game Integration
│   ├── sim_info.py               # Assetto Corsa shared memory interface
│   ├── data_read_test.py         # Test telemetry data reading
│   ├── data_write_test.py        # Test data writing capabilities
│   ├── joystick_test.py          # Test virtual joystick functionality
│   └── exetest.py                # Executable testing
│
├── 🧠 Training & Models
│   ├── PPO_logs/                 # Training progress logs
│   │   └── AssettoCorsa/         # CSV files tracking episode rewards
│   └── PPO_preTrained/          # Saved model checkpoints
│       └── AssettoCorsa/         # Trained PPO models (.pth files)
│
├── 🔧 System Dependencies
│   ├── DLLs/                     # Python DLL files for Windows compatibility
│   ├── stdlib/                   # Standard library dependencies
│   ├── ViGEmClient.dll           # Virtual gamepad emulation driver
│   └── func_test.py              # Functionality testing
│
└── 📚 Documentation
    └── README.md                 # This file
```

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
- **Key Classes**: `SPageFilePhysics`, `SPageFileGraphic`, `SPageFileStatic`, `SimInfo`

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
- **Key Classes**: `RolloutBuffer`, `ActorCritic`, `PPO`

### 4. **Training Loop** (`train.py`)
- Orchestrates the complete training process
- Manages episodes, timesteps, and model updates
- Handles logging, checkpointing, and hyperparameter tuning
- Implements action standard deviation decay for exploration vs exploitation balance

### 5. **Experimental Models** (`model.py`)
- LSTM-based neural network architecture (currently incomplete)
- Alternative approach to the PPO implementation
- Designed for sequence-based learning from telemetry data

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
4. Verify all DLL files are present in the `DLLs/` directory

### Running the Bot

#### Training Mode
```bash
python train.py
```
This will start the PPO training process, where the AI learns to drive through trial and error.

#### Testing Individual Components
- **Data Reading Test**: `python data_read_test.py` - Verifies telemetry data collection
- **Data Writing Test**: `python data_write_test.py` - Tests data output capabilities
- **Joystick Test**: `python joystick_test.py` - Tests virtual gamepad functionality
- **Model Test**: `python model.py` - Tests the LSTM model (experimental)
- **Function Test**: `python func_test.py` - Tests core functionality
- **Executable Test**: `python exetest.py` - Tests executable compatibility

## ⚙️ Configuration

### PPO Hyperparameters (in `train.py`)
- **Learning Rates**: Actor: 0.0003, Critic: 0.001
- **Update Frequency**: Every 4 episodes
- **K Epochs**: 80 policy updates per batch
- **Epsilon Clip**: 0.2 for PPO clipping
- **Discount Factor**: 0.99 for future reward consideration
- **Action Standard Deviation**: Starts at 0.6, decays to 0.1

### Reward System (in `utils.py`)
- **Progress Reward**: Based on track position advancement
- **Speed Multiplier**: 0.5x speed contribution to reward
- **Lap Completion**: +100 reward for finishing a lap
- **Failure Penalties**: -10 for going off-track or getting stuck
- **Stuck Detection**: Penalizes if car doesn't move for extended periods

## 📊 Training Output

### Logs
- Training progress is logged to `PPO_logs/AssettoCorsa/` directory
- CSV files track episode number, timestep, and average reward
- Console output shows real-time training statistics
- Logs are automatically numbered to prevent overwriting

### Model Checkpoints
- Trained models are saved to `PPO_preTrained/AssettoCorsa/` directory
- Models can be loaded to continue training or for inference
- Checkpoint format: `PPO_AssettoCorsa_{seed}_{run}.pth`

## 🔧 Technical Details

### State Vector Components
The AI receives a 24-dimensional state vector containing:
- **Wheel Slip** (4 values): Tire grip information for each wheel
- **Velocity** (3 values): 3D velocity vector (x, y, z)
- **Position** (3 values): 3D car coordinates in world space
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
- **Activation**: Tanh for continuous actions, Softmax for discrete

### Shared Memory Structure
- **Physics Data**: Real-time car physics (speed, wheel slip, tire wear, etc.)
- **Graphics Data**: Visual information (position, lap progress, flags)
- **Static Data**: Car and track information (car model, track name, etc.)

## 🎯 Current Status & Limitations

### What Works
- ✅ Real-time telemetry data collection from Assetto Corsa
- ✅ Virtual gamepad control integration through ViGEm
- ✅ PPO reinforcement learning framework with experience replay
- ✅ Basic reward system for track progress and speed
- ✅ Training loop with logging and checkpointing
- ✅ Action standard deviation decay for exploration control
- ✅ Episode management and restart functionality

### Known Limitations
- ⚠️ Assetto Corsa has no official API hooks (workaround using shared memory)
- ⚠️ Requires specific DLL files in the project directory for Windows compatibility
- ⚠️ Training can be unstable due to game physics complexity
- ⚠️ Reward function may need tuning for optimal performance
- ⚠️ LSTM model in `model.py` is incomplete and experimental

### Experimental Features
- LSTM-based model in `model.py` (not fully implemented)
- Alternative joystick control methods in `joystick_test.py`
- Multiple testing utilities for debugging and validation

## 🔍 File Descriptions

### Core AI Files
- **`PPO.py`**: Complete PPO implementation with neural networks and training logic
- **`train.py`**: Main training script that orchestrates the learning process
- **`utils.py`**: Game control, reward calculation, and environment management
- **`model.py`**: Experimental LSTM-based alternative to PPO

### Game Integration
- **`sim_info.py`**: Shared memory interface for reading Assetto Corsa data
- **`data_read_test.py`**: Simple test for reading telemetry data
- **`data_write_test.py`**: Test for data output capabilities
- **`joystick_test.py`**: Virtual joystick functionality testing
- **`exetest.py`**: Executable compatibility testing

### Testing & Utilities
- **`func_test.py`**: Core functionality testing
- **`DLLs/`**: Windows Python compatibility files
- **`stdlib/`**: Standard library dependencies
- **`ViGEmClient.dll`**: Virtual gamepad emulation driver

## 🤝 Contributing

This project is actively being developed. Areas for improvement include:
- More sophisticated reward functions based on racing line optimization
- Better state representation including track curvature and braking zones
- Hyperparameter optimization and automated tuning
- Multi-track training support and generalization
- Performance benchmarking and comparison tools
- Complete LSTM model implementation
- Better error handling and recovery mechanisms

## 📝 License

This project is open source. Feel free to modify and improve upon it for your own Assetto Corsa AI experiments!

## 🙏 Acknowledgments

- **Rombik** for the original Assetto Corsa shared memory implementation
- **ViGEm** for virtual gamepad emulation
- **PyTorch** for the deep learning framework
- **Assetto Corsa** developers for creating such a detailed racing simulation

## 🚨 Troubleshooting

### Common Issues
1. **DLL Errors**: Ensure all files in `DLLs/` directory are present
2. **ViGEm Issues**: Verify ViGEm Bus Driver is properly installed
3. **Game Not Responding**: Check that Assetto Corsa is running and a track is loaded
4. **Training Instability**: Adjust hyperparameters in `train.py`

### Debug Mode
Run individual test files to isolate issues:
```bash
python data_read_test.py    # Test telemetry reading
python joystick_test.py     # Test gamepad control
python func_test.py         # Test core functions
```

---

*Note: This bot learns through trial and error. Expect crashes, spins, and off-track excursions during early training phases. The AI will gradually improve its driving skills over time!*

