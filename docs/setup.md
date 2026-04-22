# Setup Guide

## Prerequisites

### MATLAB
- **Version**: R2024a or later
- **Toolboxes**: Reinforcement Learning Toolbox, Statistics and Machine Learning Toolbox

### Python
- **Version**: 3.12
- Ensure MATLAB can call Python: run `pyenv` in MATLAB to verify

### JSBSim

1. Clone JSBSim:
   ```bash
   cd ~/Documents/jsbsim
   git clone https://github.com/JSBSim-Team/jsbsim.git jsbsim-master
   ```

2. Install the Python package:
   ```bash
   pip3 install jsbsim
   ```

3. Verify installation:
   ```python
   import jsbsim
   fdm = jsbsim.FGFDMExec('/path/to/jsbsim-master')
   print(fdm.get_property_value('simulation/sim-time-sec'))
   ```

### FlightGear (Optional — for 3D visualisation only)

1. Download from [flightgear.org](https://www.flightgear.org/download/)
2. Launch with the Cessna 172 model before running `visualisation/flightgear_demo.m`

## Configuration

All evaluation scripts reference the JSBSim root path. Update this line in each script if your installation differs:

```matlab
root = '/path/to/your/jsbsim-master';
```

The trained agent is loaded from:

```matlab
loaded = load('models/Final_v2_Finetuned.mat');
```

## Running the Evaluation

Scripts are designed to be run independently from the repository root in MATLAB:

```matlab
cd cessna172-rl-control
run('evaluation/elevator_icing.m')
```

All figures are exported at 300 DPI to the current directory. Monte Carlo data is saved as `.mat` files.
