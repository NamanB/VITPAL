# VITPAL - Variable Information Transfer from Privileged Agent to Learners

## Installation

1. Clone this repository.
2. Install custom `gym-minigrid` by navigating to its folder then
```
pip install -e .
```

3. Install custom `torch-ac` by navigating to its folder then
```
pip install -e .
```

4. Install `rl-starter-files` dependencies by navigating to its folder then
```
pip install -r requirements.txt
```

## Example Usage
### Environment Test
To test frozen lake environment manually, run 
```
./gym-minigrid/manual_control.py --env MiniGrid-FrozenLakeS7-v0
```

### Train a sample baseline agent
To train a sample agent, navigate to the `rl-starter-files` then
```
python -m scripts.train --algo a2c --env MiniGrid-FrozenLakeS7-v0 --model FrozenLake --save-interval 10 --frames 80000
```

Note that `--model <model name>` is the name of the model. You can change the `--algo <algo name>` flag for different algorithms, and the number of processes can be changed with `--procs <num_procs defaults 16>`

### Visualize an agent
To visualize an agent, from the `rl-starter-files` folder, run
```
python -m scripts.visualize --env MiniGrid-FrozenLakeS7-v0 --model FrozenLake
```

### Evaluate an agent
To evaluate an agent, from the `rl-starter-files` folder, run
```
python -m scripts.evaluate --env MiniGrid-FrozenLakeS7-v0 --model FrozenLake
```