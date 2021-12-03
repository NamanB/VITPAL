# VITPAL - Variable Information Transfer from Privileged Agent to Learners

## Installation
*Note use python 3.7 to avoid multiprocessig issue that happens in 3.8

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
./gym-minigrid/manual_control.py
```

You can set the `--agent_normal`, `--agent_privileged`, and `--lava_render_dist <dist (-1 for all)>`  flags. The render dist flag can be used to make the privileged agent not omnicient but still have privileged information.

### Environments and Wrappers
The `VitpalTrainWrapper` `VitpalExpertImgObsWrapper` and `VitpalRGBImgObsWrapper` are the wrappers used for each type of agent or training. The train wrapper will return observations containing both the expert and rgb image obs in the `normal` and `privileged` sub fields of obs. The expert image wrapper is created with `lava_render_dist` with (-1) default, allowing privilaged agents to see lava in manhattan distance within their distance range. The normal rgb image observation wrapper does not show lava by default and returns the grid in pixel space.  

### Train a sample baseline agent
To train a sample agent, navigate to the `rl-starter-files` then
```
python -m scripts.train --algo a2c --env MiniGrid-FrozenLakeS7-v0 --model FrozenLake --save-interval 10 --frames 80000
```

Note that `--model <model name>` is the name of the model. You can change the `--algo <algo name>` flag for different algorithms, and the number of processes can be changed with `--procs <num_procs defaults 16>`

### Train a dagger agent
```
python3 -m scripts.train --algo dagger --env MiniGrid-FrozenLakeS7-v0 --model FrozenLake --expert-model expert --save-interval 10 --frames 800000 --procs 1
```

Note that `--model <model name>` is the name is the model to be trained and the expert model is the model that is trained using reinforcement learning. 

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