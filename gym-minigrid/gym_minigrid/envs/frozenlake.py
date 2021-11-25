from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class FrozenLakeEnv(MiniGridEnv):
    """
    Environment with frozen lake to cross/circumnavigate with a hole randomly in the center
    This environment is similar to LavaCrossing but simpler in structure.
    """

    def __init__(self, size, obstacle_type=Lava, seed=None, p=0.2, multiple_holes=True):
        self.obstacle_type = obstacle_type
        self.p = p
        self.multiple_holes = multiple_holes
        super().__init__(
            grid_size=size,
            max_steps=2*size*size,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=seed
        )

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # # Place barriers.
        _idx = np.arange(width * height).reshape(width, height).T

        # TODO: change to not hard coded
        self.potential_lava_locs = np.array([16, 23, 30,
                                             17, 24, 31,
                                             18, 25, 32])

        if self.multiple_holes:
        # dynamic number of holes
            probs = np.random.rand(len(self.potential_lava_locs))
            self.lava_idx = np.arange(len(self.potential_lava_locs))[probs < self.p]
            for i in range(len(self.lava_idx)):
                loc_idx = self.potential_lava_locs[self.lava_idx[i]]
                loc = np.squeeze(np.asarray(np.where(loc_idx == _idx)))
                self.put_obj(Lava(), loc[1], loc[0])
        else:
            self.lava_idx = []
            for i in range(1): # In the future, could make number of holes > 1 or dynamic
                self.lava_idx.append(np.random.choice(len(self.potential_lava_locs)))
                loc_idx = self.potential_lava_locs[self.lava_idx[-1]]
                loc = np.squeeze(np.asarray(np.where(loc_idx == _idx)))
                self.put_obj(Lava(), loc[1], loc[0])
            self.lava_idx = np.asarray(self.lava_idx)

        # Place the agent in middle left
        self.agent_pos = (1, (int) (height / 2))
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.goal_pos = np.array((width - 2, (int) (height / 2)))
        self.put_obj(Goal(), *self.goal_pos)


        self.lava_locations = []
        self._lava_idx = []
        for _i in range(len(self.grid.grid)):
            if type(self.grid.grid[_i]) == Lava:
                _idx = np.asarray(np.where(np.arange(width * height).reshape(width, height) == _i)).squeeze()
                self.lava_locations.append((_idx[1], _idx[0]))
                self._lava_idx.append(_i)
        p = 0

        self.mission = (
            "avoid the hole and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

    def step(self, action):

        self.step_count += 1

        reward = -2 # TODO Fix rewards
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward += 20
                # reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True
                reward += -100
        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}

class FrozenLakeS7Env(FrozenLakeEnv):
    def __init__(self):
        super().__init__(size=7, )


register(
    id='MiniGrid-FrozenLakeS7-v0',
    entry_point='gym_minigrid.envs:FrozenLakeS7Env'
)
