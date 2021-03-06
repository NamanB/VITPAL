import numpy
import torch
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv

class DaggerAlgo(BaseAlgo):
    """The Dagger algorithm."""

    def __init__(self, envs, acmodel, expert_model, beta_cooling=0.999999, device=None, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None, reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        import copy 
        expert_obs = list()
        for j, ob in enumerate(self.obs):
            expert_ob= copy.deepcopy(ob)
            expert_ob["image"]=expert_ob["privileged"]

            expert_obs.append(expert_ob)

        self.expert_obs = expert_obs

        self.expert_model = expert_model
        self.beta = 1
        self.beta_cooling = beta_cooling

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)
    
    def update_beta(self):
        self.beta *= self.beta_cooling

    def collect_experiences(self):
        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            with torch.no_grad():

                import random
                e = random.uniform(0, 1)

                # Expert query at random
                if e < self.beta:
                    preprocessed_obs = self.preprocess_obss(self.expert_obs, device=self.device)
                    if self.expert_model.recurrent:
                        dist, _, memory = self.expert_model(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                    else:
                        dist, _ = self.expert_model(preprocessed_obs)
                else:
                    preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
                    if self.expert_model.recurrent:
                        dist, _, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                    else:
                        dist, _ = self.acmodel(preprocessed_obs)
        
            self.update_beta()
            action = dist.sample()

            obs, reward, done, _ = self.env.step(action.cpu().numpy())

            self.obss[i] = self.obs

            import copy 
            expert_obs = list()
            for j, ob in enumerate(obs):
                expert_ob= copy.deepcopy(ob)
                expert_ob["image"]=expert_ob["privileged"]

                expert_obs.append(expert_ob)

            self.expert_obs = expert_obs
            self.obs = obs

            # Update experiences values

            # self.obss[i] = self.obs
            # self.obs = obs
            if self.expert_model.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            #self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs
        
    def update_parameters(self, exps):

        # Perform BC Update

        # Compute starting indexes

        inds = self._get_starting_indexes()

        # Initialize update values

        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0

        # Initialize memory

        if self.acmodel.recurrent:
            memory = exps.memory[inds]

        for i in range(self.recurrence):
            # Create a sub-batch of experience

            sb = exps[inds + i]

            # Compute loss

            if self.acmodel.recurrent:
                dist, _, memory = self.acmodel(sb.obs, memory * sb.mask)
            else:
                dist, _ = self.acmodel(sb.obs)

            entropy = dist.entropy().mean()

            policy_loss = -(dist.log_prob(sb.action)).mean()

            loss = policy_loss - self.entropy_coef * entropy

            # Update batch values

            update_entropy += entropy.item()
            update_value += 0
            update_policy_loss += policy_loss.item()
            update_value_loss += 0
            update_loss += loss

        # Update update values

        update_entropy /= self.recurrence
        update_value /= self.recurrence
        update_policy_loss /= self.recurrence
        update_value_loss /= self.recurrence
        update_loss /= self.recurrence

        # Update actor-critic

        self.optimizer.zero_grad()
        update_loss.backward()
        # update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
        update_grad_norm = 0
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log some values

        logs = {
            "entropy": update_entropy,
            "value": update_value,
            "policy_loss": update_policy_loss,
            "value_loss": update_value_loss,
            "grad_norm": update_grad_norm
        }

        return logs

    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the
        experiences used to compute the loss at first.

        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.

        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = numpy.arange(0, self.num_frames, self.recurrence)
        return starting_indexes
