import numpy as np

import torch
import torch.nn as nn

from pfrl.nn import Branched
import pfrl.initializers
from pfrl.agents import PPO
from pfrl.policies import SoftmaxCategoricalHead


class Agent(object):
    def __init__(self):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        self.device = torch.device(device)

    def act(self, observation):
        raise NotImplementedError

    def observe(self, observation, reward, done, info):
        raise NotImplementedError


class IndependentAgent(Agent):
    def __init__(self, config, ):
        super().__init__()
        self.config = config
        self.agents = dict()

    def act(self, observation):
        acts = dict()
        for agent_id in observation.keys():
            acts[agent_id] = self.agents[agent_id].act(observation[agent_id])
        return acts

    def observe(self, observation, reward, done, info):
        for agent_id in observation.keys():
            self.agents[agent_id].observe(
                observation[agent_id], reward[agent_id], done, info
            )
            if done:
                if info["eps"] % self.config["save_freq"] == 0:
                    self.agents[agent_id].save(
                        self.config["log_dir"] + "agent_" + agent_id
                    )


def lecun_init(layer, gain=1):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        pfrl.initializers.init_lecun_normal(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    else:
        pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
        pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
        nn.init.zeros_(layer.bias_ih_l0)
        nn.init.zeros_(layer.bias_hh_l0)
    return layer


class IPPO(IndependentAgent):
    def __init__(self, obs_act, ):
        super().__init__(obs_act, )
        for key in obs_act:
            obs_space = obs_act[key][0]
            act_space = obs_act[key][1]
            self.agents[key] = PFRLPPOAgent(obs_space, act_space)
            # if self.config["load"]:
            #     print("LOADING SAVED MODEL FOR EVALUATION")
            #     self.agents[key].load(self.config["log_dir"] + "agent_" + key + ".pt")
            #     self.agents[key].agent.training = False


class PFRLPPOAgent(Agent):
    def __init__(self, obs_space, act_space):
        super().__init__()

        def conv2d_size_out(size, kernel_size=2, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        h = conv2d_size_out(obs_space.shape[1])
        w = conv2d_size_out(obs_space.shape[2])

        self.model = nn.Sequential(
            lecun_init(nn.Conv2d(obs_space.shape[0], 64, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            lecun_init(nn.Linear(h * w * 64, 64)),
            nn.ReLU(),
            lecun_init(nn.Linear(64, 64)),
            nn.ReLU(),
            Branched(
                nn.Sequential(
                    lecun_init(nn.Linear(64, act_space[0].n), 1e-2), SoftmaxCategoricalHead()
                ),
                lecun_init(nn.Linear(64, 1)),
            ),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2.5e-4, eps=1e-5)
        self.agent = PPO(
            self.model,
            self.optimizer,
            gpu=self.device.index,
            phi=lambda x: np.asarray(x, dtype=np.float32),
            clip_eps=0.1,
            clip_eps_vf=None,
            update_interval=1024,
            minibatch_size=256,
            epochs=4,
            standardize_advantages=True,
            entropy_coef=0.001,
            max_grad_norm=0.5,
        )

    def act(self, observation):
        return [self.agent.act(observation)]

    def observe(self, observation, reward, done, info):
        self.agent.observe(observation[0], reward, done, False)

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path + ".pt",
        )

    def load(self, path):
        self.model.load_state_dict(torch.load(path)["model_state_dict"])
        self.optimizer.load_state_dict(torch.load(path)["optimizer_state_dict"])
