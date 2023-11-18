from typing import Dict, List, Tuple
from gymnasium.spaces.space import Space
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.typing import ModelConfigDict, TensorType
import torch
import torch.nn as nn


# from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.models.torch.misc import SlimFC, SlimConv2d
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override

from pfrl.nn import Branched


class CustomTorchModel(TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        ...

    def forward(self, input_dict, state, seq_lens):
        ...

    def value_function(self):
        ...


class IPPO(TorchRNN, nn.Module):
    def __init__(
        self,
        obs_space: Space,
        action_space: Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        lstm_state_size=256,
        **kwargs,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        tl_obs = obs_space.original_space["traffic_light_colors"]
        tl_colors_flat = tl_obs.shape[0] * tl_obs.shape[1] * tl_obs.shape[2]

        self.num_outputs = num_outputs

        self.lstm_state_size = lstm_state_size

        def conv2d_size_out(size, kernel_size=2, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        h = conv2d_size_out(obs_space.original_space["radar_states"].shape[1])
        w = conv2d_size_out(obs_space.original_space["radar_states"].shape[2])

        self._radar_encoder = nn.Sequential(
            SlimConv2d(
                obs_space.original_space["radar_states"].shape[0],
                32,
                kernel=(2, 2),
                stride=(1, 1),
                padding=(0, 0),
            ),
            nn.Flatten(),
            SlimFC(h * w * 32, 64, activation_fn="relu"),
            SlimFC(64, 32, activation_fn="relu"),
        )
        self._final_encoder = nn.Sequential(
            SlimFC(tl_colors_flat + 32, tl_colors_flat + 32, activation_fn="relu"),
            SlimFC(tl_colors_flat + 32, 32, activation_fn="relu"),
        )

        self.lstm = nn.LSTM(32, self.lstm_state_size, batch_first=True)

        self._output = Branched(
            nn.Sequential(
                SlimFC(self.lstm_state_size, 32, activation_fn="relu"),
                SlimFC(32, self.num_outputs, activation_fn="relu"),
            ),
            nn.Sequential(
                SlimFC(self.lstm_state_size, 32, activation_fn="relu"),
                SlimFC(32, 1, activation_fn="relu"),
            ),
        )

        self._features = None

        self._vf = None

    @override(ModelV2)
    def get_initial_state(self):
        return [
            self._final_encoder[-1]
            ._model[0]
            .weight.new(1, self.lstm_state_size)
            .zero_()
            .squeeze(0),
            self._final_encoder[-1]
            ._model[0]
            .weight.new(1, self.lstm_state_size)
            .zero_()
            .squeeze(0),
        ]

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self._vf, [-1])

    @override(ModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass.
        """
        self.time_major = self.model_config.get("_time_major", False)

        output, new_state = self.forward_rnn(
            {
                "radar_states": input_dict["obs"]["radar_states"],
                "traffic_light_colors": input_dict["obs"]["traffic_light_colors"],
            },
            state,
            seq_lens,
        )
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.

        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).

        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        x = self._radar_encoder(inputs["radar_states"])
        x = torch.cat([x, inputs["traffic_light_colors"].flatten(1)], dim=1)
        x = self._final_encoder(x)
        x = add_time_dimension(
            x,
            seq_lens=seq_lens,
            framework="torch",
            time_major=self.time_major,
        )
        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )
        action_out, self._vf = self._output(self._features)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]


ModelCatalog.register_custom_model("IPPO", IPPO)
