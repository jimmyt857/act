import pickle

import IPython
import numpy as np
import requests
import simplejpeg
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import functional as F

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer

e = IPython.embed


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        print(f"KL Weight {self.kl_weight}")

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos, image, env_state, actions, is_pad
            )
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(
                qpos, image, env_state
            )  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # TODO
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict["mse"] = mse
            loss_dict["loss"] = loss_dict["mse"]
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class PiPolicy:
    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._uri = f"http://{host}:{port}"

    def __call__(self, qpos, image, actions=None, is_pad=None):
        request = self._aloha_to_pi_request(qpos, image)
        response = self._post_request(request)
        return self._pi_response_to_aloha(response)

    def _aloha_to_pi_request(self, qpos: np.ndarray, image: np.ndarray) -> dict:
        # qpos is (14,) of type float:
        # 0-5: left arm joint angles
        # 6: left arm gripper
        # 7-12: right arm joint angles
        # 13: right arm gripper
        #
        # image is [cam_idx, channel, height, width] with values in [0, 1] of type float

        # Convert to [height, width, channel]
        converted_image = np.transpose(image[0] * 255, (1, 2, 0)).astype(np.uint8)

        return {
            "image": {
                "base_0_rgb": simplejpeg.encode_jpeg(converted_image),
                "base_0_rgb_mask": np.array(True),
                # TODO: We don't have these images yet.
                "left_wrist_0_rgb": simplejpeg.encode_jpeg(converted_image),
                "left_wrist_0_rgb_mask": np.array(True),
                "right_wrist_0_rgb": simplejpeg.encode_jpeg(converted_image),
                "right_wrist_0_rgb_mask": np.array(True),
            },
            "state": qpos,
            "raw_text": "pick up cube",
        }

    def _post_request(self, request: dict) -> np.ndarray:
        response = requests.post(self._uri, data=pickle.dumps(request))
        if response.status_code != 200:
            raise Exception(response.text)
        return pickle.loads(response.content)

    def _pi_response_to_aloha(self, response: np.ndarray) -> np.ndarray:
        # response is (50,14) of type float32
        # 0-5: left arm joint angles
        # 6-11: right arm joint angles
        # 12: left arm gripper
        # 13: right arm gripper
        aloha_action = response[:, [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]]
        return aloha_action[0]


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
