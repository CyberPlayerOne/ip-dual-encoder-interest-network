# Noam LR Scheduler with Warmup for Adam Optimizer

import torch


def calc_lr(step, dim_embed, warmup_steps, scale_factor=1.0):
    return scale_factor * (
            dim_embed ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    )


class NoamLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, dim_embed: int, warmup_steps: int, scale_factor: float = 1.0, last_epoch: int = -1, verbose: bool = False) -> None:
        """

        :param optimizer:
        :param dim_embed:
        :param warmup_steps:
        :param scale_factor: float, optional (default = 1.0) The overall scale factor for the learning rate decay.
        :param last_epoch:
        :param verbose:
        """
        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)
        self.scale_factor = scale_factor
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        lr = calc_lr(self._step_count, self.dim_embed, self.warmup_steps, self.scale_factor)
        return [lr] * self.num_param_groups
