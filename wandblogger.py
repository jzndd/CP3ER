import torch
import logging
import wandb


logger = logging.getLogger(__name__)


class WandbLogger(object):

    def __init__(self, task_name="dog_walk", policy_type="cp3er", config=None, seed=0):
        """
        General logger.

        Args:
            log_dir (str): log directory
        """
        self.info = logger.info
        self.debug = logger.debug
        self.warning = logger.warning

        wandb_proj = "CP3ER-dmc-{}".format(task_name)
        wandb_name = "{}-seed:{}".format(policy_type, seed)
        wandb_group = "{}".format(policy_type)
        wandb.init(config=config, project=wandb_proj, name=wandb_name, group=wandb_group, job_type="training")

    def scalar_summary(self, tag, value, step):
        """
        Log scalar value to the disk.
        Args:
            tag (str): name of the value
            value (float): value
            step (int): update step
        """
        # self.writer.add_scalar(tag, value, step)
        data = {}
        data[tag]=value
        # wandb.log(data, step)
        try:
            # print(0)
            wandb.log(data, step)
        except Exception as e:
            print(e)

    def log_metrics(self, metrics, step, ty):
        """
        Logs a dictionary of metrics to Weights & Biases.
        
        Args:
            metrics (dict): A dictionary of metrics to log, where keys are the metric names and values are the metric values.
            step (int): The current step or epoch in training or evaluation.
            ty (str): The type of the metrics, typically 'train' or 'eval'.
        """
        # Preprocess the metrics dictionary to include the type (train/eval) in the keys
        wandb_metrics = {f'{ty}/{key}': value.item() if isinstance(value, torch.Tensor) else value for key, value in metrics.items()}

        # Log the metrics to wandb
        wandb.log(wandb_metrics, step=step)


    def close(self):
        # self.writer.close()
        pass
