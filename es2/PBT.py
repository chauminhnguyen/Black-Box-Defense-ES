# Population based training of neural networks
import torch
import ray
from ray import air, tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig


# from ray.tune.examples.pbt_dcgan_mnist.common import Net
from tasks.classification import Classification
import torch.nn as nn

import logging

logger = logging.getLogger("ray.serve")
_ray_log_level = logging.ERROR
ray.init(log_to_driver=False, local_mode=False, logging_level=_ray_log_level)


def train_classification(args):
    # args.epochs = 1
    task = Classification(args)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            task.model = nn.DataParallel(task.model)
    task.model.to(device)

    task.train()


class PBT:
    def __init__(self, args) -> None:
        perturbation_interval = 5
        scheduler = PopulationBasedTraining(
            perturbation_interval=perturbation_interval,
            hyperparam_mutations={
                # Distribution for resampling
                "batch": tune.uniform(32, 512),
                "lr": tune.uniform(1e-2, 1e-5),
                "q": tune.uniform(50, 300),
            },
        )

        scaling_config = ScalingConfig(
            # Number of distributed workers.
            num_workers=2,
            # Turn on/off GPU.
            use_gpu=True,
            # Specify resources used for trainer.
            trainer_resources={"CPU": 1},
            # Try to schedule workers on different nodes.
            placement_strategy="SPREAD",
        )

        smoke_test = False  # For testing purposes: set this to False to run the full experiment
        self.tuner = tune.Tuner(
            train_classification,
            run_config=air.RunConfig(
                name="pbt_dcgan_mnist_tutorial",
                stop={"training_iteration": 50 if smoke_test else 6000},
                verbose=0
            ),
            tune_config=tune.TuneConfig(
                metric="is_score",
                mode="max",
                num_samples=2 if smoke_test else 8,
                scheduler=scheduler,
            ),
            param_space={
                # Define how initial values of the learning rates should be chosen.
                "lr": tune.choice([0.0001, 0.0002, 0.0005]),
                "checkpoint_interval": perturbation_interval,
            },
            scaling_config=scaling_config,
        )

    def run(self):
        self.results_grid = self.tuner.fit()

    def visualize(self):
        import matplotlib.pyplot as plt

        result_dfs = [result.metrics_dataframe for result in self.results_grid]
        best_result = self.results_grid.get_best_result(metric="is_score", mode="max")

        plt.figure(figsize=(7, 4))
        for i, df in enumerate(result_dfs):
            plt.plot(df["is_score"], label=i)
        plt.legend()
        plt.title("Inception Score During Training")
        plt.xlabel("Training Iterations")
        plt.ylabel("Inception Score")
        plt.savefig("inception_score.png")
