# This code is adapted from VERL
# https://github.com/volcengine/verl/blob/main/verl/utils/tracking.py
# The original copyright is reproduced below:
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Union
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import pprint


# TODO(tgriggs): Test all backends.
class Tracking:
    supported_backends = ["wandb", "mlflow", "swanlab", "tensorboard", "console"]

    def __init__(self, project_name, experiment_name, backends: Union[str, List[str]] = "console", config=None):
        if isinstance(backends, str):
            backends = [backends]
        for backend in backends:
            assert backend in self.supported_backends, f"{backend} is not supported"

        self.logger = {}

        if "wandb" in backends:
            import wandb
            from omegaconf import OmegaConf

            wandb.init(project=project_name, name=experiment_name, config=OmegaConf.to_container(config, resolve=True))
            self.logger["wandb"] = wandb

        if "mlflow" in backends:
            self.logger["mlflow"] = _MlflowLoggingAdapter(project_name, experiment_name, config)

        if "swanlab" in backends:
            import os

            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config=config,
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab

        if "tensorboard" in backends:
            self.logger["tensorboard"] = _TensorboardAdapter()

        if "console" in backends:
            self.console_logger = ConsoleLogger()
            self.logger["console"] = self.console_logger

    def log(self, data, step, commit=False):
        for logger_name, logger_instance in self.logger.items():
            if logger_name == "wandb":
                logger_instance.log(data=data, step=step, commit=commit)
            else:
                logger_instance.log(data=data, step=step)

    def __del__(self):
        # NOTE (sumanthrh): We use a try-except block here while finishing tracking.
        # This is because wandb often errors out with a BrokenPipeError when closing.
        # https://github.com/wandb/wandb/issues/6449
        # TODO (sumanthrh): Check if this is really needed. Trackers like wandb will automatically finish at program exit.
        try:
            if "wandb" in self.logger:
                self.logger["wandb"].finish(exit_code=0)
            if "swanlab" in self.logger:
                self.logger["swanlab"].finish()
            if "tensorboard" in self.logger:
                self.logger["tensorboard"].finish()
            if "mlflow" in self.logger:
                self.logger["mlflow"].finish()
        except Exception as e:
            logger.warning(f"Attempted to finish tracking but got error {e}")


class ConsoleLogger:
    def __init__(self):
        pass

    def log(self, data: Dict[str, Any], step: int):
        # pprint the data and log with logger
        data_as_str = pprint.pformat(ConsoleLogger.stringify_floats(data))
        logger.info(f"Step {step}: \n{data_as_str}")

    def finish(self):
        pass

    @staticmethod
    def stringify_floats(obj: Any) -> Any:
        if isinstance(obj, float):
            return f"{obj:.4f}"
        elif isinstance(obj, dict):
            return {k: ConsoleLogger.stringify_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ConsoleLogger.stringify_floats(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(ConsoleLogger.stringify_floats(v) for v in obj)
        return obj


class _TensorboardAdapter:
    def __init__(self):
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", "tensorboard_log")
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        self.writer.close()


class _MlflowLoggingAdapter:
    def __init__(self, project_name, experiment_name, config):
        import os

        import mlflow

        if mlflow.active_run() is None:
            self.we_created_mlflow = True
            if mlflow_tracking_uri := os.environ.get("MLFLOW_TRACKING_URI", None):
                mlflow.set_tracking_uri(mlflow_tracking_uri)

            # Project_name is actually experiment_name in MLFlow
            # If experiment does not exist, will create a new experiment
            experiment = mlflow.set_experiment(project_name)
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)

        else:
            self.we_created_mlflow = False

        mlflow.log_params(_compute_mlflow_params_from_objects(config))
        self.mlflow = mlflow

    def log(self, data, step):
        results = {k.replace("@", "_at_"): v for k, v in data.items()}
        self.mlflow.log_metrics(metrics=results, step=step)

    def finish(self):
        if self.we_created_mlflow:
            self.mlflow.end_run()


def _compute_mlflow_params_from_objects(params) -> Dict[str, Any]:
    if params is None:
        return {}

    if isinstance(params, DictConfig):
        params = OmegaConf.to_container(params, resolve=True)

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: Dict[str, Any], *, sep: str) -> Dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans


@dataclasses.dataclass
class ValidationGenerationsLogger:
    def log(self, loggers, samples, step):
        if "wandb" in loggers:
            self.log_generations_to_wandb(samples, step)
        if "swanlab" in loggers:
            self.log_generations_to_swanlab(samples, step)
        if "mlflow" in loggers:
            self.log_generations_to_mlflow(samples, step)

    def log_generations_to_wandb(self, samples, step):
        """Log samples to wandb as a table"""
        import wandb

        # Create column names for all samples
        columns = ["step"] + sum(
            [[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], []
        )

        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(step)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table

    def log_generations_to_swanlab(self, samples, step):
        """Log samples to swanlab as text"""
        import swanlab

        swanlab_text_list = []
        for i, sample in enumerate(samples):
            row_text = f"""
            input: {sample[0]}
            
            ---
            
            output: {sample[1]}
            
            ---
            
            score: {sample[2]}
            """
            swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i + 1}"))

        # Log to swanlab
        swanlab.log({"val/generations": swanlab_text_list}, step=step)

    def log_generations_to_mlflow(self, samples, step):
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import json
        import tempfile

        import mlflow

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for sample in samples:
                    data = {"input": sample[0], "output": sample[1], "score": sample[2]}
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            logger.warning(f"save validation generation file to mlflow failed with error {e}")
