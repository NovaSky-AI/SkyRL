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
import pprint
import traceback
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from skyrl.train.config import SkyRLTrainConfig, get_config_as_dict


# TODO(tgriggs): Test all backends.
class Tracking:
    supported_backends = ["wandb", "mlflow", "swanlab", "tensorboard", "console"]

    def __init__(
        self,
        project_name,
        experiment_name,
        backend: str = "console",
        config: Optional[Union[SkyRLTrainConfig, DictConfig]] = None,
        tags: Optional[List[str]] = None,
    ):
        assert backend in self.supported_backends, f"{backend} is not supported"
        self.backend = backend

        if backend == "wandb":
            import wandb

            wandb.init(project=project_name, name=experiment_name, config=get_config_as_dict(config), tags=tags)
            self.logger: Any = wandb
        elif backend == "mlflow":
            self.logger = _MlflowLoggingAdapter(project_name, experiment_name, config)
        elif backend == "swanlab":
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
            self.logger = swanlab
        elif backend == "tensorboard":
            self.logger = _TensorboardAdapter()
        else:  # "console"
            self.logger = ConsoleLogger()

        self._exception_logged = False

    def log(self, data, step, commit=False):
        if self.backend == "wandb":
            self.logger.log(data=data, step=step, commit=commit)
        else:
            self.logger.log(data=data, step=step)

    def finish(self):
        if self.backend == "console":
            return
        # NOTE (sumanthrh): We use a try-except block here while finishing tracking.
        # This is because wandb often errors out with a BrokenPipeError when closing.
        # https://github.com/wandb/wandb/issues/6449
        try:
            if self.backend == "wandb":
                self.logger.finish(exit_code=0)
            else:
                self.logger.finish()
        except Exception as e:
            logger.warning(f"Attempted to finish tracking with backend {self.backend} but got error {e}")

    def log_exception(self, e: BaseException, step: int = 0) -> None:
        """Log the active exception's traceback to the configured backend.

        Always prints the traceback on the driver via loguru (so it lands in
        Ray driver logs instead of being swallowed). If the wandb backend is
        active, also logs a row to an `error/tracebacks` wandb.Table and calls
        `finish()` to flush the async upload before the caller re-raises.

        Ray-wrapped worker errors (e.g. OOMs raised inside actors) include
        both local and remote frames in `traceback.format_exc()`.

        Idempotent: a second call is a no-op so an outer except handler can
        safely call this even if an inner handler already did.
        """
        if self._exception_logged:
            return
        self._exception_logged = True
        tb_str = traceback.format_exc()[-10000:]
        logger.error(f"Training failed at step {step} with {type(e).__name__}:\n{tb_str}")
        if self.backend == "wandb":
            try:
                import wandb

                error_table = wandb.Table(columns=["step", "type", "traceback"])
                error_table.add_data(step, type(e).__name__, tb_str)
                # Note: omit `step=` here. Per-step logs use commit=True, so
                # re-logging at the same step would be dropped. The step value
                # is also embedded in the table row itself.
                self.logger.log({"error/tracebacks": error_table})
                # Tables upload asynchronously. Finish the run so the upload
                # completes before the caller re-raises and the process dies.
                try:
                    self.finish()
                except Exception as finish_exc:
                    logger.warning(f"tracker.finish() raised after logging exception: {finish_exc}")
            except Exception as log_exc:
                logger.warning(f"Failed to log exception traceback to wandb: {log_exc}")

    def log_samples_to_table(
        self,
        key: str,
        columns: List[str],
        samples: List[Tuple[Any, ...]],
        step: int,
    ) -> None:
        """Append rows to an accumulating wandb table at ``key``.

        Each call extends the existing table at ``key`` (or creates one on
        first call) with ``samples`` and logs the new table to wandb at the
        given ``step``. ``columns`` defines the table schema and must stay
        consistent across calls for the same ``key``; each row in ``samples``
        must have ``len(columns)`` values in the matching order.

        No-op for non-wandb backends -- only the wandb backend supports
        ``wandb.Table``.
        """
        if self.backend != "wandb":
            return
        import wandb

        # Cache one table per key so different callers (e.g. eval vs train
        # trajectory loggers, error traceback table) don't trample each other.
        if not hasattr(self, "_sample_tables"):
            self._sample_tables: Dict[str, Any] = {}
        if key not in self._sample_tables:
            self._sample_tables[key] = wandb.Table(columns=columns)
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        # Shallow-copy the previous table's rows: wandb.Table holds `data` by reference, so the
        # subsequent add_data() calls would otherwise mutate the row list of the already-logged
        # table (a data race with wandb's async upload of the prior step).
        new_table = wandb.Table(columns=columns, data=list(self._sample_tables[key].data))
        for row in samples:
            new_table.add_data(*row)
        self.logger.log({key: new_table}, step=step)
        self._sample_tables[key] = new_table

    def __del__(self):
        try:
            self.finish()
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
    def __init__(self, project_name, experiment_name, config: Optional[Union[SkyRLTrainConfig, DictConfig]] = None):
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
