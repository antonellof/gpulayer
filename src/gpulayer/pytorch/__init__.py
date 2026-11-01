"""PyTorch integration (requires ``torch``). See ADR-0014, ADR-0018 and ADR-0024.

Three layers, each usable alone:

* ``launch()``  -- run any PyTorch script, whatever framework it uses, on GPUlayer workers
* ``Runtime``   -- device, DDP, data sharding and precision helpers
* ``Module`` + ``Trainer`` -- a training API with PyTorch Lightning's names (hooks, callbacks,
  loggers, profilers, tuner) that also picks where training runs
"""

from gpulayer.pytorch.callbacks import (
    BackboneFinetuning,
    BaseFinetuning,
    BasePredictionWriter,
    Callback,
    DeviceStatsMonitor,
    EarlyStopping,
    EMAWeightAveraging,
    GradientAccumulationScheduler,
    LambdaCallback,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelPruning,
    ModelSummary,
    OnExceptionCheckpoint,
    ProgressBar,
    SpikeDetection,
    StochasticWeightAveraging,
    ThroughputMonitor,
    Timer,
    TQDMProgressBar,
    TrainingSpikeException,
    WeightAveraging,
)
from gpulayer.pytorch.cli import GPULayerCLI, LightningCLI
from gpulayer.pytorch.combined import CombinedLoader
from gpulayer.pytorch.launch import LaunchResult, in_gpulayer_job, is_launcher_process, launch
from gpulayer.pytorch.loggers import CSVLogger, Logger, MLFlowLogger, TensorBoardLogger, WandbLogger
from gpulayer.pytorch.module import AttributeDict, DataModule, Module
from gpulayer.pytorch.plugins import (
    AsyncCheckpointIO,
    CheckpointIO,
    DoublePrecision,
    HalfPrecision,
    MixedPrecision,
    Precision,
    TorchCheckpointIO,
    TorchSyncBatchNorm,
)
from gpulayer.pytorch.profilers import (
    AdvancedProfiler,
    PassThroughProfiler,
    Profiler,
    PyTorchProfiler,
    SimpleProfiler,
)
from gpulayer.pytorch.runtime import Runtime, resolve_accelerator, seed_everything
from gpulayer.pytorch.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy
from gpulayer.pytorch.trainer import LRSchedulerConfig, Trainer
from gpulayer.pytorch.tuner import BatchSizeFinder, LearningRateFinder, Tuner

# Lightning's names, for code moving over.
LightningModule = Module
LightningDataModule = DataModule

__all__ = [
    "AsyncCheckpointIO",
    "CheckpointIO",
    "CombinedLoader",
    "DeepSpeedStrategy",
    "DoublePrecision",
    "GPULayerCLI",
    "HalfPrecision",
    "LightningCLI",
    "MixedPrecision",
    "Precision",
    "TorchCheckpointIO",
    "TorchSyncBatchNorm",
    "AdvancedProfiler",
    "AttributeDict",
    "BackboneFinetuning",
    "BaseFinetuning",
    "BasePredictionWriter",
    "BatchSizeFinder",
    "CSVLogger",
    "Callback",
    "DDPStrategy",
    "DataModule",
    "DeviceStatsMonitor",
    "EMAWeightAveraging",
    "EarlyStopping",
    "FSDPStrategy",
    "GradientAccumulationScheduler",
    "LRSchedulerConfig",
    "LambdaCallback",
    "LaunchResult",
    "LearningRateFinder",
    "LearningRateMonitor",
    "LightningDataModule",
    "LightningModule",
    "Logger",
    "MLFlowLogger",
    "ModelCheckpoint",
    "ModelPruning",
    "ModelSummary",
    "Module",
    "OnExceptionCheckpoint",
    "PassThroughProfiler",
    "Profiler",
    "ProgressBar",
    "PyTorchProfiler",
    "Runtime",
    "SimpleProfiler",
    "SpikeDetection",
    "StochasticWeightAveraging",
    "TQDMProgressBar",
    "TensorBoardLogger",
    "ThroughputMonitor",
    "Timer",
    "Trainer",
    "TrainingSpikeException",
    "Tuner",
    "WandbLogger",
    "WeightAveraging",
    "in_gpulayer_job",
    "is_launcher_process",
    "launch",
    "resolve_accelerator",
    "seed_everything",
]
