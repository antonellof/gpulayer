"""Trainer callbacks, named as in PyTorch Lightning.

trainer = Trainer(max_epochs=50, callbacks=[
    EarlyStopping(monitor="val_loss", patience=3),
    ModelCheckpoint(monitor="val_loss"),
])
"""

from gpulayer.pytorch.callbacks.base import Callback, _call
from gpulayer.pytorch.callbacks.monitor import EarlyStopping, ModelCheckpoint, OnExceptionCheckpoint
from gpulayer.pytorch.callbacks.utility import (
    BasePredictionWriter,
    DeviceStatsMonitor,
    GradientAccumulationScheduler,
    LambdaCallback,
    LearningRateMonitor,
    ModelSummary,
    ProgressBar,
    SpikeDetection,
    ThroughputMonitor,
    Timer,
    TQDMProgressBar,
    TrainingSpikeException,
)
from gpulayer.pytorch.callbacks.weights import (
    BackboneFinetuning,
    BaseFinetuning,
    EMAWeightAveraging,
    ModelPruning,
    StochasticWeightAveraging,
    WeightAveraging,
)

__all__ = [
    "BackboneFinetuning",
    "BaseFinetuning",
    "BasePredictionWriter",
    "Callback",
    "DeviceStatsMonitor",
    "EMAWeightAveraging",
    "EarlyStopping",
    "GradientAccumulationScheduler",
    "LambdaCallback",
    "LearningRateMonitor",
    "ModelCheckpoint",
    "ModelPruning",
    "ModelSummary",
    "OnExceptionCheckpoint",
    "ProgressBar",
    "SpikeDetection",
    "StochasticWeightAveraging",
    "TQDMProgressBar",
    "ThroughputMonitor",
    "Timer",
    "TrainingSpikeException",
    "WeightAveraging",
    "_call",
]
