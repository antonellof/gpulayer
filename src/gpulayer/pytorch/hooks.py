"""When each training hook runs. Shared by ``Module``, ``DataModule`` and ``Callback``, whose
hooks have the same names (callbacks receive ``trainer, pl_module`` first).

Call order of ``fit`` (callbacks first, then the model, at each point):

    setup("fit") -> configure_model -> on_fit_start -> on_sanity_check_start ->
    [validation hooks] -> on_sanity_check_end -> on_train_start ->
    per epoch: on_train_epoch_start ->
        per batch: on_train_batch_start -> training_step -> on_before_backward ->
            backward -> on_after_backward -> on_before_optimizer_step -> optimizer step ->
            on_before_zero_grad -> zero_grad -> on_train_batch_end
        on_validation_start -> on_validation_epoch_start ->
            per batch: on_validation_batch_start -> validation_step -> on_validation_batch_end
        on_validation_epoch_end -> on_validation_end -> on_train_epoch_end
    on_train_end -> on_fit_end -> teardown("fit")
"""

from __future__ import annotations

from typing import Any

HOOK_DOCS: dict[str, str] = {
    "setup": "At the start of fit, validate, test or predict, in every process (``stage`` "
    'is "fit", "validate", "test" or "predict"); build layers or datasets here.',
    "teardown": "At the end of fit, validate, test or predict, in every process.",
    "prepare_data": "Once per machine (local rank 0) before ``setup``; download data here. "
    "Do not set state here: other processes do not run it.",
    "on_fit_start": "At the start of ``fit``, after setup and optimizers, before the sanity check.",
    "on_fit_end": "At the end of ``fit``. In the launching process of a submitted run, only "
    "this hook runs (after the outputs came back).",
    "on_sanity_check_start": "Before the validation batches run as a check before training.",
    "on_sanity_check_end": "After the sanity-check validation batches.",
    "on_train_start": "Once, before the first training epoch.",
    "on_train_end": "Once, after the last training epoch and before ``on_fit_end``.",
    "on_train_epoch_start": "At the start of each training epoch.",
    "on_train_epoch_end": "At the end of each training epoch, after its validation; "
    "``trainer.callback_metrics`` holds the epoch's values. ``self.log`` here adds to them.",
    "on_train_batch_start": "Before ``training_step``. On the model, returning ``-1`` ends "
    "the epoch.",
    "on_train_batch_end": "After the optimizer step of a batch (or after the batch, when "
    "accumulating); ``outputs`` is ``{'loss': loss}`` or the dict ``training_step`` returned.",
    "on_validation_start": "Before a validation run (also inside fit).",
    "on_validation_end": "After a validation run.",
    "on_validation_epoch_start": "Before the first validation batch, with the model in eval mode.",
    "on_validation_epoch_end": "After the last validation batch; ``self.log`` here adds to the "
    "validation results.",
    "on_validation_batch_start": "Before ``validation_step``.",
    "on_validation_batch_end": "After ``validation_step``, with what it returned.",
    "on_test_start": "Before ``Trainer.test`` runs its batches.",
    "on_test_end": "After ``Trainer.test``.",
    "on_test_epoch_start": "Before the first test batch.",
    "on_test_epoch_end": "After the last test batch; ``self.log`` here adds to the results.",
    "on_test_batch_start": "Before ``test_step``.",
    "on_test_batch_end": "After ``test_step``, with what it returned.",
    "on_predict_start": "Before ``Trainer.predict`` runs its batches.",
    "on_predict_end": "After ``Trainer.predict``.",
    "on_predict_epoch_start": "Before the first prediction batch.",
    "on_predict_epoch_end": "After the last prediction batch.",
    "on_predict_batch_start": "Before ``predict_step``.",
    "on_predict_batch_end": "After ``predict_step``, with the prediction (on the CPU).",
    "on_before_backward": "Before ``loss.backward()``, with the loss (unscaled).",
    "on_after_backward": "After the backward pass, before the optimizer step; gradients are set.",
    "on_before_optimizer_step": "Before the optimizer step and gradient clipping; with "
    "16-mixed the gradients are already unscaled.",
    "on_before_zero_grad": "Before the gradients are reset after an optimizer step.",
    "on_save_checkpoint": "When a checkpoint is written, with the dict about to be saved; "
    "add or change entries in place.",
    "on_load_checkpoint": "When a checkpoint is loaded (resume or ``load_from_checkpoint``), "
    "before the weights are restored; change entries in place.",
    "on_exception": "When fit stops with an exception or Ctrl+C, before it is raised (or, for "
    "Ctrl+C, before fit returns).",
    "on_before_batch_transfer": "Change a batch before it is moved to the device (returns the "
    "batch).",
    "on_after_batch_transfer": "Change a batch after it was moved to the device (returns the "
    "batch); for example GPU augmentation.",
}


def apply_hook_docs(cls: type, prefix: str = "") -> None:
    """Give ``cls``'s hooks the shared description when they have none."""
    for name, text in HOOK_DOCS.items():
        fn: Any = cls.__dict__.get(name)
        if fn is not None and not getattr(fn, "__doc__", None):
            fn.__doc__ = prefix + text
