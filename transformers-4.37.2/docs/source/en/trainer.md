<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Trainer

The [`Trainer`] is a complete training and evaluation loop for PyTorch models implemented in the Transformers library. You only need to pass it the necessary pieces for training (model, tokenizer, dataset, evaluation function, training hyperparameters, etc.), and the [`Trainer`] class takes care of the rest. This makes it easier to start training faster without manually writing your own training loop. But at the same time, [`Trainer`] is very customizable and offers a ton of training options so you can tailor it to your exact training needs.

<Tip>

In addition to the [`Trainer`] class, Transformers also provides a [`Seq2SeqTrainer`] class for sequence-to-sequence tasks like translation or summarization. There is also the [`~trl.SFTTrainer`] class from the [TRL](https://hf.co/docs/trl) library which wraps the [`Trainer`] class and is optimized for training language models like Llama-2 and Mistral with autoregressive techniques. [`~trl.SFTTrainer`] also supports features like sequence packing, LoRA, quantization, and DeepSpeed for efficiently scaling to any model size.

<br>

Feel free to check out the [API reference](./main_classes/trainer) for these other [`Trainer`]-type classes to learn more about when to use which one. In general, [`Trainer`] is the most versatile option and is appropriate for a broad spectrum of tasks. [`Seq2SeqTrainer`] is designed for sequence-to-sequence tasks and [`~trl.SFTTrainer`] is designed for training language models.

</Tip>

Before you start, make sure [Accelerate](https://hf.co/docs/accelerate) - a library for enabling and running PyTorch training across distributed environments - is installed.

```bash
pip install accelerate

# upgrade
pip install accelerate --upgrade
```

This guide provides an overview of the [`Trainer`] class.

## Basic usage

[`Trainer`] includes all the code you'll find in a basic training loop:

1. perform a training step to calculate the loss
2. calculate the gradients with the [`~accelerate.Accelerator.backward`] method
3. update the weights based on the gradients
4. repeat this process until you've reached a predetermined number of epochs

The [`Trainer`] class abstracts all of this code away so you don't have to worry about manually writing a training loop every time or if you're just getting started with PyTorch and training. You only need to provide the essential components required for training, such as a model and a dataset, and the [`Trainer`] class handles everything else.

If you want to specify any training options or hyperparameters, you can find them in the [`TrainingArguments`] class. For example, let's define where to save the model in `output_dir` and push the model to the Hub after training with `push_to_hub=True`.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="your-model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)
```

Pass `training_args` to the [`Trainer`] along with a model, dataset, something to preprocess the dataset with (depending on your data type it could be a tokenizer, feature extractor or image processor), a data collator, and a function to compute the metrics you want to track during training.

Finally, call [`~Trainer.train`] to start training!

```py
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### Checkpoints

The [`Trainer`] class saves your model checkpoints to the directory specified in the `output_dir` parameter of [`TrainingArguments`]. You'll find the checkpoints saved in a `checkpoint-000` subfolder where the numbers at the end correspond to the training step. Saving checkpoints are useful for resuming training later.

```py
# resume from latest checkpoint
trainer.train(resume_from_checkpoint=True)

# resume from specific checkpoint saved in output directory
trainer.train(resume_from_checkpoint="your-model/checkpoint-1000")
```

You can save your checkpoints (the optimizer state is not saved by default) to the Hub by setting `push_to_hub=True` in [`TrainingArguments`] to commit and push them. Other options for deciding how your checkpoints are saved are set up in the [`hub_strategy`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.hub_strategy) parameter:

* `hub_strategy="checkpoint"` pushes the latest checkpoint to a subfolder named "last-checkpoint" from which you can resume training
* `hug_strategy="all_checkpoints"` pushes all checkpoints to the directory defined in `output_dir` (you'll see one checkpoint per folder in your model repository)

When you resume training from a checkpoint, the [`Trainer`] tries to keep the Python, NumPy, and PyTorch RNG states the same as they were when the checkpoint was saved. But because PyTorch has various non-deterministic default settings, the RNG states aren't guaranteed to be the same. If you want to enable full determinism, take a look at the [Controlling sources of randomness](https://pytorch.org/docs/stable/notes/randomness#controlling-sources-of-randomness) guide to learn what you can enable to make your training fully deterministic. Keep in mind though that by making certain settings deterministic, training may be slower.

## Customize the Trainer

While the [`Trainer`] class is designed to be accessible and easy-to-use, it also offers a lot of customizability for more adventurous users. Many of the [`Trainer`]'s method can be subclassed and overridden to support the functionality you want, without having to rewrite the entire training loop from scratch to accommodate it. These methods include:

* [`~Trainer.get_train_dataloader`] creates a training DataLoader
* [`~Trainer.get_eval_dataloader`] creates an evaluation DataLoader
* [`~Trainer.get_test_dataloader`] creates a test DataLoader
* [`~Trainer.log`] logs information on the various objects that watch training
* [`~Trainer.create_optimizer_and_scheduler`] creates an optimizer and learning rate scheduler if they weren't passed in the `__init__`; these can also be separately customized with [`~Trainer.create_optimizer`] and [`~Trainer.create_scheduler`] respectively
* [`~Trainer.compute_loss`] computes the loss on a batch of training inputs
* [`~Trainer.training_step`] performs the training step
* [`~Trainer.prediction_step`] performs the prediction and test step
* [`~Trainer.evaluate`] evaluates the model and returns the evaluation metrics
* [`~Trainer.predict`] makes predictions (with metrics if labels are available) on the test set

For example, if you want to customize the [`~Trainer.compute_loss`] method to use a weighted loss instead.

```py
from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss for 3 labels with different weights
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```

### Callbacks

Another option for customizing the [`Trainer`] is to use [callbacks](callbacks). Callbacks *don't change* anything in the training loop. They inspect the training loop state and then execute some action (early stopping, logging results, etc.) depending on the state. In other words, a callback can't be used to implement something like a custom loss function and you'll need to subclass and override the [`~Trainer.compute_loss`] method for that.

For example, if you want to add an early stopping callback to the training loop after 10 steps.

```py
from transformers import TrainerCallback

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, num_steps=10):
        self.num_steps = num_steps
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.num_steps:
            return {"should_training_stop": True}
        else:
            return {}
```

Then pass it to the [`Trainer`]'s `callback` parameter.

```py
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callback=[EarlyStoppingCallback()],
)
```

## Logging

<Tip>

Check out the [logging](./main_classes/logging) API reference for more information about the different logging levels.

</Tip>

The [`Trainer`] is set to `logging.INFO` by default which reports errors, warnings, and other basic information. A [`Trainer`] replica - in distributed environments - is set to `logging.WARNING` which only reports errors and warnings. You can change the logging level with the [`log_level`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.log_level) and [`log_level_replica`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.log_level_replica) parameters in [`TrainingArguments`].

To configure the log level setting for each node, use the [`log_on_each_node`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.log_on_each_node) parameter to determine whether to use the log level on each node or only on the main node.

<Tip>

[`Trainer`] sets the log level separately for each node in the [`Trainer.__init__`] method, so you may want to consider setting this sooner if you're using other Transformers functionalities before creating the [`Trainer`] object.

</Tip>

For example, to set your main code and modules to use the same log level according to each node:

```py
logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

trainer = Trainer(...)
```

Use different combinations of `log_level` and `log_level_replica` to configure what gets logged on each of the nodes.

<hfoptions id="logging">
<hfoption id="single node">

```bash
my_app.py ... --log_level warning --log_level_replica error
```

</hfoption>
<hfoption id="multi-node">

Add the `log_on_each_node 0` parameter for multi-node environments.

```bash
my_app.py ... --log_level warning --log_level_replica error --log_on_each_node 0

# set to only report errors
my_app.py ... --log_level error --log_level_replica error --log_on_each_node 0
```

</hfoption>
</hfoptions>

## NEFTune

[NEFTune](https://hf.co/papers/2310.05914) is a technique that can improve performance by adding noise to the embedding vectors during training. To enable it in [`Trainer`], set the `neftune_noise_alpha` parameter in [`TrainingArguments`] to control how much noise is added.

```py
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(..., neftune_noise_alpha=0.1)
trainer = Trainer(..., args=training_args)
```

NEFTune is disabled after training to restore the original embedding layer to avoid any unexpected behavior.

## Accelerate and Trainer

The [`Trainer`] class is powered by [Accelerate](https://hf.co/docs/accelerate), a library for easily training PyTorch models in distributed environments with support for integrations such as [FullyShardedDataParallel (FSDP)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) and [DeepSpeed](https://www.deepspeed.ai/).

<Tip>

Learn more about FSDP sharding strategies, CPU offloading, and more with the [`Trainer`] in the [Fully Sharded Data Parallel](fsdp) guide.

</Tip>

To use Accelerate with [`Trainer`], run the [`accelerate.config`](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-config) command to set up training for your training environment. This command creates a `config_file.yaml` that'll be used when you launch your training script. For example, some example configurations you can setup are:

<hfoptions id="config">
<hfoption id="DistributedDataParallel">

```yml
compute_environment: LOCAL_MACHINE                                                                                             
distributed_type: MULTI_GPU                                                                                                    
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0 #change rank as per the node
main_process_ip: 192.168.20.1
main_process_port: 9898
main_training_function: main
mixed_precision: fp16
num_machines: 2
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</hfoption>
<hfoption id="FSDP">

```yml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: BertLayer
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</hfoption>
<hfoption id="DeepSpeed">

```yml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: /home/user/configs/ds_zero3_config.json
  zero3_init_flag: true
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</hfoption>
<hfoption id="DeepSpeed with Accelerate plugin">

```yml
compute_environment: LOCAL_MACHINE                                                                                             
deepspeed_config:                                                                                                              
  gradient_accumulation_steps: 1
  gradient_clipping: 0.7
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: true
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

</hfoption>
</hfoptions>

The [`accelerate_launch`](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch) command is the recommended way to launch your training script on a distributed system with Accelerate and [`Trainer`] with the parameters specified in `config_file.yaml`. This file is saved to the Accelerate cache folder and automatically loaded when you run `accelerate_launch`.

For example, to run the [run_glue.py](https://github.com/huggingface/transformers/blob/f4db565b695582891e43a5e042e5d318e28f20b8/examples/pytorch/text-classification/run_glue.py#L4) training script with the FSDP configuration:

```bash
accelerate launch \
    ./examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path bert-base-cased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --output_dir /tmp/$TASK_NAME/ \
    --overwrite_output_dir
```

You could also specify the parameters from the `config_file.yaml` file directly in the command line:

```bash
accelerate launch --num_processes=2 \
    --use_fsdp \
    --mixed_precision=bf16 \
    --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP  \
    --fsdp_transformer_layer_cls_to_wrap="BertLayer" \
    --fsdp_sharding_strategy=1 \
    --fsdp_state_dict_type=FULL_STATE_DICT \
    ./examples/pytorch/text-classification/run_glue.py
    --model_name_or_path bert-base-cased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --output_dir /tmp/$TASK_NAME/ \
    --overwrite_output_dir
```

Check out the [Launching your Accelerate scripts](https://huggingface.co/docs/accelerate/basic_tutorials/launch) tutorial to learn more about `accelerate_launch` and custom configurations.
