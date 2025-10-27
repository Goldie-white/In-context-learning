# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transformer predictor that predicts w instead of y."""
from typing import Optional, Tuple, Union

from absl import flags
import flax.linen as nn
from incontext import transformer_lib_flax
import jax.numpy as jnp
import numpy as np

Array = Union[jnp.ndarray, np.ndarray]

flags.DEFINE_bool("loss_on_x_steps", default=False, help="Take loss on x steps")


def extract_y(seq, offset = 0):
  """Extracts the y vector from the input tensor.

  Args:
          seq (torch.Tensor): tensor with shape (batch_size, seq_length,
            hidden_size)
          offset (int, optional): optional offset for where ys start. Defaults
            to 0.

  Returns:
          torch.Tensor: tensor with shape (batch_size, num_exemplars,
          hidden_size)
  """
  return seq[:, jnp.arange(offset, seq.shape[1], 2), :1]


def extract_x(seq, offset = 0):
  """Extracts the x vector from the input tensor.
  
  Args:
          seq (torch.Tensor): tensor with shape (batch_size, seq_length,
            hidden_size)
          offset (int, optional): optional offset for where xs start. Defaults
            to 0.

  Returns:
          torch.Tensor: tensor with shape (batch_size, num_exemplars,
          x_dim)
  """
  # x在偶数位置：0, 2, 4, ...
  # x向量格式：[0, x1, x2, ..., x_dim]，我们取第1维到最后
  return seq[:, jnp.arange(offset, seq.shape[1], 2), 1:]


def compute_y_from_w_and_x(w_pred, x, task_ids):
  """Compute y based on task type.
  
  Args:
    w_pred: predicted w vectors, shape (batch, num_exemplars, x_dim)
    x: input x vectors, shape (batch, num_exemplars, x_dim)
    task_ids: task type for each sample, shape (batch,), values in {0, 1, 2, 3}
  
  Returns:
    y_pred: predicted y values, shape (batch, num_exemplars, 1)
  """
  batch_size, num_exemplars, dim = x.shape
  
  # Compute all task outputs (always compute all, then select based on mask)
  # Task 0: y = w^T x (standard linear regression)
  y_task0 = jnp.einsum('bni,bni->bn', w_pred, x)
  
  # Task 1: y = w^T sort(x) (sorted linear regression)
  x_sorted = jnp.sort(x, axis=-1)
  y_task1 = jnp.einsum('bni,bni->bn', w_pred, x_sorted)
  
  # Task 2: y = (dim/sqrt(2)) * w^T softmax(x) (scaled softmax linear regression)
  x_shifted = x - jnp.max(x, axis=-1, keepdims=True)
  exp_x = jnp.exp(x_shifted)
  x_softmax = exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)
  scale_factor = dim / jnp.sqrt(2.0)
  y_task2 = scale_factor * jnp.einsum('bni,bni->bn', w_pred, x_softmax)
  
  # Task 3: y = ||x - w||^2 (squared distance)
  diff = x - w_pred
  y_task3 = jnp.sum(diff ** 2, axis=-1)
  
  # Select output based on task_ids using jnp.where cascades
  # Start with task 0
  y_pred = y_task0
  
  # Override with task 1 where task_ids == 1
  mask_1 = (task_ids == 1)[:, None]  # (batch, 1)
  y_pred = jnp.where(mask_1, y_task1, y_pred)
  
  # Override with task 2 where task_ids == 2
  mask_2 = (task_ids == 2)[:, None]
  y_pred = jnp.where(mask_2, y_task2, y_pred)
  
  # Override with task 3 where task_ids == 3
  mask_3 = (task_ids == 3)[:, None]
  y_pred = jnp.where(mask_3, y_task3, y_pred)
  
  return y_pred[..., None]  # shape: (batch, num_exemplars, 1)


class CausalLM_W(nn.Module):
  """CausalLM model that predicts w (weight vector) instead of y."""
  config: transformer_lib_flax.TransformerConfig
  x_dim: int  # Dimension of x vector (needed to determine w dimension)

  @nn.compact
  def __call__(
      self,
      *,
      inputs,
      train,
      task_ids = None,
      return_attention = False,
  ):
    """CausalLM that predicts w and computes y = w^T x.

    This is an alternative approach where:
    1. Transformer outputs w (x_dim dimensional weight vector)
    2. We compute y_pred = w^T x
    3. Loss is computed on (y_pred - y_target)^2

    Args:
      inputs (Array): input tensor of shape (batch, seq_len, x_dim+1)
      train (bool): training mode
      return_attention (bool): whether to return attentions

    Returns:
      Tuple[Array, Tuple[Array, ...]]: Tuple of loss and extras
    """
    config = self.config
    seq_from = inputs[:, :-1, :]
    mask = nn.attention.make_causal_mask(seq_from[:, :, 0])
    seq_enc, seq_hiddens, attn_weights = transformer_lib_flax.Transformer(
        config)(
            inputs=seq_from,
            mask=mask,
            train=train,
            return_attention=return_attention)

    # ⭐ 关键改动1：输出w向量（x_dim维），而不是标量y
    output_shape = self.x_dim  # 20维，不是1维
    
    seq_pred = nn.Dense(
        output_shape,
        kernel_init=config.linear_w_init,
        bias_init=config.linear_bias_init)(
            seq_enc)
    # seq_pred shape: (batch, seq_len-1, x_dim)

    # 提取y位置的w预测
    # y在奇数位置：1, 3, 5, ...（在seq_from中是0, 2, 4, ...因为去掉了最后一个）
    w_pred = seq_pred[:, jnp.arange(0, seq_pred.shape[1], 2), :]
    # w_pred shape: (batch, num_exemplars, x_dim)
    
    # ⭐ 关键改动2：获取对应的x值
    seq_target = inputs[:, 1:, :]
    # 从 seq_from 中提取x值（x在偶数位置：0, 2, 4, ...）
    x_for_y = seq_from[:, jnp.arange(0, seq_pred.shape[1], 2), 1:]
    # x_for_y shape: (batch, num_exemplars, x_dim)
    
    # ⭐ 关键改动3：根据任务类型计算 y_pred
    if task_ids is None:
      # 向后兼容：如果没有提供 task_ids，默认使用 Task 0 (标准线性回归)
      task_ids = jnp.zeros(inputs.shape[0], dtype=jnp.int32)
    
    y_pred = compute_y_from_w_and_x(w_pred, x_for_y, task_ids)
    # y_pred shape: (batch, num_exemplars, 1)
    
    # 提取真实的y值
    y_target = extract_y(seq_target, offset=0)
    # y_target shape: (batch, num_exemplars, 1)
    
    # 计算误差
    y_errors = ((y_pred - y_target)**2).sum(axis=-1)
    # y_errors shape: (batch, num_exemplars)
    
    if config.loss_on_x_steps:
      errors = ((seq_pred - seq_target)**2).sum(axis=-1)
    else:
      errors = y_errors

    if return_attention:
      return errors, (y_errors, y_pred, seq_pred, seq_hiddens, attn_weights)
    else:
      return errors, (y_errors, y_pred, seq_pred, seq_hiddens)

  def extract_y(self, seq, offset = 0):
    return extract_y(seq, offset=offset)

