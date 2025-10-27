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
"""Sampler library for data generation."""
import functools
from typing import Callable, Optional, Tuple

import numpy as np
import numpy.matlib as npm


def sort_x_elements(x):
  """Sort elements of x vectors from smallest to largest.
  
  Args:
    x (np.array): input array of shape (n, dim), where each row is a vector
  
  Returns:
    np.array: array where each row has been sorted in ascending order
  
  Example:
    >>> x = np.array([[3, 1, 2], [5, 2, 4]])
    >>> sort_x_elements(x)
    array([[1, 2, 3],
           [2, 4, 5]])
  """
  return np.sort(x, axis=1)


def sample_and_sort_x(distribution_fn, n, dim):
  """Sample random x vectors and sort their elements.
  
  Args:
    distribution_fn (Callable): random function to sample x vector units
    n (int): number of samples
    dim (int): dimension of each x vector
  
  Returns:
    np.array: sorted x vectors of shape (n, dim)
  
  Example:
    >>> x_sorted = sample_and_sort_x(np.random.randn, 10, 20)
    >>> # Each row of x_sorted has elements sorted in ascending order
  """
  x = distribution_fn(n, dim)
  return sort_x_elements(x)


def apply_softmax(x):
  """Apply softmax to x vectors element-wise.
  
  Args:
    x (np.array): input array of shape (n, dim), where each row is a vector
  
  Returns:
    np.array: array where softmax is applied to each row
  
  Example:
    >>> x = np.array([[1, 2, 3], [0, 0, 0]])
    >>> apply_softmax(x)
    # Each row sums to 1
  """
  # For numerical stability, subtract max along each row
  x_shifted = x - np.max(x, axis=1, keepdims=True)
  exp_x = np.exp(x_shifted)
  return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def compute_squared_distance(w, x):
  """Compute ||x - w||^2 for task 4.
  
  Task 4: y = ||x - w||^2 where ||·|| is L2 norm.
  
  Args:
    w (np.array): coefficient array of shape (n, dim)
    x (np.array): input array of shape (n, dim)
  
  Returns:
    np.array: squared distances of shape (n,), where each value is ||x[i] - w[i]||^2
  
  Example:
    >>> w = np.array([[1, 2, 3], [0, 0, 0]])
    >>> x = np.array([[1, 2, 3], [1, 1, 1]])
    >>> compute_squared_distance(w, x)
    array([0, 3])  # [0^2+0^2+0^2, 1^2+1^2+1^2]
  """
  diff = x - w  # shape: (n, dim)
  squared_distances = np.sum(diff ** 2, axis=1)  # shape: (n,)
  return squared_distances


def str_to_distribution_fn(distribution_str):
  """Convert string representaiton to function."""
  mixtures = distribution_str.split(",")
  if len(mixtures) > 1:
    fns = [str_to_distribution_fn(mixture) for mixture in mixtures]

    def mixture_sample_fn(*args, **kwargs):
      samples = [fn(*args, **kwargs) for fn in fns]
      samples = np.stack(samples, axis=0)
      flat = samples.reshape(samples.shape[0], -1)
      indexer = np.random.randint(flat.shape[0], size=flat.shape[-1])
      flat = flat[indexer, np.arange(flat.shape[-1])]
      return flat.reshape(*samples.shape[1:])

    return mixture_sample_fn
  else:
    distribution_type, beta = distribution_str.split("+")
    distribution_type, alpha = distribution_type.split("*")
    alpha = float(alpha)
    beta = float(beta)
    if distribution_type == "uniform":
      distribution_fn = np.random.rand
    elif distribution_type == "normal":
      distribution_fn = np.random.randn
    elif distribution_type == "fixed":
      # Fixed distribution: generates the same vector for all samples
      # Seed is specified by alpha, value is specified by beta
      # Format: "fixed*seed+unused" (beta is ignored for fixed)
      seed = int(alpha)
      def fixed_distribution_fn(*args, **kwargs):
        """Generate fixed vector using specified seed."""
        rng = np.random.RandomState(seed)
        # Generate one fixed vector and repeat for all samples
        n_samples = args[0] if len(args) > 0 else kwargs.get('n', 1)
        dim = args[1] if len(args) > 1 else kwargs.get('dim', 1)
        fixed_vector = rng.randn(dim)  # Generate fixed vector from seed
        return np.tile(fixed_vector, (n_samples, 1))
      return fixed_distribution_fn
    else:
      raise ValueError("Unknown distribution type.")

    def distribution_fn_scaled(*args, **kwargs):
      return distribution_fn(*args, **kwargs) * alpha + beta

    return distribution_fn_scaled


class Sampler(object):
  """Samples linear regression data from specified distributions."""

  def __init__(
      self,
      length,
      dim,
      hidden_size,
      x_distribution_fn = np.random.randn,
      w_distribution_fn = np.random.randn,
      noise_std = 0.0,
      task_probs = None,
      use_sorted_x = False,
      task_mix_alpha = 1.0,
      task3_prob = 0.0,
  ):
    """Initializes the sampler.

    Args:
      length (int): Number of examplers to generate.
      dim (int): dimension of the x vectors.
      hidden_size (int): dimension of the generated vectors
      x_distribution_fn (Callable): random function to sample x vector units
      w_distribution_fn (Callable): random function to sample w vector units
      noise_std (float): adds gaussian noise if the value > 0.0. Default is 0.0
      task_probs (list or np.array, optional): probabilities for each task type [p1, p2, p3, p4].
        - Task 1: y = w^T x (standard linear regression)
        - Task 2: y = w^T sort(x) (sorted linear regression)
        - Task 3: y = (dim/sqrt(2)) * w^T softmax(x) (scaled softmax linear regression)
        - Task 4: y = ||x - w||^2 (squared distance)
        Default is [1.0, 0.0, 0.0, 0.0] (pure task 1).
      use_sorted_x (bool): DEPRECATED. Use task_probs instead.
      task_mix_alpha (float): DEPRECATED. Use task_probs instead.
      task3_prob (float): DEPRECATED. Use task_probs instead.
    """
    self.length = length
    self.dim = dim
    self.hidden_size = hidden_size
    self.x_distribution_fn = x_distribution_fn
    self.w_distribution_fn = w_distribution_fn
    self.noise_std = noise_std
    
    # Handle task probabilities
    if task_probs is not None:
      # New interface: use task_probs
      self.task_probs = np.array(task_probs, dtype=float)
      # Ensure we have exactly 4 task probabilities
      if len(self.task_probs) != 4:
        raise ValueError(f"task_probs must have exactly 4 elements, got {len(self.task_probs)}")
      # Verify probabilities sum to 1.0 (allow small numerical tolerance)
      prob_sum = np.sum(self.task_probs)
      if abs(prob_sum - 1.0) > 1e-6:
        raise ValueError(
            f"task_probs must sum to 1.0, got {prob_sum}. "
            f"Probabilities: {self.task_probs}"
        )
    else:
      # Legacy interface: convert old parameters to task_probs
      # task3_prob: probability of task 4 (||x-w||^2)
      # (1-task3_prob) * task_mix_alpha: probability of task 1 (w^T x)
      # (1-task3_prob) * (1-task_mix_alpha): probability of task 2 (w^T sort(x))
      # task 3 (w^T softmax(x)): 0.0
      p1 = (1.0 - task3_prob) * task_mix_alpha
      p2 = (1.0 - task3_prob) * (1.0 - task_mix_alpha)
      p3 = 0.0
      p4 = task3_prob
      self.task_probs = np.array([p1, p2, p3, p4], dtype=float)
    
    # Store legacy parameters for backward compatibility
    self.use_sorted_x = use_sorted_x
    self.task_mix_alpha = task_mix_alpha
    self.task3_prob = task3_prob

  def sample_x(self, n = 1):
    """Generates a random x vector.

    Args:
      n (int, optional): number of samples. Defaults to 1.

    Returns:
      Tuple[np.array, np.array]: x vector, x vector with paddings
      
    Note:
      The returned x is the raw sampled x (not sorted). If use_sorted_x=True,
      the sorting will be applied in calculate_y when computing y=w^Tx_sorted.
    """
    x = self.x_distribution_fn(n, self.dim)  # - 0.5
    x_vec = np.concatenate(
        (
            np.zeros((n, 1)),
            x,
            #            np.zeros((n, self.hidden_size - self.dim - 1)),
        ),
        axis=1,
    )
    return x, x_vec

  def calculate_y(self, x, coefficients, use_standard_task=None):
    """Calculates the y vector from the x vector and the coefficients.

    Args:
      x (np.array): x vector of shape (n_samples, dim).
      coefficients (np.array): weights of the linear regressor.
      use_standard_task (np.array, optional): boolean array of shape (n_samples,)
        indicating which samples use standard task (True) vs sorted task (False).
        If None, will be determined based on task_mix_alpha.

    Returns:
      Tuple[np.array, np.array]: y vector and y_vec (with padding)
      
    Note:
      - Each sample in a sequence uses the task type determined at sequence level
      - Standard task: y = w^T x
      - Sorted task: y = w^T x_sorted
    """
    n_samples = x.shape[0]
    
    # If use_standard_task is not provided, determine it based on task_mix_alpha
    if use_standard_task is None:
      if self.task_mix_alpha >= 1.0:
        use_standard_task = np.ones(n_samples, dtype=bool)
      elif self.task_mix_alpha <= 0.0:
        use_standard_task = np.zeros(n_samples, dtype=bool)
      else:
        random_vals = np.random.rand(n_samples)
        use_standard_task = random_vals < self.task_mix_alpha
    
    # Apply task-specific transformation
    if np.all(use_standard_task):
      # All use standard task: y = w^T x
      x_for_computation = x
    elif not np.any(use_standard_task):
      # All use sorted task: y = w^T x_sorted
      x_for_computation = sort_x_elements(x)
    else:
      # Mixed: some use standard, some use sorted
      x_sorted = sort_x_elements(x)
      x_for_computation = np.where(
          use_standard_task[:, np.newaxis],  # Broadcast to match x dimensions
          x,  # Use standard x
          x_sorted  # Use sorted x
      )
    
    # Compute y = w^T x (or w^T x_sorted depending on task type)
    y = np.einsum("bi,bi->b", coefficients, x_for_computation)[:, None]
    
    if self.noise_std > 0:
      y += self.noise_std * np.random.randn(*y.shape)
    y_vec = np.concatenate((y, np.zeros((x.shape[0], self.dim))), axis=1)
    return y, y_vec

  def sample_coefficients(self, n = 1, alpha = 1.0):
    """Generates a random coefficients vector for the linear regressor.

    Args:
      n (int, optional): batch size. Defaults to 1. alpha(float, optional):
        additional sscale. Defaults to 1.0
      alpha (float, optional): scale distribution. Defaults to 1

    Returns:
      np.array: coefficients vector
    """
    return self.w_distribution_fn(n, self.dim) * alpha

  def get_delimiter_vector(self, n = 1):
    """Generates a constant delimiter vector."""
    return np.zeros((n, self.hidden_size))

  @functools.lru_cache(maxsize=10)
  def get_precision(self,):
    # x = self.x_distribution_fn(10000, self.dim)
    # return np.linalg.inv(x.T @ x) * x.shape[0]
    return np.eye(self.dim)

  def sample(
      self,
      n = 1,
      alpha = 1.0,
      coefficients = None,
  ):
    """Generates a random sequence of x and y vector comes from a linear regressor.

    Args:
      n (int, optional): batch size. Defaults to 1.
      alpha (float, optional): scale distribution. Defaults to 1
      coefficients (Optional[np.ndarray], optional): weights of the regressor.
        Defaults to None.

    Returns:
      Tuple[np.array, np.array]: x,y sequences, weights of the regressor
      
    Note:
      For each sequence in the batch, the task type (standard vs sorted) is 
      determined once and used consistently for all (x,y) pairs in that sequence.
      The coefficient w is also fixed for each sequence.
    """
    if coefficients is None:
      coefficients = self.sample_coefficients(n, alpha)
    else:
      coefficients = npm.repmat(coefficients, n, 1)
    
    # Step 1: Sample task type for each sequence in the batch
    # Each sequence (sample) uses the same task type throughout
    n_samples = coefficients.shape[0]
    task_ids = np.random.choice(4, size=n_samples, p=self.task_probs)  # task_ids in {0, 1, 2, 3}
    
    # Store task_ids for external access (e.g., train_w.py needs this)
    self._last_task_ids = task_ids
    
    # Step 2: Coefficients (w) are already sampled (passed as argument or sampled above)
    # Each sequence has its own w, which is fixed for that sequence
    
    # Step 3: Generate (x, y) pairs for each position in the sequence
    out = []
    xs = []
    ys = []
    for _ in range(self.length):
      # Sample x for this position (i.i.d. across positions)
      x, x_vec = self.sample_x(coefficients.shape[0])
      
      # Compute y based on task type
      # Initialize y array
      y = np.zeros((n_samples, 1))
      
      # Process each task type
      for task_id in range(4):
        mask = (task_ids == task_id)
        if not np.any(mask):
          continue
        
        if task_id == 0:
          # Task 1: y = w^T x
          y[mask] = np.einsum("bi,bi->b", coefficients[mask], x[mask])[:, None]
        elif task_id == 1:
          # Task 2: y = w^T sort(x)
          x_sorted = sort_x_elements(x[mask])
          y[mask] = np.einsum("bi,bi->b", coefficients[mask], x_sorted)[:, None]
        elif task_id == 2:
          # Task 3: y = sqrt(dim^2 / 2) * w^T softmax(x) = (dim / sqrt(2)) * w^T softmax(x)
          x_softmax = apply_softmax(x[mask])
          dim = x.shape[1]  # dimension of x
          scale_factor = dim / np.sqrt(2.0)
          y[mask] = scale_factor * np.einsum("bi,bi->b", coefficients[mask], x_softmax)[:, None]
        elif task_id == 3:
          # Task 4: y = ||x - w||^2
          y[mask] = compute_squared_distance(coefficients[mask], x[mask])[:, None]
      
      # Add noise if specified (i.i.d. across positions)
      if self.noise_std > 0:
        y += self.noise_std * np.random.randn(*y.shape)
      
      # Create y_vec with padding
      y_vec = np.concatenate((y, np.zeros((x.shape[0], self.dim))), axis=1)
      out.append(x_vec)
      out.append(y_vec)
      xs.append(x)
      ys.append(y)
      # out.append(self.get_delimiter_vector(coefficients.shape[0]))
    out = np.stack(out, axis=1)
    xs = np.stack(xs, axis=1)
    ys = np.stack(ys, axis=1)
    return out, coefficients, xs, ys

  def get_last_task_ids(self):
    """Returns the task IDs from the last sample() call.
    
    Returns:
        np.array: array of task IDs with shape (n_samples,), values in {0, 1, 2, 3}
                  Returns None if sample() has not been called yet.
    """
    return getattr(self, '_last_task_ids', None)
