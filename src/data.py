"""Data utils.
  Provide functions to create regression datasets.
"""

from functools import partial
import jax
from jax import vmap
import jax.numpy as jnp
import numpy as np


def create_reg_data_TI_onehot_eval(rng, i_size, c_size, size_distract, input_range, w_scale, queries=(0,1), return_true_order=False):
    """Create a linear regression data set corresponding to transitive inference data.
    Each X will be a concatenation of two onehot vectors representing two classes. Y will be 1 or -1, depending on the
    relative order of the two classes in the transitive inference hierarchy.

    Note, the weights of this regression should just reflect the rank of each item. 
    """
    assert i_size % 2 == 0, "i_size must be even"
    rng, new_rng, new_rng2, new_rng3, new_rng4 = jax.random.split(rng, 5)
    n_items = i_size // 2
    assert (queries[0] < n_items) & (queries[1] < n_items), "query contains item that's not there"
    true_order = jax.random.permutation(new_rng, jnp.arange(n_items))
    true_ranks = jnp.argsort(true_order)
    true_ranks = true_ranks - jnp.mean(true_ranks)
    pairs = list(zip(true_order, true_order[1:]))
    x = jnp.zeros((c_size, i_size))
    for i, (j, k) in enumerate(pairs):
        # fill first half with right order pairs
        x = x.at[i, j].set(1)
        x = x.at[i, n_items+k].set(1)
        # second half with left order pairs
        x = x.at[i+len(pairs), k].set(1)
        x = x.at[i+len(pairs), n_items+j].set(1)
    
    # the true weights are just reflections of the relative ranks 
    w = jnp.concat([true_ranks, -true_ranks])
    
    # first randomly shuffle the rows 
    random_order = jax.random.permutation(new_rng2, jnp.arange(x.shape[0]))
    x = x[random_order]
    
    # then the query is given in the argument 
    x_query = jnp.zeros((1, i_size))
    x_query = x_query.at[0, true_order[queries[0]]].set(1)
    x_query = x_query.at[0, n_items + true_order[queries[1]]].set(1)
    y_data = x @ w 
    y_target = x_query @ w
    y_target = y_target[..., None]
    
    # then write it as a sequence for the transformer
    seq = jnp.concatenate([x, y_data[..., None]], -1)
    target = jnp.concatenate([x_query, y_target], -1)
    x_querry_init = -1*x_query.dot(jnp.ones_like(x_query).T*0.0)
    zero = jnp.concatenate([x_query, x_querry_init], -1)
    seq = jnp.concatenate([seq, zero], 0)
    if return_true_order:
        return jnp.squeeze(seq), jnp.squeeze(target), w, true_order
    return jnp.squeeze(seq), jnp.squeeze(target), w


@partial(jax.jit, static_argnums=(1, 2, 3))
def create_reg_data_TI_onehot_train(rng, i_size, c_size, size_distract, input_range, w_scale):
    """Create a linear regression data set corresponding to transitive inference data.
    Each X will be a concatenation of two onehot vectors representing two classes. Y will be 1 or -1, depending on the
    relative order of the two classes in the transitive inference hierarchy.

    Note, the weights of this regression should just reflect the rank of each item. 
    """
    assert i_size % 2 == 0, "i_size must be even"
    rng, new_rng, new_rng2, new_rng3, new_rng4 = jax.random.split(rng, 5)
    n_items = i_size // 2
    true_order = jax.random.permutation(new_rng, jnp.arange(n_items))
    true_ranks = jnp.argsort(true_order)
    pairs = list(zip(true_order, true_order[1:]))
    
    x = jnp.zeros((c_size+1, i_size))
    for i, (j, k) in enumerate(pairs):
        # fill first half with right order pairs
        x = x.at[i, j].set(1)
        x = x.at[i, n_items+k].set(1)
        # second half with left order pairs
        x = x.at[i+len(pairs), k].set(1)
        x = x.at[i+len(pairs), n_items+j].set(1)
    
    # the true weights are just reflections of the relative ranks 
    w = jnp.concat([true_ranks, -true_ranks])
    
    # first randomly shuffle the rows 
    random_order = jax.random.permutation(new_rng2, jnp.arange(x.shape[0]))
    x = x[random_order]
    
    # then pick the query to be the final row
    x_query = jnp.expand_dims(x[-1],0)
    x = x[:-1]
    y_data = x @ w 
    y_target = x_query @ w
    y_target = y_target[..., None]
    
    # then write it as a sequence for the transformer
    seq = jnp.concatenate([x, y_data[..., None]], -1)
    target = jnp.concatenate([x_query, y_target], -1)
    x_querry_init = -1*x_query.dot(jnp.ones_like(x_query).T*0.0)
    zero = jnp.concatenate([x_query, x_querry_init], -1)
    seq = jnp.concatenate([seq, zero], 0)

    return jnp.squeeze(seq), jnp.squeeze(target), w


@partial(jax.jit, static_argnums=(1, 2, 3))
def create_reg_data(rng, i_size, c_size, size_distract, input_range, w_scale):
  """Create a linear regression data set: X*w where x ~ U(-1, 1), w ~ N(0,1)."""

  rng, new_rng, new_rng2, new_rng3, new_rng4 = jax.random.split(rng, 5)
  w = jax.random.normal(rng, shape=[i_size])*w_scale

  x = jax.random.uniform(new_rng, shape=[c_size, i_size],
                         minval=-input_range/2, maxval=input_range/2)
  x_querry = jax.random.uniform(new_rng2, shape=[1, i_size],
                                minval=-input_range/2, maxval=input_range/2)

  y_data = jnp.squeeze(x@w)
  choice = jax.random.choice(new_rng4, c_size, shape=[size_distract],
                             replace=False)
  y_data = y_data.at[choice].set(jax.random.normal(new_rng3, 
                                                   shape=[size_distract]))

  y_target = x_querry@w
  y_target = y_target[..., None]

  seq = jnp.concatenate([x, y_data[..., None]], -1)
  target = jnp.concatenate([x_querry, y_target], -1)
  x_querry_init = -1*x_querry.dot(jnp.ones_like(x_querry).T*0.0)
  zero = jnp.concatenate([x_querry, x_querry_init], -1)
  seq = jnp.concatenate([seq, zero], 0)
  return jnp.squeeze(seq), jnp.squeeze(target), w

data_creator = vmap(create_reg_data,
                    in_axes=(0, None, None, None, None, None), out_axes=0)

rng = jax.random.PRNGKey(0)
rng, test_rng_avg = jax.random.split(rng, 2)
test_data = data_creator(jax.random.split(rng, num=1), 3, 10, 0, 2, 1)


@partial(jax.jit, static_argnums=(1, 2))
def create_ood_data(rng, i_size, c_size, input_range, w_scale):
  """Create a ood data set: X*w where X ~ Normal, Exponential, Poisson."""

  rng, new_rng, new_rng2, new_rng3 = jax.random.split(rng, 4)
  w = jax.random.normal(rng, shape=[i_size])*w_scale

  selector = jnp.zeros([3])
  choice = jax.random.choice(new_rng3, 3, replace=False)
  selector = selector.at[choice].set(1)

  x_sample = jax.random.exponential(new_rng, shape=[c_size, i_size])
  norm_x_sample = jnp.linalg.norm(x_sample)
  x = x_sample/norm_x_sample*input_range*selector[0]
  x_q_sample = jax.random.exponential(new_rng2, shape=[1, i_size])
  x_querry = x_q_sample/norm_x_sample*input_range*selector[0]

  x_sample = jax.random.normal(new_rng, shape=[c_size, i_size])
  norm_x_sample = jnp.linalg.norm(x_sample)
  x += x_sample/norm_x_sample*input_range*selector[1]
  x_q_sample = jax.random.normal(new_rng2, shape=[1, i_size])
  x_querry += x_q_sample/norm_x_sample*input_range*selector[1]

  x_sample = jax.random.laplace(new_rng, shape=[c_size, i_size])
  norm_x_sample = jnp.linalg.norm(x_sample)
  x += x_sample/norm_x_sample*input_range*selector[2]
  x_q_sample = jax.random.laplace(new_rng2, shape=[1, i_size])
  x_querry += x_q_sample/norm_x_sample*input_range*selector[2]

  y_data = jnp.squeeze(x@w)

  y_target = x_querry@w
  y_target = y_target[..., None]

  seq = jnp.concatenate([x, y_data[..., None]], -1)
  target = jnp.concatenate([x_querry, y_target], -1)
  x_querry_init = -1*x_querry.dot(jnp.ones_like(x_querry).T*0.0)
  zero = jnp.concatenate([x_querry, x_querry_init], -1)
  seq = jnp.concatenate([seq, zero], 0)
  return jnp.squeeze(seq), jnp.squeeze(target), w

data_creator = vmap(create_ood_data,
                    in_axes=(0, None, None, None, None), out_axes=0)

rng = jax.random.PRNGKey(0)
rng, test_rng_avg = jax.random.split(rng, 2)
test_data = data_creator(jax.random.split(rng, num=1), 2, 4, 1, 1)


@partial(jax.jit, static_argnums=(1, 2, 3))
def create_reg_data_sin(rng, i_size, c_size, size_distract, 
                        input_range=10, w_scale=1):
  """Create a sin wave regression data set."""

  rng, new_rng, new_rng2, new_rng3, new_rng4 = jax.random.split(rng, 5)
  amp = jax.random.uniform(rng, shape=[1], minval=0.1, maxval=0.5)*w_scale
  phase = jax.random.uniform(rng, shape=[1], minval=0.0,
                             maxval=1)*jnp.pi*w_scale

  x = jax.random.uniform(new_rng, shape=[c_size, 1],
                         minval=-input_range/2, maxval=input_range/2)
  x_querry = jax.random.uniform(new_rng2, shape=[1, 1],
                                minval=-input_range/2, maxval=input_range/2)

  y_data = jnp.sin(x + phase)*amp
  choice = jax.random.choice(new_rng4, c_size, shape=[size_distract],
                             replace=False)
  y_data = y_data.at[choice].set(jax.random.normal(new_rng3,
                                                   shape=[size_distract, 1]))

  y_target = jnp.sin(x_querry + phase)*amp
  seq = jnp.concatenate([x, y_data], -1)
  target = jnp.concatenate([x_querry, y_target], -1)
  y_querry_init = jnp.zeros_like(y_target)

  zero = jnp.concatenate([x_querry, y_querry_init], -1)
  seq = jnp.concatenate([seq, zero], 0)
  return jnp.squeeze(seq), jnp.squeeze(target), (phase, amp)

data_creator = vmap(create_reg_data_sin,
                    in_axes=(0, None, None, None, None, None), out_axes=0)

rng = jax.random.PRNGKey(0)
rng, test_rng_avg = jax.random.split(rng, 2)
test_data = data_creator(jax.random.split(rng, num=1), 1, 10, 2, 10, 1)


@partial(jax.jit, static_argnums=(2, 3))
def create_reg_data_sin_test(rng, rng2, c_size, input_range, w_scale):
  """Dublicate of the obove - TODO."""

  amp = jax.random.uniform(rng2, shape=[1], minval=0.1,maxval=0.5)*w_scale
  phase = jax.random.uniform(rng2, shape=[1], minval=0.0,
                             maxval=1)*jnp.pi*w_scale

  x = jax.random.uniform(rng2, shape=[c_size, 1],
                         minval=-input_range/2, maxval=input_range/2)
  x_querry = jax.random.uniform(rng, shape=[1, 1],
                                minval=-input_range/2, maxval=input_range/2)
  y_data = jnp.sin(x + phase)*amp
  y_target = jnp.sin(x_querry + phase)*amp
  seq = jnp.concatenate([x, y_data], -1)
  target = jnp.concatenate([x_querry, y_target], -1)
  y_querry_init = jnp.zeros_like(y_target)

  zero = jnp.concatenate([x_querry, y_querry_init], -1)
  seq = jnp.concatenate([seq, zero], 0)
  return jnp.squeeze(seq), jnp.squeeze(target), (phase, amp)

data_creator = vmap(create_reg_data_sin_test,
                    in_axes=(0, None, None, None, None), out_axes=0)

rng = jax.random.PRNGKey(0)
rng, test_rng_avg = jax.random.split(rng, 2)
test_data = data_creator(jax.random.split(rng, num=10), test_rng_avg, 10, 10, 1)


@partial(jax.jit, static_argnums=(1, 2, 3))
def create_reg_data_classic_token(rng, i_size, c_size, size_distract,
                                  input_range, w_scale):
  """Create a linear regression data set: X*w where x ~ U[-1,1], w ~ N(0,1)."""

  rng, new_rng, new_rng2, new_rng3, new_rng4 = jax.random.split(rng, 5)
  w = jax.random.normal(rng, shape=[i_size])*w_scale

  x = jax.random.uniform(new_rng,
                         shape=[c_size, i_size])*input_range - (input_range/2)
  x_querry = jax.random.uniform(new_rng2,
                                shape=[1, i_size])*input_range - (input_range/2)
  y_data = jnp.squeeze(x@w) 
  y_data_zero = jnp.zeros_like(x[:, :-1])
  y_data = jnp.concatenate([y_data_zero, y_data[..., None]], axis=-1)
  y_target = x_querry@w
  choice = jax.random.choice(new_rng4, c_size, shape=[size_distract],
                             replace=False)

  y_data = y_data.at[choice].set(jax.random.normal(new_rng3,
                                                   shape=[size_distract,
                                                          i_size]))
  y_target_zero = jnp.zeros_like(x_querry[:, :-1])
  y_target = y_target[..., None]

  seq = jnp.concatenate([x, y_data], 1)
  seq = seq.reshape(-1, i_size)
  target = jnp.concatenate([y_target_zero, y_target], -1)
  seq = jnp.concatenate([seq, x_querry], 0)
  return jnp.squeeze(seq), jnp.squeeze(target), w

data_creator = vmap(create_reg_data_classic_token, 
                    in_axes=(0, None, None, None, None, None), out_axes=0)

rng = jax.random.PRNGKey(0)
rng, test_rng_avg = jax.random.split(rng, 2)
test_data = data_creator(jax.random.split(rng, num=1), 2, 10, 0, 2, 1)


def create_weights(i_size, o_size, c_size, lr, w_init, second_zero=False,
                   lin_diag=False, gd_deq=False, num_layers=1,
                   input_mlp_rnd=None, in_proj=False):
  """Create linear regression gradient descent weights for self-attention
  layer."""

  one = jnp.ones([i_size+o_size])
  one_in_size = jnp.ones([i_size])
  zero_out_size = jnp.zeros([o_size])
  one_out_size = jnp.ones([o_size])

  # Value matrix
  value_upper = jnp.zeros([i_size, i_size+o_size])
  value_lower_left = w_init[0]
  if lin_diag:
    value_lower_right = jnp.diag(one_out_size)*-2
  else:
    value_lower_right = jnp.diag(one_out_size)*-1

  if second_zero:
    value_lower_right = jnp.diag(zero_out_size)

  value_lower_part = jnp.concatenate([value_lower_left, value_lower_right],
                                     axis=1)
  value_matrix = jnp.concatenate([value_upper, value_lower_part], axis=0).T
  if lin_diag:
    value_matrix += jnp.diag(one)

  #value_bias = jnp.zeros([i_size + o_size])

  # Query and Key matrix
  query_upper_part = jnp.zeros([o_size, i_size+o_size])
  query_lower_left = jnp.diag(one_in_size)
  query_lower_right = jnp.zeros([i_size, o_size])
  query_lower_part = jnp.concatenate([query_lower_left, query_lower_right],
                                     axis=1)
  query_matrix = jnp.concatenate([query_lower_part, query_upper_part], axis=0)
  key_matrix = query_matrix

  #query_bias = jnp.zeros([i_size + o_size])
  #key_bias = query_bias

  # Projection matrix
  projection_upper_part = jnp.zeros([i_size, i_size+o_size])
  projection_lower_left = jnp.zeros([o_size, i_size])

  projection_lower_right = jnp.diag(one_out_size)*((1/c_size)*lr)

  if lin_diag:
    shifted_lr = jnp.diag(one_out_size)*(1/c_size)*(1/c_size)*lr
    projection_lower_right += shifted_lr

  projection_lower_part = jnp.concatenate([projection_lower_left,
                                           projection_lower_right], axis=1)
  projection_matrix = jnp.concatenate([projection_upper_part,
                                       projection_lower_part], axis=0)
  if lin_diag:
    projection_matrix -= jnp.diag(one)*(1/c_size)*(1/c_size)*lr

  #projection_bias = jnp.zeros([i_size + o_size])

  params_new = {}
  for l in range(num_layers):
    if num_layers == 1 or gd_deq:
      tra_name = 'Transformer_gd/multi_head_attention/'
    else:
      tra_name = 'Transformer_gd/~trans_block/layer_'+str(l)+'/'
    params_new[tra_name+ 'query'] = {'w': jnp.array(query_matrix)}
    params_new[tra_name+ 'value'] = {'w': jnp.array(value_matrix)}
    params_new[tra_name+ 'key'] = {'w': jnp.array(key_matrix)}
    params_new[tra_name+ 'linear'] = {'w': jnp.array(projection_matrix)}

  if in_proj:
    rng1, rng2, rng3 = jax.random.split(input_mlp_rnd, 3)
    w_embedding = jax.random.normal(rng1, shape=[11, 11])*jnp.sqrt(0.002/11)
    params_new['Transformer_gd/emb'] = {'w': w_embedding}
  elif input_mlp_rnd is not None:
    rng1, rng2, rng3 = jax.random.split(input_mlp_rnd, 3)
    w1 = jax.random.normal(rng1, shape=[40, 160])*jnp.sqrt(0.002/2)
    w2 = jax.random.normal(rng2, shape=[160, 40])*jnp.sqrt(0.002/40)
    #w3 = jax.random.normal(rng3, shape=[40, 41])*jnp.sqrt(0.002/40)
    b1 = jax.random.normal(rng1, shape=[160])*0
    b2 = jax.random.normal(rng2, shape=[40])*0
    #b3 = jax.random.normal(rng3, shape=[41])*0
    w_embedding = jax.random.normal(rng1, shape=[2, 40])*jnp.sqrt(0.002/2)
    params_new['Transformer_gd/input_mlp/linear'] = {'w': w1, 'b': b1}
    params_new['Transformer_gd/input_mlp/linear_1'] = {'w': w2, 'b': b2}
    params_new['Transformer_gd/emb'] = {'w': w_embedding}
    #params_new['Transformer_gd/mlp/linear_2'] = {'w': w3, 'b': b3}
    #params_new['Transformer_gd/mlp/linear_3'] = {'w': w3, 'b': b3}
  
  return params_new

create_weights(10, 1, 10, 0.1, jnp.ones([1, 1, 10])*0.1)




