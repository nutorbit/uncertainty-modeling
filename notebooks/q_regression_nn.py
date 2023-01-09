"""
An implementation of Neural Quantile Regression.
"""

import jax
import optax
import jax.numpy as jnp
import haiku as hk

from collections import namedtuple
from dataclasses import dataclass


@dataclass
class MLP(hk.Module):
    hiddens = [32, 32]
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for n in self.hiddens:
            x = hk.Linear(n)(x)
            # x = jax.nn.leaky_relu(x)
        x = hk.Linear(1)(x)
        return x
    

class QuantileNN:
    def __init__(self, seed: int, n_features: int):
        self.rng = jax.random.PRNGKey(seed)
        self.n_features = n_features
        self.qs = [0.05, 0.5, 0.95]
        self.model = hk.without_apply_rng(hk.transform(
            lambda x: MLP()(x)
        ))
        self.opt = optax.adam(1e-2)
        
        self.params = self.init_params()
        self.opt_states = self.init_optimizers(self.params)
        
        self.loss_fn = jax.jit(self.loss_fn)
        self.update_model = jax.jit(self.update_model)
        
    def init_params(self) -> Tuple:
        params = []
        for _ in range(len(self.qs)):
            rng_init, rng = jax.random.split(self.rng, 2)
            param = self.model.init(rng_init, jnp.zeros((3, self.n_features)))
            params.append(param)
            self.rng = rng
        return tuple(params)
    
    def init_optimizers(self, params: Tuple) -> Tuple:
        opts = []
        for param in params:
            opt_state = self.opt.init(param)
            opts.append(opt_state)
        return tuple(opts)
    
    def loss_fn(self, param, X: jnp.ndarray, y: jnp.ndarray, q) -> jnp.ndarray:
        pred = self.model.apply(param, X)
        return jnp.maximum(
            (y - pred) * q, 
            (pred - y) * (1 - q)
        ).mean()
        
    def update_model(self, param, opt_state, X, y, q) -> Tuple:
        grad_fn = jax.value_and_grad(self.loss_fn)
        loss, grads = grad_fn(param, X, y, q)
        updates, opt_state = self.opt.update(grads, opt_state)
        param = optax.apply_updates(param, updates)
        
        return (
            param,
            opt_state,
            loss
        )
    
    def fit(self, X: jnp.ndarray, y: jnp.ndarray, epoch: int = 100) -> jnp.ndarray:
        rs = []
        for _ in range(epoch):
            r = []
            params = []
            opt_states = []
            for i, q in enumerate(self.qs):
                param = self.params[i]
                opt_state = self.opt_states[i]
                
                param, opt_state, loss = self.update_model(param, opt_state, X, y, q)
                
                params.append(param)
                opt_states.append(opt_state)
                
                r.append(loss)

            rs.append(r)
            self.params = params
            self.opt_states = opt_states
        return jnp.array(rs)
    
    def predict(self, X: jnp.ndarray, median: bool = True) -> jnp.ndarray:
        rs = []
        for param in self.params:
            pred = self.model.apply(param, X)
            rs.append(pred)
        preds = jnp.concatenate(rs, axis=1)
        return preds[:, 1] if median else preds
    

"""
model = QuantileNN(123, n_features)

model.fit(
    X, y
)

y = model.predict(X)
"""
