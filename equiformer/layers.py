from jax import vmap
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from einops import einsum
import jraph

import equiformer.spherical as spherical
import equiformer.tensor_product as tensor_product

COORDS = -1


class SphericalFilter(eqx.Module):
    """Module for learnable filter applied to displacements.

    Uses radial basis functions and spherical harmonics.
    Calculated for multiple channels simultaneously.
    """

    coeffs: jnp.ndarray
    centers: jnp.ndarray
    l_f: int = eqx.field(static=True)
    n_c: int = eqx.field(static=True)

    def __init__(self, n_c: int, l_f: int, num_basis: int, max_center: float, key: jr.PRNGKey,):
        # Basis for radial basis functions
        self.coeffs = jr.uniform(key, (n_c, num_basis))
        self.centers = jnp.linspace(0, max_center, num=num_basis)

        self.l_f = l_f
        self.n_c = n_c

    def __call__(self, edges: jnp.ndarray):
        """Apply function to edges of graph, containing displacements [..., num_edges, 3]."""
        # Encode angular part
        # NOTE Wasteful to be recomputing spherical harmonics many times
        # Could easily move these to edge definition themselves
        y = spherical.solid_harmonics_jit(edges, self.l_f)
        y = jnp.nan_to_num(y, copy=False)  # Needed if self-edges added accidentally

        spread = (jnp.max(self.centers) - jnp.min(self.centers)) / len(self.centers)

        # Encode radial part and multiply
        radius = jnp.linalg.norm(edges, ord=2, axis=-1)
        rbfs = vmap(
            lambda c_i: jnp.exp(- spread * (radius - c_i) ** 2), out_axes=-1
        )(self.centers)
        rbfs = einsum(rbfs, self.coeffs, "... i, c i -> ... c")
        return einsum(rbfs, y, "... c, ... m -> ... c m")


class TFNLayer(eqx.Module):
    """Module for applying tensor product to graph."""

    filters: dict[int, list[SphericalFilter]]
    input_channels: dict[int, int] = eqx.field(static=True)
    l_max: int | None = eqx.field(static=True)

    # TODO Move CG coefficients into this layer
    # TODO Add pointwise skip weights

    # Create filters for spherical harmonics
    def __init__(
        self,
        filters: dict[int, list[int]],
        input_channels: dict[int, int],
        key: jr.PRNGKey,
        l_max: int | None = None,
    ):
        # Create filters
        self.filters = {}
        for l_i, l_fs in filters.items():
            for l_f in l_fs:
                # TODO Change default number of basis functions and spacing
                key, subkey = jr.split(key, num=2)
                f = SphericalFilter(input_channels[l_i], l_f, 5, 3.5, subkey)
                self.filters.setdefault(l_i, []).append(f)

        self.l_max = l_max
        self.input_channels = input_channels

    def __call__(self, g: jraph.GraphsTuple):
        sum_n_node = g.nodes[COORDS].shape[0]

        messages_all = {}
        for l_i, filters_l_i in self.filters.items():
            for sph_filter in filters_l_i:
                edges_l = sph_filter(g.edges)

                # Nodes should have shape [node, channel, m]
                nodes_senders = g.nodes[l_i][g.senders, :, :]
                messages = tensor_product.tensor_product(
                    nodes_senders,
                    edges_l,
                    l_max=self.l_max,
                    elementwise=False,
                )
                for l_o, messages_l_o in messages.items():
                    messages_all.setdefault(l_o, []).append(messages_l_o)

        # Combine all channels
        messages_all = {
            l_o: jnp.concatenate(messages_l_o, axis=-2)
            for l_o, messages_l_o in messages_all.items()
        }

        # Initialising with coordinates
        nodes_new = {COORDS: g.nodes[COORDS]}
        for l_i, messages in messages_all.items():
            # Performing aggregation of messages
            nodes_new[l_i] = jax.ops.segment_sum(
                messages, g.receivers, num_segments=sum_n_node
            )

        return g._replace(nodes=nodes_new)




class SelfInteractionLayer(eqx.Module):
    """Module for applying local self-interaction between features of same order."""

    in_channels: dict[int, int] = eqx.field(static=True)
    out_channels: dict[int, int] = eqx.field(static=True)
    weights: dict[int, jnp.ndarray]

    def __init__(self, channel_map: dict[int, tuple[int]], key: jr.PRNGKey):
        """Initialise self-interaction layer with {l: (nc_l_in, nc_l_out), ...}"""
        self.weights = {}
        self.in_channels = {}
        self.out_channels = {}
        for l, (nc_in, nc_out) in channel_map.items():
            key, subkey = jr.split(key, num=2)
            self.weights[l] = jr.normal(subkey, (nc_out, nc_in)) + 1.0
            self.in_channels[l] = nc_in
            self.out_channels[l] = nc_out

    def __call__(self, g: jraph.GraphsTuple):
        nodes_new = {-1: g.nodes[-1]}
        for l, w in self.weights.items():
            nodes_new[l] = einsum(w, g.nodes[l], "o i, ... i m -> ... o m")

        return g._replace(nodes=nodes_new)


class GateLayer(eqx.Module):
    """Module for implementing non-linear gating layer."""

    act_fn: callable
    biases: dict[int, jnp.ndarray]

    def __init__(self, n_channels: dict[int, int], key: jr.PRNGKey, act_fn: callable = jax.nn.sigmoid):
        self.act_fn = act_fn

        self.biases = {}
        for l, n_c in n_channels.items():
            key, subkey = jr.split(key, num=2)
            self.biases[l] = jr.normal(subkey, (n_c, 1))

    def __call__(self, g: jraph.GraphsTuple):
        nodes_new = {COORDS: g.nodes[COORDS]}

        for l, bias in self.biases.items():
            nodes_l = g.nodes[l]
            nodes_scalar = jnp.linalg.norm(nodes_l, ord=2, axis=-1, keepdims=True)
            nodes_new[l] = self.act_fn(nodes_scalar + bias) * nodes_l

        return g._replace(nodes=nodes_new)
