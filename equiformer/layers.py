from jax import vmap
import jax
import jax.numpy as jnp
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
    spread: float
    l_f: int

    n_c: int

    def __init__(self, n_c: int, l_f: int, num_basis: int, max_center: float):
        # Basis for radial basis functions
        self.coeffs = jnp.ones((n_c, num_basis))
        self.centers = jnp.linspace(0, max_center, num=num_basis)
        self.spread = max_center / num_basis

        self.l_f = l_f
        self.n_c = n_c

    def __call__(self, edges: jnp.ndarray):
        """Apply function to edges of graph, containing displacements [..., num_edges, 3]."""
        # Encode angular part
        y = spherical.solid_harmonics_jit(edges, self.l_f)
        y = jnp.nan_to_num(y, copy=False)

        # Encode radial part and multiply
        radius = jnp.linalg.norm(edges, ord=2, axis=-1)
        rbfs = vmap(
            lambda c_i: jnp.exp(-self.spread * (radius - c_i) ** 2), out_axes=-1
        )(self.centers)
        rbfs = einsum(rbfs, self.coeffs, "... i, c i -> ... c")
        return einsum(rbfs, y, "... c, ... m -> ... c m")


class TensorProductLayer(eqx.Module):
    """Module for applying tensor product to graph."""

    filters: dict[int, list[SphericalFilter]]
    tp_weights: dict[int, dict[int, dict[int, jnp.ndarray]]]
    input_channels: dict[int, int]
    l_max: int | None

    # Create filters for spherical harmonics
    def __init__(
        self,
        filters: dict[int, list[int]],
        input_channels: dict[int, int],
        l_max: int | None = None,
    ):
        # Create filters
        self.filters = {}
        for l, l_fs in filters.items():
            for l_f in l_fs:
                # TODO Change default number of basis functions and spacing
                f = SphericalFilter(input_channels[l], l_f, 5, 3.5)
                self.filters.setdefault(l, []).append(f)

        # Initialise weights for tensor product
        self.l_max = l_max
        self.tp_weights = {}
        for l, n_c in input_channels.items():
            for l_f in filters[l]:
                l_os = tensor_product._possible_output(l, l_f, l_max=self.l_max)
                for l_o in l_os:
                    self.tp_weights.setdefault(l, {}).setdefault(l_f, {})[
                        l_o
                    ] = jnp.ones((n_c,))

        self.input_channels = input_channels

    def __call__(self, g: jraph.GraphsTuple):
        sum_n_node = g.nodes[COORDS].shape[-2]

        messages_all = {}
        for l, filters_l in self.filters.items():
            for filter in filters_l:
                # NOTE Be careful there are no self-edges
                edges_l = filter(g.edges)

                # Nodes should have shape [..., node, channel, m]
                nodes_relevant = g.nodes[l][..., g.senders, :, :]
                messages = tensor_product.tensor_product(
                    nodes_relevant,
                    edges_l,
                    self.tp_weights[l][filter.l_f],
                    l_max=self.l_max,
                    depthwise=True,
                )
                for l_o, messages_l_o in messages.items():
                    messages_all.setdefault(l_o, []).append(messages_l_o)

        # Combine all channels
        # TODO Look into vmap
        messages_all = {
            l_o: jnp.concatenate(messages_l_o, axis=-2)
            for l_o, messages_l_o in messages_all.items()
        }

        # Initialising with coordinates
        nodes_new = {-1: g.nodes[-1]}
        # TODO Look into vmap
        for l, messages in messages_all.items():
            # Performing aggregation of messages
            nodes_new[l] = jax.ops.segment_sum(
                messages, g.receivers, num_segments=sum_n_node
            )

        return g._replace(nodes=nodes_new)


class SelfInteractionLayer(eqx.Module):
    """Module for applying local self-interaction between features of same order."""

    in_channels: dict[int, int]
    out_channels: dict[int, int]
    weights: dict[int, jnp.ndarray]

    def __init__(self, channel_map: dict[int, tuple[int]]):
        """Initialise self-interaction layer with {l: (nc_l_in, nc_l_out), ...}"""
        self.weights = {}
        self.in_channels = {}
        self.out_channels = {}
        for l, (nc_in, nc_out) in channel_map.items():
            # Identity by default
            self.weights[l] = jnp.eye(nc_out, nc_in)
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

    def __init__(self, n_channels: dict[int, int], act_fn: callable = jax.nn.sigmoid):
        self.act_fn = act_fn

        self.biases = {}
        for l, n_c in n_channels.items():
            self.biases[l] = jnp.zeros((n_c, 1))

    def __call__(self, g: jraph.GraphsTuple):
        nodes_new = {-1: g.nodes[-1]}

        for l, bias in self.biases.items():
            nodes_l = g.nodes[l]
            nodes_scalar = jnp.linalg.norm(nodes_l, ord=2, axis=-1, keepdims=True)
            nodes_new[l] = self.act_fn(nodes_scalar + bias) * nodes_l

        return g._replace(nodes=nodes_new)
