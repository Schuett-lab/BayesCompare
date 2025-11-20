import torch
import numpy as np
import re
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import torch
from torchvision.models.feature_extraction import (
    NodePathTracer,
    DualGraphModule,
    _warn_graph_differences,
    _set_default_tracer_kwargs,
)
from torch import fx, nn


def get_cov(activations):
    """computes the covariance matrix for a set of DNN activations

    This is the first step for calculating differences between
    predictive distributions, because random zero-mean weights will
    reproduce the covariance of the activations.

    This assumes the first dimension of the activations tensor is the stimulus
    dimension.
    """

    if torch.is_tensor(activations):
        module = torch
        # x = activations.detach().clone() # we dont want the covs to be detached from the graph because we would like to use them for training
    elif isinstance(activations, np.array):
        module = np
    else:  # Also we can catch bad arguments (not mandatory)
        raise NotImplementedError(
            "Activations must be either a torch tensor or a numpy array."
        )

    x = module.reshape(activations, [activations.shape[0], -1])
    x -= module.mean(x, 1, keepdims=True)
    return module.matmul(x, x.T)


def cov_extractor(
    model: torch.nn.Module,
    return_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
    train_return_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
    eval_return_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
    tracer_kwargs: Optional[Dict[str, Any]] = None,
    suppress_diff_warning: bool = False,
    concrete_args: Optional[Dict[str, Any]] = None,
) -> torch.fx.GraphModule:
    """
    Creates a minimal torch graph module that returns covariance matrices for the specified layers.
    This function is a modified version of the torchvision.models.feature_extraction.create_feature_extractor
    function, with the addition of covariance computation after the specified layers. It is used in the same
    way as create_feature_extractor except for this function returns covariance matrices rather than activations.
    For the documentation of create_feature_extractor, see: https://docs.pytorch.org/vision/stable/feature_extraction.html
    """

    tracer_kwargs = _set_default_tracer_kwargs(tracer_kwargs)
    is_training = model.training

    if all(
        arg is None for arg in [return_nodes, train_return_nodes, eval_return_nodes]
    ):

        raise ValueError(
            "Either `return_nodes` or `train_return_nodes` and `eval_return_nodes` together, should be specified"
        )

    if (train_return_nodes is None) ^ (eval_return_nodes is None):
        raise ValueError(
            "If any of `train_return_nodes` and `eval_return_nodes` are specified, then both should be specified"
        )

    if not ((return_nodes is None) ^ (train_return_nodes is None)):
        raise ValueError(
            "If `train_return_nodes` and `eval_return_nodes` are specified, then both should be specified"
        )

    # Put *_return_nodes into Dict[str, str] format
    def to_strdict(n) -> Dict[str, str]:
        if isinstance(n, list):
            return {str(i): str(i) for i in n}
        return {str(k): str(v) for k, v in n.items()}

    if train_return_nodes is None:
        return_nodes = to_strdict(return_nodes)
        train_return_nodes = deepcopy(return_nodes)
        eval_return_nodes = deepcopy(return_nodes)
    else:
        train_return_nodes = to_strdict(train_return_nodes)
        eval_return_nodes = to_strdict(eval_return_nodes)

    # Repeat the tracing and graph rewriting for train and eval mode
    tracers = {}
    graphs = {}
    mode_return_nodes: Dict[str, Dict[str, str]] = {
        "train": train_return_nodes,
        "eval": eval_return_nodes,
    }
    for mode in ["train", "eval"]:
        if mode == "train":
            model.train()
        elif mode == "eval":
            model.eval()

        # Instantiate our NodePathTracer and use that to trace the model
        tracer = NodePathTracer(**tracer_kwargs)
        graph = tracer.trace(model, concrete_args=concrete_args)

        name = (
            model.__class__.__name__ if isinstance(model, nn.Module) else model.__name__
        )
        graph_module = fx.GraphModule(tracer.root, graph, name)

        # ---- my insertion
        def _wrap_with_cov(graph_module: fx.GraphModule, tnode: fx.Node) -> fx.Node:
            with graph_module.graph.inserting_after(tnode):
                return graph_module.graph.call_function(get_cov, args=(tnode,))

        # ---- end of my insertion

        available_nodes = list(tracer.node_to_qualname.values())
        # FIXME We don't know if we should expect this to happen
        if len(set(available_nodes)) != len(available_nodes):
            raise ValueError(
                "There are duplicate nodes! Please raise an issue https://github.com/pytorch/vision/issues"
            )
        # Check that all outputs in return_nodes are present in the model
        for query in mode_return_nodes[mode].keys():
            # To check if a query is available we need to check that at least
            # one of the available names starts with it up to a .
            if not any(
                [re.match(rf"^{query}(\.|$)", n) is not None for n in available_nodes]
            ):
                raise ValueError(
                    f"node: '{query}' is not present in model. Hint: use "
                    "`get_graph_node_names` to make sure the "
                    "`return_nodes` you specified are present. It may even "
                    "be that you need to specify `train_return_nodes` and "
                    "`eval_return_nodes` separately."
                )

        # Remove existing output nodes (train mode)
        orig_output_nodes = []
        for n in reversed(graph_module.graph.nodes):
            if n.op == "output":
                orig_output_nodes.append(n)
        if not orig_output_nodes:
            raise ValueError("No output nodes found in graph_module.graph.nodes")

        for n in orig_output_nodes:
            graph_module.graph.erase_node(n)

        # Find nodes corresponding to return_nodes and make them into output_nodes
        nodes = [n for n in graph_module.graph.nodes]
        output_nodes = OrderedDict()

        # iterate over a snapshot of keys since we pop inside the loop
        pending_queries = list(mode_return_nodes[mode].keys())

        for n in reversed(nodes):
            module_qualname = tracer.node_to_qualname.get(n)
            if module_qualname is None:
                continue
            for query in pending_queries:
                depth = query.count(".")
                if ".".join(module_qualname.split(".")[: depth + 1]) == query:

                    # Insert cov node after the activation node and collect it
                    cov_node = _wrap_with_cov(graph_module, n)
                    output_nodes[mode_return_nodes[mode][query]] = cov_node

                    # remove from both our snapshot and the dict
                    pending_queries.remove(query)
                    mode_return_nodes[mode].pop(query)

                    break

        # Keep ordering stable
        output_nodes = OrderedDict(reversed(list(output_nodes.items())))

        # Refresh the graph tail after all cov nodes were inserted,
        # then append the output node at the real end of the graph
        graph_module.graph.lint()

        nodes_after = [n for n in graph_module.graph.nodes]
        with graph_module.graph.inserting_after(nodes_after[-1]):
            graph_module.graph.output(output_nodes)

        graph_module.graph.lint()
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        # Keep track of the tracer and graph, so we can choose the main one
        tracers[mode] = tracer
        graphs[mode] = graph

    # Warn user if there are any discrepancies between the graphs of the
    # train and eval modes
    if not suppress_diff_warning:
        _warn_graph_differences(tracers["train"], tracers["eval"])

    # Build the final graph module
    graph_module = DualGraphModule(
        model, graphs["train"], graphs["eval"], class_name=name
    )

    # Restore original training mode
    model.train(is_training)
    graph_module.train(is_training)

    return graph_module
