import torch
import numpy as np
import copy
import inspect
import math
import re
import warnings
from collections import OrderedDict
from copy import deepcopy
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torchvision
from torch import fx, nn
from torch.fx.graph_module import _CodeOnlyModule, _copy_attr, _USER_PRESERVED_ATTRIBUTES_KEY


def get_cov(activations):
    """computes the covariance matrix for a set of DNN activations

    This is the first step for calculating differences between
    predictive distributions, because random zero-mean weights will
    reproduce the covariance of the activations.

    This assumes the first dimension of the activations tensor is the stimulus
    dimension.
    """
    if torch.is_tensor(activations):
        x = activations.detach().clone()
        x = torch.reshape(x, [x.shape[0], -1])
        x -= torch.mean(x, 1, keepdim=True)
        return torch.matmul(x, x.T)
    else:
        x = np.reshape(activations, [activations.shape[0], -1])
        x -= np.mean(x, 1, keepdims=True)
        return np.matmul(x, x.T)

class LeafModuleAwareTracer(fx.Tracer):
    """
    An fx.Tracer that allows the user to specify a set of leaf modules, i.e.
    modules that are not to be traced through. The resulting graph ends up
    having single nodes referencing calls to the leaf modules' forward methods.
    """

    def __init__(self, *args, **kwargs):
        self.leaf_modules = {}
        if "leaf_modules" in kwargs:
            leaf_modules = kwargs.pop("leaf_modules")
            self.leaf_modules = leaf_modules
        super().__init__(*args, **kwargs)

    def is_leaf_module(self, m: nn.Module, module_qualname: str) -> bool:
        if isinstance(m, tuple(self.leaf_modules)):
            return True
        return super().is_leaf_module(m, module_qualname)


class NodePathTracer(LeafModuleAwareTracer):
    """
    NodePathTracer is an FX tracer that, for each operation, also records the
    name of the Node from which the operation originated. A node name here is
    a `.` separated path walking the hierarchy from top level module down to
    leaf operation or leaf module. The name of the top level module is not
    included as part of the node name. For example, if we trace a module whose
    forward method applies a ReLU module, the name for that node will simply
    be 'relu'.

    Some notes on the specifics:
        - Nodes are recorded to `self.node_to_qualname` which is a dictionary
          mapping a given Node object to its node name.
        - Nodes are recorded in the order which they are executed during
          tracing.
        - When a duplicate node name is encountered, a suffix of the form
          _{int} is added. The counter starts from 1.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track the qualified name of the Node being traced
        self.current_module_qualname = ""
        # A map from FX Node to the qualified name\#
        # NOTE: This is loosely like the "qualified name" mentioned in the
        # torch.fx docs https://pytorch.org/docs/stable/fx.html but adapted
        # for the purposes of the torchvision feature extractor
        self.node_to_qualname = OrderedDict()

    def call_module(self, m: torch.nn.Module, forward: Callable, args, kwargs):
        """
        Override of `fx.Tracer.call_module`
        This override:
        1) Stores away the qualified name of the caller for restoration later
        2) Adds the qualified name of the caller to
           `current_module_qualname` for retrieval by `create_proxy`
        3) Once a leaf module is reached, calls `create_proxy`
        4) Restores the caller's qualified name into current_module_qualname
        """
        old_qualname = self.current_module_qualname
        try:
            module_qualname = self.path_of_module(m)
            self.current_module_qualname = module_qualname
            if not self.is_leaf_module(m, module_qualname):
                out = forward(*args, **kwargs)
                return out
            return self.create_proxy("call_module", module_qualname, args, kwargs)
        finally:
            self.current_module_qualname = old_qualname

    def create_proxy(
        self, kind: str, target: fx.node.Target, args, kwargs, name=None, type_expr=None, *_
    ) -> fx.proxy.Proxy:
        """
        Override of `Tracer.create_proxy`. This override intercepts the recording
        of every operation and stores away the current traced module's qualified
        name in `node_to_qualname`
        """
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr)
        self.node_to_qualname[proxy.node] = self._get_node_qualname(self.current_module_qualname, proxy.node)
        return proxy

    def _get_node_qualname(self, module_qualname: str, node: fx.node.Node) -> str:
        node_qualname = module_qualname

        if node.op != "call_module":
            # In this case module_qualname from torch.fx doesn't go all the
            # way to the leaf function/op, so we need to append it
            if len(node_qualname) > 0:
                # Only append '.' if we are deeper than the top level module
                node_qualname += "."
            node_qualname += str(node)

        # Now we need to add an _{index} postfix on any repeated node names
        # For modules we do this from scratch
        # But for anything else, torch.fx already has a globally scoped
        # _{index} postfix. But we want it locally (relative to direct parent)
        # scoped. So first we need to undo the torch.fx postfix
        if re.match(r".+_[0-9]+$", node_qualname) is not None:
            node_qualname = node_qualname.rsplit("_", 1)[0]

        # ... and now we add on our own postfix
        for existing_qualname in reversed(self.node_to_qualname.values()):
            # Check to see if existing_qualname is of the form
            # {node_qualname} or {node_qualname}_{int}
            if re.match(rf"{node_qualname}(_[0-9]+)?$", existing_qualname) is not None:
                postfix = existing_qualname.replace(node_qualname, "")
                if len(postfix):
                    # existing_qualname is of the form {node_qualname}_{int}
                    next_index = int(postfix[1:]) + 1
                else:
                    # existing_qualname is of the form {node_qualname}
                    next_index = 1
                node_qualname += f"_{next_index}"
                break

        return node_qualname


def _is_subseq(x, y):
    """Check if y is a subsequence of x
    https://stackoverflow.com/a/24017747/4391249
    """
    iter_x = iter(x)
    return all(any(x_item == y_item for x_item in iter_x) for y_item in y)


def _warn_graph_differences(train_tracer: NodePathTracer, eval_tracer: NodePathTracer):
    """
    Utility function for warning the user if there are differences between
    the train graph nodes and the eval graph nodes.
    """
    train_nodes = list(train_tracer.node_to_qualname.values())
    eval_nodes = list(eval_tracer.node_to_qualname.values())

    if len(train_nodes) == len(eval_nodes) and all(t == e for t, e in zip(train_nodes, eval_nodes)):
        return

    suggestion_msg = (
        "When choosing nodes for feature extraction, you may need to specify "
        "output nodes for train and eval mode separately."
    )

    if _is_subseq(train_nodes, eval_nodes):
        msg = (
            "NOTE: The nodes obtained by tracing the model in eval mode "
            "are a subsequence of those obtained in train mode. "
        )
    elif _is_subseq(eval_nodes, train_nodes):
        msg = (
            "NOTE: The nodes obtained by tracing the model in train mode "
            "are a subsequence of those obtained in eval mode. "
        )
    else:
        msg = "The nodes obtained by tracing the model in train mode are different to those obtained in eval mode. "
    warnings.warn(msg + suggestion_msg)


def _get_leaf_modules_for_ops() -> List[type]:
    members = inspect.getmembers(torchvision.ops)
    result = []
    for _, obj in members:
        if inspect.isclass(obj) and issubclass(obj, torch.nn.Module):
            result.append(obj)
    return result


def _set_default_tracer_kwargs(original_tr_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    default_autowrap_modules = (math, torchvision.ops)
    default_leaf_modules = _get_leaf_modules_for_ops()
    result_tracer_kwargs = {} if original_tr_kwargs is None else original_tr_kwargs
    result_tracer_kwargs["autowrap_modules"] = (
        tuple(set(result_tracer_kwargs["autowrap_modules"] + default_autowrap_modules))
        if "autowrap_modules" in result_tracer_kwargs
        else default_autowrap_modules
    )
    result_tracer_kwargs["leaf_modules"] = (
        list(set(result_tracer_kwargs["leaf_modules"] + default_leaf_modules))
        if "leaf_modules" in result_tracer_kwargs
        else default_leaf_modules
    )
    return result_tracer_kwargs

class DualGraphModule(fx.GraphModule):
    """
    A derivative of `fx.GraphModule`. Differs in the following ways:
    - Requires a train and eval version of the underlying graph
    - Copies submodules according to the nodes of both train and eval graphs.
    - Calling train(mode) switches between train graph and eval graph.
    """

    def __init__(
        self, root: torch.nn.Module, train_graph: fx.Graph, eval_graph: fx.Graph, class_name: str = "GraphModule"
    ):
        """
        Args:
            root (nn.Module): module from which the copied module hierarchy is
                built
            train_graph (fx.Graph): the graph that should be used in train mode
            eval_graph (fx.Graph): the graph that should be used in eval mode
        """
        super(fx.GraphModule, self).__init__()

        self.__class__.__name__ = class_name

        self.train_graph = train_graph
        self.eval_graph = eval_graph

        # Copy all get_attr and call_module ops (indicated by BOTH train and
        # eval graphs)
        for node in chain(iter(train_graph.nodes), iter(eval_graph.nodes)):
            if node.op in ["get_attr", "call_module"]:
                if not isinstance(node.target, str):
                    raise TypeError(f"node.target should be of type str instead of {type(node.target)}")
                _copy_attr(root, self, node.target)

        # train mode by default
        self.train()
        self.graph = train_graph

        # (borrowed from fx.GraphModule):
        # Store the Tracer class responsible for creating a Graph separately as part of the
        # GraphModule state, except when the Tracer is defined in a local namespace.
        # Locally defined Tracers are not pickleable. This is needed because torch.package will
        # serialize a GraphModule without retaining the Graph, and needs to use the correct Tracer
        # to re-create the Graph during deserialization.
        if self.eval_graph._tracer_cls != self.train_graph._tracer_cls:
            raise TypeError(
                f"Train mode and eval mode should use the same tracer class. Instead got {self.eval_graph._tracer_cls} for eval vs {self.train_graph._tracer_cls} for train"
            )
        self._tracer_cls = None
        if self.graph._tracer_cls and "<locals>" not in self.graph._tracer_cls.__qualname__:
            self._tracer_cls = self.graph._tracer_cls

    def train(self, mode=True):
        """
        Swap out the graph depending on the selected training mode.
        NOTE this should be safe when calling model.eval() because that just
        calls this with mode == False.
        """
        # NOTE: Only set self.graph if the current graph is not the desired
        # one. This saves us from recompiling the graph where not necessary.
        if mode and not self.training:
            self.graph = self.train_graph
        elif not mode and self.training:
            self.graph = self.eval_graph
        return super().train(mode=mode)

    def _deepcopy_init(self):
        # See __deepcopy__ below
        return DualGraphModule.__init__

    def __deepcopy__(self, memo):
        # Same as the base class' __deepcopy__ from pytorch, with minor
        # modification to account for train_graph and eval_graph
        # https://github.com/pytorch/pytorch/blob/f684dbd0026f98f8fa291cab74dbc4d61ba30580/torch/fx/graph_module.py#L875
        #
        # This is using a bunch of private stuff from torch, so if that breaks,
        # we'll likely have to remove this, along with the associated
        # non-regression test.
        res = type(self).__new__(type(self))
        memo[id(self)] = res
        fake_mod = _CodeOnlyModule(copy.deepcopy(self.__dict__, memo))
        self._deepcopy_init()(res, fake_mod, fake_mod.__dict__["train_graph"], fake_mod.__dict__["eval_graph"])

        extra_preserved_attrs = [
            "_state_dict_hooks",
            "_load_state_dict_pre_hooks",
            "_load_state_dict_post_hooks",
            "_replace_hook",
            "_create_node_hooks",
            "_erase_node_hooks",
        ]
        for attr in extra_preserved_attrs:
            if attr in self.__dict__:
                setattr(res, attr, copy.deepcopy(self.__dict__[attr], memo))
        res.meta = copy.deepcopy(getattr(self, "meta", {}), memo)
        if _USER_PRESERVED_ATTRIBUTES_KEY in res.meta:
            for attr_name, attr in res.meta[_USER_PRESERVED_ATTRIBUTES_KEY].items():
                setattr(res, attr_name, attr)
        return res

def cov_extractor(
    model: torch.nn.Module,
    return_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
    train_return_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
    eval_return_nodes: Optional[Union[List[str], Dict[str, str]]] = None,
    tracer_kwargs: Optional[Dict[str, Any]] = None,
    suppress_diff_warning: bool = False,
    concrete_args: Optional[Dict[str, Any]] = None
) -> torch.fx.GraphModule:

    tracer_kwargs = _set_default_tracer_kwargs(tracer_kwargs)
    is_training = model.training

    if all(arg is None for arg in [return_nodes, train_return_nodes, eval_return_nodes]):

        raise ValueError(
            "Either `return_nodes` or `train_return_nodes` and `eval_return_nodes` together, should be specified"
        )

    if (train_return_nodes is None) ^ (eval_return_nodes is None):
        raise ValueError(
            "If any of `train_return_nodes` and `eval_return_nodes` are specified, then both should be specified"
        )

    if not ((return_nodes is None) ^ (train_return_nodes is None)):
        raise ValueError("If `train_return_nodes` and `eval_return_nodes` are specified, then both should be specified")

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
    mode_return_nodes: Dict[str, Dict[str, str]] = {"train": train_return_nodes, "eval": eval_return_nodes}
    for mode in ["train", "eval"]:
        if mode == "train":
            model.train()
        elif mode == "eval":
            model.eval()

        # Instantiate our NodePathTracer and use that to trace the model
        tracer = NodePathTracer(**tracer_kwargs)
        graph = tracer.trace(model, concrete_args=concrete_args)
        
        name = model.__class__.__name__ if isinstance(model, nn.Module) else model.__name__
        graph_module = fx.GraphModule(tracer.root, graph, name)
        
        
        #---- my insertion
        def _wrap_with_cov(graph_module: fx.GraphModule, tnode: fx.Node) -> fx.Node:
            with graph_module.graph.inserting_after(tnode):
                return graph_module.graph.call_function(get_cov, args=(tnode,))
        #---- end of my insertion

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
            if not any([re.match(rf"^{query}(\.|$)", n) is not None for n in available_nodes]):
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
        for n in reversed(nodes):
            module_qualname = tracer.node_to_qualname.get(n)
            if module_qualname is None:
                # NOTE - Know cases where this happens:
                # - Node representing creation of a tensor constant - probably
                #   not interesting as a return node
                # - When packing outputs into a named tuple like in InceptionV3
                continue
            for query in mode_return_nodes[mode]:
                depth = query.count(".")
                if ".".join(module_qualname.split(".")[: depth + 1]) == query:
                    #---- my insertion
                    cov_node = _wrap_with_cov(graph_module, n)
                    output_nodes[mode_return_nodes[mode][query]] = cov_node
                    #---- my insertion ends
                    #output_nodes[mode_return_nodes[mode][query]] = n
                    mode_return_nodes[mode].pop(query)
                    break
        output_nodes = OrderedDict(reversed(list(output_nodes.items())))
        
        # And add them in the end of the graph
        with graph_module.graph.inserting_after(nodes[-1]):
            graph_module.graph.output(output_nodes)
        
        # Remove unused modules / parameters
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
    graph_module = DualGraphModule(model, graphs["train"], graphs["eval"], class_name=name)

    # Restore original training mode
    model.train(is_training)
    graph_module.train(is_training)

    return graph_module