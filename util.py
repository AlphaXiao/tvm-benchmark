import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.tools.graph_transforms import TransformGraph


def _remove_assert(all_nodes):
    all_nodes_dict = {}
    for node in all_nodes:
        all_nodes_dict[node.name] = node

    new_nodes = []
    for i,node in enumerate(all_nodes):
        if "assert" in node.name.lower():
            continue

        new_inputs = []
        for inp in node.input:
            if "assert" in inp.lower():
                continue
            else:
                new_inputs.append(inp)

        del node.input[:]
        node.input.extend(new_inputs)
        new_nodes.append(node)

    graph_def = graph_pb2.GraphDef()
    graph_def.node.extend(new_nodes)
    return graph_def

def export_pb(session, output, inputs, outputs):
    with tf.gfile.GFile(output, "wb") as f:
        graph_def = session.graph.as_graph_def(add_shapes=True)
        graph_def = _remove_assert(graph_def.node)
        graph_def = tf.graph_util.convert_variables_to_constants(session, graph_def, outputs)
        graph_def = TransformGraph(
            graph_def,
            inputs,
            outputs,
            [
                "remove_nodes(op=Identity, op=CheckNumerics, op=StopGradient)",
                "sort_by_execution_order", # sort by execution order after each transform to ensure correct node ordering
                "remove_device",
                "sort_by_execution_order",
                "fold_batch_norms",
                "sort_by_execution_order",
                "fold_old_batch_norms",
                "sort_by_execution_order"
            ]
        )
        f.write(graph_def.SerializeToString())