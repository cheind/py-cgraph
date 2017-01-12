
import cgraph.graphs as graphs



def simplify(node, other_rules=None):
    t = graphs.GraphTraversal(node)

    rules = [multiply_by_one_rule]
    if other_rules:
        rules.extend(other_rules)

    for n in t.forward_order():
        # Apply rules, stop after first success
        for r in rules:
            if r(n):
                break

    return node

def applies_to(klass):
    def wrapper(func):
        def wrapped_func(node):
            if isinstance(node, klass):
                return func(node)
            else:
                return False
        return wrapped_func
    return wrapper

import cgraph.ops.multiplication as mul
import cgraph.constants as c

def is_const(node, value=None):
    if isinstance(node, c.Constant):
        if value is not None:
            return node._value == value
        else:
            return True            
    return False

def replace_source(edges, node):
    return [(node, e[1]) for e in edges]

@applies_to(mul.Mul)
def multiply_by_one_rule(node):
    in_e = graphs.graph.in_edges(node)
    if is_const(in_e[0][0], 1):
        out_e = graphs.graph.out_edges(node)
        graphs.graph.remove_edges(in_e)
        graphs.graph.remove_edges(out_e)
        graphs.graph.add_edges(replace_source(out_e, in_e[1][0]))
        return True
    elif is_const(in_e[1][0], 1):
        out_e = graphs.graph.out_edges(node)
        graphs.graph.remove_edges(in_e)
        graphs.graph.remove_edges(out_e)
        graphs.graph.add_edges(replace_source(out_e, in_e[0][0]))
        return True
    return False
