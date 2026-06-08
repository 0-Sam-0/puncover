import re

from puncover import collector


class BacktraceHelper:
    def __init__(self, collector):
        self.collector = collector

    derive_functions_symbols_pattern = re.compile(r"\b(\w+)\b")

    def derive_function_symbols(self, text):
        result = []
        for f in self.derive_functions_symbols_pattern.finditer(text):
            s = self.collector.symbol(f.group(1), False)
            if s and s[collector.TYPE] == collector.TYPE_FUNCTION:
                result.append(s)
        return result

    def transform_known_symbols(self, text, transformer):
        def f(match):
            symbol_name = match.group(1)
            symbol = self.collector.symbol(symbol_name, False)
            return transformer(symbol) if symbol else symbol_name

        return self.derive_functions_symbols_pattern.sub(f, text)

    def deepest_call_tree(self, f, list_attribute, cache_attribute, visited=None):
        result, _ = self._deepest_call_tree(f, list_attribute, cache_attribute, visited)
        return result

    def _deepest_call_tree(self, f, list_attribute, cache_attribute, visited=None):
        # Returns (result, hit_cycle). The deepest call tree is the longest
        # *simple* path (each function counted once); cycles are broken via the
        # `visited` set.
        #
        # `hit_cycle` is True when a back-edge to a node already on the current
        # path had to be dropped. Such a result is PATH-DEPENDENT, so it must not
        # be memoized: caching it under one start node and reusing it from another
        # is exactly what produced order-dependent, under-counted worst-case
        # stacks (e.g. a function inside a mutual-recursion cycle reporting only
        # its own frame). Acyclic results do not depend on the path, so they are
        # cached as before and the traversal stays fast on the common case.
        if cache_attribute in f:
            return f[cache_attribute], False

        visited = [f] + (visited if visited else [])
        result = (0, [])
        hit_cycle = False

        for c in f[list_attribute]:
            if c in visited:
                hit_cycle = True
                continue
            candidate, child_hit_cycle = self._deepest_call_tree(
                c, list_attribute, cache_attribute, visited
            )
            hit_cycle = hit_cycle or child_hit_cycle
            if candidate[0] > result[0]:
                result = candidate

        result = (result[0] + f.get(collector.STACK_SIZE, 0), [f] + result[1])
        if not hit_cycle:
            f[cache_attribute] = result
        return result, hit_cycle

    def annotate_call_tree_flags(self):
        """Set the per-function call-tree warning flags (unresolved indirect
        calls, missing stack sizes, dynamic/unbound frames).

        Kept separate from the memoized deepest_call_tree on purpose: cyclic
        results are intentionally not cached (see _deepest_call_tree), so folding
        this bookkeeping into the traversal would double-count whenever a node is
        re-expanded. This pass runs once per function and is order-independent.
        Semantics match the historical behaviour: a function is flagged once for
        each direct callee/caller that performs an indirect call, lacks a stack
        size, or has a dynamic stack frame.
        """
        for f in self.collector.all_functions():
            unresolved = missing = unbound = 0
            for c in f.get(collector.CALLEES, []) + f.get(collector.CALLERS, []):
                if c.get(collector.PERFORMS_INDIRECT_CALL):
                    unresolved += 1
                if collector.STACK_SIZE not in c:
                    missing += 1
                if c.get(collector.STACK_QUALIFIERS) == "dynamic":
                    unbound += 1
            if unresolved:
                f[collector.UNRESOLVED_CALLS_IN_CALL_TREE] = unresolved
            if missing:
                f[collector.MISSING_STACKSIZE_IN_CALL_TREE] = missing
            if unbound:
                f[collector.UNBOUND_STACKSIZE_IN_CALL_TREE] = unbound

    def deepest_callee_tree(self, f):
        return self.deepest_call_tree(f, collector.CALLEES, collector.DEEPEST_CALLEE_TREE)

    def deepest_caller_tree(self, f):
        return self.deepest_call_tree(f, collector.CALLERS, collector.DEEPEST_CALLER_TREE)
