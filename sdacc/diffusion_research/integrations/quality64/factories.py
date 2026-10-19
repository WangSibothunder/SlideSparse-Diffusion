"""Reuse frozen pilot setup/generation; omit their timing loops and prior fitting.

The original generation function is retained. The context-manager conversion
stops immediately after its definition and retains the original cleanup.
"""
import ast
import contextlib
import copy
import importlib
import importlib.util
import sys
from pathlib import Path


def factory(worker, root):
    source = Path(worker['source'])
    kind = worker['worker']
    if kind == 'sigma_plain':
        sys.path[:0] = [str(root/'code/v079/src'), str(root/'code/cem_plain_s15_20261016')]
        path = source/'run.py'
        spec = importlib.util.spec_from_file_location('quality64_sigma_driver', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        name = 'main'
    else:
        sys.path.insert(0, str(source/'src'))
        module = importlib.import_module('sdacc.diffusion_research.' +
            ('duca_pilot' if kind == 'alpha_duca' else 'sparge_pilot' if kind == 'flux_sparge' else 'cache_pilot'))
        path = Path(module.__file__)
        name = 'execute'
    tree = ast.parse(path.read_text())
    fn = copy.deepcopy(next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name))
    namespace = dict(module.__dict__)
    namespace['RATES'] = (15, 25, 35, 45)
    namespace['FROZEN_SCHEDULE'] = worker.get('schedule')

    # Annotate inherited setup receipts so the old speed-only scope is not
    # confused with the newly authorized 64-image evaluation.
    for key in ('write', 'atomic_json'):
        original = namespace.get(key)
        if original is None:
            continue
        def writer(path, data, *args, _write=original, **kwargs):
            if Path(path).name in ('frozen_inputs.json', 'frozen.json'):
                path = Path(path).with_name('inherited_runtime_inputs.json')
                data = dict(evaluation_samples=64, speed_repeated=False,
                    schedule_refit=False, inherited_speed_driver_fields=data)
            return _write(path, data, *args, **kwargs)
        namespace[key] = writer

    if kind == 'sigma_plain':
        fn.args.args = [ast.arg(arg='output')]
        class OutputAndTiming(ast.NodeTransformer):
            def visit_Assign(self, node):
                if any(isinstance(t, ast.Name) and t.id == 'out' for t in node.targets):
                    node.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()),
                        args=[ast.Name(id='output', ctx=ast.Load())], keywords=[])
                return self.generic_visit(node)
            def visit_Call(self, node):
                if (isinstance(node.func, ast.Attribute) and node.func.attr == 'generate'
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == 'model'
                    and node.args and isinstance(node.args[-1], ast.Constant) and node.args[-1].value is True):
                    node.args[-1] = ast.Constant(False)
                return self.generic_visit(node)
        fn = OutputAndTiming().visit(fn)

    yield_pair = ast.Expr(value=ast.Yield(value=ast.Tuple(elts=[
        ast.Name(id='model', ctx=ast.Load()), ast.Name(id='generate', ctx=ast.Load())], ctx=ast.Load())))
    direct = next((i for i,x in enumerate(fn.body) if isinstance(x, ast.FunctionDef) and x.name == 'generate'), None)
    if direct is not None:
        final = next(x.finalbody for x in fn.body[direct+1:] if isinstance(x, ast.Try))
        fn.body = fn.body[:direct+1] + [ast.Try(body=[yield_pair], handlers=[], orelse=[], finalbody=final)]
    else:
        ti, block = next((i,x) for i,x in enumerate(fn.body) if isinstance(x, ast.Try)
            and any(isinstance(y, ast.FunctionDef) and y.name == 'generate' for y in x.body))
        gi = next(i for i,x in enumerate(block.body) if isinstance(x, ast.FunctionDef) and x.name == 'generate')
        block.body = block.body[:gi+1]
        if kind in ('alpha_plain','flux_plain'):
            block.body.append(ast.Assign(targets=[ast.Name(id='schedule', ctx=ast.Store())],
                value=ast.Name(id='FROZEN_SCHEDULE', ctx=ast.Load())))
        block.body.append(yield_pair)
        block.handlers = []; block.orelse = []
        fn.body = fn.body[:ti] + [block]
    fn.decorator_list = []
    fn.name = '_quality64_factory'
    code = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(code, str(path), 'exec'), namespace)
    return contextlib.contextmanager(namespace[fn.name])
