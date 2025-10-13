import typing
import logging

logger = logging.getLogger("func_mapper")

def map_function(
        func_map: dict = {},
        arg_map: typing.Callable[[list], list] = lambda x: x,
        map_all_attrs: bool = True,
        attr_map: dict = {},
        #  attr_def: dict = {},
        ):
    """ Sugar Method to create target function for existing library.

        func_map: map original function name to target implemented func.
        arg_map: make process for arguments
        attr_map: make attributes to transform automatically.
    """
    def _update_attrs(old_attrs):
        new_attrs = {}
        #  for k, def_val in attr_def:
        #      old_attrs.setdefault(k, def_val)

        for k, v in old_attrs.items():
            if k not in attr_map and not map_all_attrs:
                continue
            new_k = attr_map.get(k, k)
            if new_k in new_attrs:
                logger.warning(f"target attr:{new_k} already exists val:{new_attrs[new_k]} from mapper: {k}")
            new_attrs[new_k] = v

        return new_attrs

    def _wrapper(func):
        def _run(*args, **old_attrs):
            new_args = arg_map(args)
            new_attrs = _update_attrs(old_attrs)
            return func(*new_args, **new_attrs)
        return _run
    return {k: _wrapper(f) for k, f in func_map.items()}

class FunctionMapper(dict):
    """
        Key: original function name.
        Val: target implemented function to be called.
    """
    def add_mapper(self, func_map: dict):
        for k in func_map:
            if k in self:
                logger.warning(f"func:{k} already exists in mapper, override.")
        self.update(func_map)

    def map_functions(self, *func_names: list):
        def _wrapper(func):
            self.add_mapper({f: func for f in func_names})
            return func
        return _wrapper


    def add_arg_mapper(self, func_map):
        def _wrapper(arg_map_func):
            new_mapper = map_function(
                    func_map=func_map,
                    arg_map=arg_map_func)
            self.add_mapper(new_mapper)
            return arg_map_func
        return _wrapper


