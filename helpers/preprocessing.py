
def execute_preprocessing_tasks(fn_list: [[str, ()]]) -> None:
    """
    This is a helper function for executing an arbitrary number of preprocessing functions.
    They should be past as a list of strings like: ['lib.module.function', ]
    @param fn_list: a list of lists, inner list being ['path.to.function', (argument1, argument2, ...)]
    @return: None
    """
    for fn_string, args in fn_list:
        fns = fn_string.split('.')
        fn = __import__(fns[0])
        for fn_next in fns[1:]:
            fn = getattr(fn, fn_next)
        fn(*args)

#
# class PreProcessor:
#
#     def __init__(self, module, function):
#
#         module = __import__(module)
