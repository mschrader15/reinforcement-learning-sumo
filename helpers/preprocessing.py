
def execute_preprocessing_tasks(fn_string_list: [str], args_list: tuple) -> None:
    """
    This is a helper function for executing an arbitrary number of preprocessing functions.
    They should be past as a list of strings like: ['lib.module.function', ]
    @param args_list: a list of arguments corresponding to those in the function string
    @param fn_string_list: a list 
    @return:
    """
    for fn_string, args in zip(fn_string_list, args_list):
        fns = fn_string.split('.')
        fn = __import__(fns[0])
        for fn_next in fns[1:]:
            fn = getattr(fn, fn_next)
        fn(*args)






class PreProcessor:

    def __init__(self, module, function):

        module = __import__(module)
