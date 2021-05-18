import inspect


def retrieve_name(var):
    """
    utility function to return the actual name of the variable used in the Python code.

    @param var: a Python variable of any type
    @return: str, name of input Python variable
    """
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    reqd_var_name = ''
    for var_name, var_val in callers_local_vars:
        if var_val is var:
            reqd_var_name = var_name
    return reqd_var_name