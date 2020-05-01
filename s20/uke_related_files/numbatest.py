import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict

# First create a dictionary using Dict.empty()
# Specify the data types for both key and value pairs

# Dict with key as strings and values of type float array
dict_param1 = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.float64[:],
)

# Dict with keys as string and values of type float
dict_param2 = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.float64,
)

# Type-expressions are currently not supported inside jit functions.
float_array = types.float64[:]

@njit
def add_values(d_param1, d_param2):
    # Make a result dictionary to store results
    # Dict with keys as string and values of type float array
    result_dict = Dict.empty(
        key_type=types.unicode_type,
        value_type=float_array,
    )

    for key in d_param1.keys():
      result_dict[key] = d_param1[key] + d_param2[key]

    return result_dict

dict_param1["hello"]  = np.asarray([1.5, 2.5, 3.5], dtype='f8')
dict_param1["world"]  = np.asarray([10.5, 20.5, 30.5], dtype='f8')

dict_param2["hello"]  = 1.5
dict_param2["world"]  = 10

final_dict = add_values(dict_param1, dict_param2)

print(final_dict)
# Output : {hello: [3. 4. 5.], world: [20.5 30.5 40.5]}