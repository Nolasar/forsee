from src.utils.naming import to_snake_case
from src.activations.activations import Sigmoid, Softmax

ALL_OBJECTS = {
    Sigmoid,
    Softmax
}
ALL_OBJECTS_DICT = {
    func.__name__: func for func in ALL_OBJECTS
}
ALL_OBJECTS_DICT.update(
    {to_snake_case(func.__name__): func for func in ALL_OBJECTS}
)
def get(identifier):
    if identifier is None:
        return None 
    elif not isinstance(identifier, str):
        raise TypeError(
            f'identifier type must be str, not {type(identifier)}'
        )
    return ALL_OBJECTS_DICT.get(identifier, None)
