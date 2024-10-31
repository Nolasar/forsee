from src.utils.naming import to_snake_case
from src.initializers.initializers import GlorotUniform, Zeros

ALL_OBJECTS = {
    GlorotUniform,
    Zeros
}
ALL_OBJECTS_DICT = {
    cls.__name__: cls for cls in ALL_OBJECTS
}
ALL_OBJECTS_DICT.update({
    to_snake_case(cls.__name__): cls for cls in ALL_OBJECTS
})


def get(identifier):
    if identifier is None:
        return None 
    elif not isinstance(identifier, str):
        raise TypeError(
            f'identifier type must be str, not {type(identifier)}'
        )
    return ALL_OBJECTS_DICT.get(identifier, None)


