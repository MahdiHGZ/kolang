from functools import wraps
from types import FunctionType
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Union, ValuesView, cast, overload)

from pyspark.sql.column import Column


def kolang_column_wrapper(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        result = method(*args, **kwargs)
        if isinstance(result, Column):
            result.__class__ = KolangColumn
        return result

    return wrapped


class KolangColumnMetaClass(type):
    def __new__(meta, classname, bases, classDict):
        new_class_dict = {}
        for attributeName, attribute in classDict.items():
            if isinstance(attribute, FunctionType):
                attribute = kolang_column_wrapper(attribute)
            new_class_dict[attributeName] = attribute

        child = super().__new__(meta, classname, bases, new_class_dict)

        for base in bases:
            if base == Column:
                for attributeName, attribute in base.__dict__.items():
                    if isinstance(attribute, FunctionType):
                        setattr(child, attributeName, kolang_column_wrapper(attribute))
        return child


class KolangColumn(Column, metaclass=KolangColumnMetaClass):
    """
    A column in a DataFrame base on pyspark.sql.column.Column with new methods!
    """

    def isNullOrIn(self, *vals: Any):
        """
        A boolean expression that is evaluated to true if the value of this
        expression is contained by the evaluated values of the arguments or being Null.
        Parameters
        ----------
        vals: Any
            The values to be checked.
        """
        return self.isNull() | self.isin(*vals)
