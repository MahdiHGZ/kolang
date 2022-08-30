from typing import (
    Any,
    cast,
    Callable,
    Dict,
    List,
    overload,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
    ValuesView,
)

from functools import wraps
from types import FunctionType
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
        newClassDict = {}
        for attributeName, attribute in classDict.items():
            if isinstance(attribute, FunctionType):
                attribute = kolang_column_wrapper(attribute)
            newClassDict[attributeName] = attribute

        child = super().__new__(meta, classname, bases, newClassDict)

        for base in bases:
            if base == Column:
                for attributeName, attribute in base.__dict__.items():
                    if isinstance(attribute, FunctionType):
                        setattr(child, attributeName, kolang_column_wrapper(attribute))
        return child


class KolangColumn(Column, metaclass=KolangColumnMetaClass):

    def isNullOrIn(self, *cols: Any):
        return self.isNull() | self.isin(*cols)
