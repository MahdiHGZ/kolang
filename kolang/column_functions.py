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

import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.column import Column


def percent(col: Union[Column, str],
            partition_by: Union[Column, str] = None,
            r: int = 2) -> Column:
    """
    returns the percent of the value.
    .. versionadded:: 0.1.0
    Parameters
    ----------
    col : str, :class:`Column`
        column containing number values
    partition_by : str, :class:`Column`, optional
        partition of data.
    r : int, optional
        rounding a final result base on this (default = 2)

    Examples
    --------
    >>> df = spark.range(1, 5).toDF('count').withColumn('percent', percent('count'))
    >>> df.show()
    +-----+-------+
    |count|percent|
    +-----+-------+
    |    1|   10.0|
    |    2|   20.0|
    |    3|   30.0|
    |    4|   40.0|
    +-----+-------+
    """
    if partition_by is None:
        w = Window.partitionBy()
    else:
        w = Window.partitionBy(partition_by)
    return F.round(100 * F.col(col) / F.sum(col).over(w), r)


def median(col: str) -> Column:
    """
    Aggregate function: returns the median of the values in a group.
    .. versionadded:: 0.1.0
    Parameters
    ----------
    col : str
        column containing values.

    Examples
    --------
    >>> df = spark.range(0, 34, 3).toDF('value').withColumn('even', F.col('value') % 2 == 0)
    >>> df = df.groupBy('even').agg(median('value'))
    >>> df.show()
    +-----+-------------+
    | even|median(value)|
    +-----+-------------+
    | true|         15.0|
    |false|         18.0|
    +-----+-------------+
    """
    return (F.expr(f"percentile({col}, array(0.5))")[0]).alias(f'median({col})')


def str_array_to_array(col: Union[Column, str]) -> Column:
    """
        convert str_array to pysaprk array.
    .. versionadded:: 0.1.0
    Parameters
    ----------
    col : str, :class:`Column`
        column containing str_array.

    Examples
    --------
    >>> df = spark.createDataFrame([("['a', 'b', 'c']",),
    >>>                             ("[QYYpm9yz, QYY9l2m1, QYYlm0C6, QYYdWjNY, QYYdmgKC]",),
    >>>                             ("[]",), ("",), ("a",), (None,)], ['str_array'])
    >>> df = df.withColumn('array', str_array_to_array('str_array'))
    >>> df.show()
    >>> df.printSchema()
    +--------------------+--------------------+
    |           str_array|               array|
    +--------------------+--------------------+
    |     ['a', 'b', 'c']|     ['a', 'b', 'c']|
    |[QYYpm9yz, QYY9l2...|[QYYpm9yz, QYY9l2...|
    |                  []|                null|
    |                    |                null|
    |                   a|                 [a]|
    |                null|                null|
    +--------------------+--------------------+

    root
     |-- str_array: string (nullable = true)
     |-- array: array (nullable = true)
     |    |-- element: string (containsNull = true)
    """
    col = F.translate(col, '[]', '')
    col = F.when(col != '', col)
    return F.split(col, ', ')


def number_normalizer(col: Union[Column, str]) -> Column:
    """
       normalize numbers in string to en number.
    .. versionadded:: 0.1.0
    Parameters
    ----------
    col : str, :class:`Column`
        column containing string.

    Examples
    --------
    >>> df = spark.createDataFrame([("۰۹۱۲۴۱۷۸۷۵۷",),
    >>>                             ("۲۴۱۷۷7656۱۲",),
    >>>                             ("۲۴۱ a سلام ab8",),],
    >>>                             ['string'])
    >>> df = df.withColumn('normal_str', number_normalizer('string'))
    >>> df.show()
    +--------------+--------------+
    |        string|    normal_str|
    +--------------+--------------+
    |   ۰۹۱۲۴۱۷۸۷۵۷|   09124178757|
    |   ۲۴۱۷۷7656۱۲|   24177765612|
    |۲۴۱ a سلام ab8|241 a سلام ab8|
    +--------------+--------------+
    """
    number_map = {
        # ARABIC NUMBERS
        "٠١٢٣٤٥٦٧٨٩": "0123456789",
        # PERSIAN NUMBERS
        "۰۱۲۳۴۵۶۷۸۹": "0123456789"
    }

    for key, val in number_map.items():
        col = F.translate(col, key, val)
    return col


def cumulative_sum(col: Union[Column, str],
                   on_col: Union[Column, str],
                   ascending: bool = True,
                   partition_by: Union[Column, str] = None):
    """
       normalize numbers in string to en number.
    .. versionadded:: 0.1.0
    Parameters
    ----------
    col : str, :class:`Column`
        column containing string.
    on_col : str, :class:`Column`
        order data base on this column.
    ascending : str, :class:`Column`, optional
        type of ordering is ascending. (default = True)
    partition_by : str, :class:`Column`, optional
        partition of data.
    Examples
    --------
    >>> df = spark.range(0, 5).toDF('id').withColumn('value', F.lit(2))
    >>> df.withColumn('cumulative_sum', cumulative_sum('value','id'))
    >>> df.show()
    +---+-----+--------------+
    | id|value|cumulative_sum|
    +---+-----+--------------+
    |  0|    3|             3|
    |  1|    3|             6|
    |  2|    3|             9|
    |  3|    3|            12|
    |  4|    3|            15|
    +---+-----+--------------+
    """
    on_col = on_col if ascending else F.desc(on_col)
    if partition_by is None:
        w = Window.orderBy(on_col).rangeBetween(Window.unboundedPreceding, 0)
    else:
        w = Window.partitionBy(partition_by).orderBy(on_col).rangeBetween(Window.unboundedPreceding, 0)
    return F.sum(col).over(w)
