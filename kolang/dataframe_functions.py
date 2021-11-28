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
from pyspark.sql.column import Column
from pyspark.sql.dataframe import DataFrame


def unpivot(df: DataFrame,
            on_columns: List[str],
            in_column: str,
            value_column: str = 'value',
            ignore_null: bool = True) -> DataFrame:
    """
       unpivot DataFrame.
    .. versionadded:: 0.1.0
    Parameters
    ----------
    df : :class:`DataFrame`
        your DataFrame.
    on_columns : list of str
        array contain name of columns you need to unpivot.
    in_column : str or :class:`Column`
        name of column will contain your unpivot columns.
    value_column : str , optional
        name of column will contain values. (default = 'value')
    ignore_null : bool, optional
        if be True filter the rows with null value.(default = True)
    Examples
    --------
    >>> data = [("Banana", 1000, "USA"), ("Beans", 1600, "USA"), ("Orange", 2000, "USA"),
    >>>         ("Orange", 2000, "USA"), ("Banana", 400, "China"), ("Beans", 1500, "China"),
    >>>         ("Orange", 4000, "China"), ("Banana", 2000, "Canada"), ("Beans", 2000, "Mexico")]
    >>> columns = ["Product", "Amount", "Country"]
    >>> df = spark.createDataFrame(data=data, schema=columns)
    >>> df.show()
    +-------+------+-------+
    |Product|Amount|Country|
    +-------+------+-------+
    | Banana|  1000|    USA|
    |  Beans|  1600|    USA|
    | Orange|  2000|    USA|
    | Orange|  2000|    USA|
    | Banana|   400|  China|
    |  Beans|  1500|  China|
    | Orange|  4000|  China|
    | Banana|  2000| Canada|
    |  Beans|  2000| Mexico|
    +-------+------+-------+
    >>> pivotDF = df.groupBy("Product").pivot("Country").sum("Amount")
    >>> pivotDF.show()
    +-------+------+-----+------+----+
    |Product|Canada|China|Mexico| USA|
    +-------+------+-----+------+----+
    | Orange|  null| 4000|  null|4000|
    |  Beans|  null| 1500|  2000|1600|
    | Banana|  2000|  400|  null|1000|
    +-------+------+-----+------+----+
    >>> unpivotDF = unpivot(df=pivotDF,
    >>>                     on_columns=["USA", "China", "Canada", "Mexico"],
    >>>                     in_column='Country', value_column='Amount')
    >>> unpivotDF.show()
    +-------+-------+------+
    |Product|Country|Amount|
    +-------+-------+------+
    | Orange|    USA|  4000|
    | Orange|  China|  4000|
    |  Beans|    USA|  1600|
    |  Beans|  China|  1500|
    |  Beans| Mexico|  2000|
    | Banana|    USA|  1000|
    | Banana|  China|   400|
    | Banana| Canada|  2000|
    +-------+-------+------+
    """
    columns = df.columns
    unpivot_df = (
        df.select(
            *(list(set(columns) - set(on_columns))),  # select other cols
            F.expr(
                f"""stack({len(on_columns)},{', '.join(map(lambda col: f"'{col}', {col}", on_columns))}) as ({in_column},{value_column})"""
            ))
    )
    if ignore_null:
        return unpivot_df.where(F.col(value_column).isNotNull())
    return unpivot_df
