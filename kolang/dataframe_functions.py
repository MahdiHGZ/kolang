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

import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.column import Column
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession

import pandas as pd


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
    df: :class:`DataFrame`
        your DataFrame.
    on_columns: list of str
        array contain name of columns you need to unpivot (Dont use number as column name).
    in_column: str or :class:`Column`
        name of column will contain your unpivot columns.
    value_column: str , optional
        name of column will contain values. (default = 'value')
    ignore_null: bool, optional
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


def pandas_to_spark(df: pd.DataFrame) -> DataFrame:
    """
        Given pandas dataframe, it will return a spark's dataframe.
    .. versionadded:: 0.3.0
    Parameters
    ----------
    df: :class:`Pandas DataFrame`
    """
    spark = SparkSession.builder.getOrCreate()
    try:
        return spark.createDataFrame(df)
    except:
        def equivalent_type(type: str):
            if type == 'datetime64[ns]':
                return T.TimestampType()
            elif type == 'int64':
                return T.LongType()
            elif type == 'int32':
                return T.IntegerType()
            elif type == 'float64':
                return T.FloatType()
            else:
                return T.StringType()

        def define_structure(string, format_type):
            try:
                typo = equivalent_type(format_type)
            except:
                typo = T.StringType()
            return T.StructField(string, typo)

        columns = list(df.columns)
        types = list(df.dtypes)
        struct_list = []
        for column, typo in zip(columns, types):
            struct_list.append(define_structure(column, typo))
        p_schema = T.StructType(struct_list)
        return spark.createDataFrame(df, p_schema)


def transpose(df: DataFrame,
              col: str) -> DataFrame:
    """
        transpose your DataFrame.
        Warnings: Dont use it for big DataFrames!!
    .. versionadded:: 0.3.0
    Parameters
    ----------
    df: :class:`DataFrame`
        your dataframe.
    col: str
        name of col you want transpose base on it.
    """
    pandas_df = df.toPandas().set_index(col).transpose().reset_index(level=0, inplace=False)
    return pandas_to_spark(pandas_df)


def safe_union(df1: DataFrame, df2: DataFrame) -> DataFrame:
    """
        union your dataframes with different columns.
    .. versionadded:: 0.4.0
    Parameters
    ----------
    df1: :class:`DataFrame`
        your first dataframe.
    df2: :class:`DataFrame`
        your secend dataframe.
    Examples
    --------
    >>> df1 = spark.createDataFrame([(1, "foo", 4), (2, "bar", 4), ], ["col1", "col2", "col4"])
    >>> df2 = spark.createDataFrame([(3, "foo", "6"), (4, "bar", "4"), ], ["col1", "col3", "col4"])
    >>> df = safe_union(df1, df2)
    >>> df.show()
    +----+----+----+----+
    |col1|col4|col2|col3|
    +----+----+----+----+
    |   1|   4| foo|null|
    |   2|   4| bar|null|
    |   3|   6|null| foo|
    |   4|   4|null| bar|
    +----+----+----+----+
    >>> df.printSchema()
    root
     |-- col1: long (nullable = true)
     |-- col4: string (nullable = true)
     |-- col2: string (nullable = true)
     |-- col3: string (nullable = true)
    """
    columns1 = set(df1.columns)
    columns2 = set(df2.columns)
    df1 = df1.select(*columns1, *[F.lit(None).alias(col) for col in (columns2 - columns1)])
    df2 = df2.select(*columns2, *[F.lit(None).alias(col) for col in (columns1 - columns2)])

    try:
        return df1.unionByName(df2)
    except pyspark.sql.utils.AnalysisException as e:
        common_columns = columns1.intersection(columns2)
        incompatible_data_types = {
            ('boolean', 'string'): 'string',
            ('array<string>', 'string'): 'string',
        }
        incompatible_columns = list()
        for col in common_columns:
            for key, val in incompatible_data_types.items():
                if {df1.schema[col].dataType.simpleString(), df2.schema[col].dataType.simpleString()} == set(key):
                    df1 = df1.withColumn(col, F.col(col).cast(val))
                    df2 = df2.withColumn(col, F.col(col).cast(val))
                    incompatible_columns.append(col)
        if len(incompatible_columns) > 0:
            print(
                f"Warning: safe_union function handle incompatible data types of this columns: {', '.join(incompatible_columns)}")
        return df1.unionByName(df2)
