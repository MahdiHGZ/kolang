import datetime
import itertools
import os
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Union, ValuesView, cast, overload)

import pandas as pd
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame


def unpivot(df: DataFrame,
            on_columns: List[str],
            in_column: str,
            value_column: str = 'value',
            ignore_null: bool = True) -> DataFrame:
    """
        Unpivot DataFrame.

    .. versionadded:: 0.1.0
    Parameters
    ----------
    df: :class:`DataFrame`
        your DataFrame.
    on_columns: list of str
        array contain name of columns you need to unpivot (Dont use number as column name!).
    in_column: str or :class:`Column`
        name of column will contain your unpivot columns.
    value_column: str , optional
        name of column will contain values. (default = 'value')
    ignore_null: bool, optional
        if be True filter the rows with null value. (default = True)
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
        Transpose your DataFrame.
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


def union_all(*dfs: Union[DataFrame, List[DataFrame]],
              force: bool = False,
              ) -> DataFrame:
    """
        Union your dataframes with different columns.

    .. versionadded:: 1.2.0
    Parameters
    ----------
    dfs: :class:`DataFrame` or list of :class:`DataFrame`
        your dataframes.
    force: bool, optional (default = False)
        if be True, it will force to union dataframes with handeling incompatible columns types.
    Examples
    --------
    >>> df1 = spark.createDataFrame([(1, "foo", 4), (2, "bar", 4), ], ["col1", "col2", "col4"])
    >>> df2 = spark.createDataFrame([(3, "foo", "6"), (4, "bar", "4"), ], ["col1", "col3", "col4"])
    >>> df = union_all(df1, df2)
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
    if len(dfs) == 1 and isinstance(dfs[0], list):
        dfs = dfs[0]
    if len(dfs) == 1:
        return dfs[0]
    if len(dfs) > 2:
        return union_all(dfs[0], union_all(*dfs[1:], force=force), force=force)
    df1, df2 = dfs[0], dfs[1]
    columns1 = set(df1.columns)
    columns2 = set(df2.columns)
    df1 = df1.select(*columns1, *[F.lit(None).alias(col) for col in (columns2 - columns1)])
    df2 = df2.select(*columns2, *[F.lit(None).alias(col) for col in (columns1 - columns2)])

    try:
        return df1.unionByName(df2)
    except pyspark.sql.utils.AnalysisException as e:
        if not force:
            raise e
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


def safe_union(*dfs: Union[DataFrame, List[DataFrame]]) -> DataFrame:
    """
        this function will union your all dataframes with handeling incompatible columns types.

    .. versionadded:: 0.4.0
    Parameters
    ----------
    dfs: :class:`DataFrame`
        your dataframes.
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
    return union_all(*dfs, force=True)


def load_or_calculate_parquet(
        func: Callable,
        path: str,
        range_params: Dict[str, List[Any]] = {},
        constant_params: Dict[str, Any] = {},
        overwrite: bool = False,
        partition_size: int = 1,
        log: bool = True,
        error: str = 'ignore') -> DataFrame:
    """
        Run your function with your all given params and parquet result when parquet not exist.
        Eventually, it returns the all dataframe parked at the path.

    .. versionadded:: 1.0.0
    Parameters
    ----------
    func: Callable
        Your function that returns a dataframe.
    path: str
        Your directory for parquet.
    range_params: dict, optional
        dictionary for your function params you want function run with them. (default = {})
    constant_params: dict, optional
        dictionary for your function params that are fixed. (default = {})
    overwrite: bool, optional
        when is True run function for existing values and overwrte parquet. (default = False)
    partition_size: int, optional
        The size of each parquet partition. (default = 1)
    log: bool, optional
        when is True print log.  (default = True)
    error: str, optional
        how reacte to error. (ignore, stop) (default = 'ignore')

    Examples
    --------
    >>> def calculate_new_user(ds,type):
    >>>     users = spark.range(0, random.randint(2, 150),3).toDF('id').withColumn('type',F.col('id')%2==0)
    >>>     users_count = users.groupBy('type').count()
    >>>     users_count = users_count.withColumn('percent',column_functions.percent()).filter(F.col('type')==type)
    >>>     return users_count
    >>>
    >>> df = calculate_new_user('2022-09-10',True)
    >>> df.show()
    +----+-----+-------+
    |type|count|percent|
    +----+-----+-------+
    |true|   24|  51.06|
    +----+-----+-------+
    >>> df = load_or_calculate_parquet(
    ...         func=calculate_new_user,
    ...         path='/my_directory',
    ...         range_params={'ds':['2022-09-03','2022-09-05'],
    ...                       'type':[True,False]})
    >>> df.show()
    calculate {'ds': '2022-09-03', 'type': True}
    calculate {'ds': '2022-09-03', 'type': False}
    load {'ds': '2022-09-05', 'type': True}
    load {'ds': '2022-09-05', 'type': False}
    +-----+-----+-------+----------+
    | type|count|percent|        ds|
    +-----+-----+-------+----------+
    | True|    7|   50.0|2022-09-05|
    | True|    4|   50.0|2022-09-03|
    |False|   17|   50.0|2022-09-05|
    |False|   10|  47.62|2022-09-03|
    +-----+-----+-------+----------+
    """
    spark = SparkSession.builder.getOrCreate()

    def logger(*args):
        if log:
            print(*args)

    def make_products():
        range_keys = []
        range_vals = []

        for key, val in range_params.items():
            range_keys.append(key)
            range_vals.append(list(val))

        range_products = list(itertools.product(*range_vals))
        return list(map(lambda x: dict(zip(range_keys, x)), range_products))

    def make_product_path(product):
        return os.path.join(path, '/'.join(map(lambda x: f'{x[0]}={x[1]}', list(product.items()))))

    def load_product(product):
        spark.read.parquet(make_product_path(product))
        logger('load', product)

    def calculate_product(product):
        try:
            params = {**product, **constant_params}
            df = func(**params)
            df.repartition(partition_size).write.parquet(make_product_path(product), mode="overwrite")
            logger('calculate', product)
        except Exception as e:
            logger('error on calculate', product)
            if error == 'ignore':
                logger(e)
            elif error == 'stop':
                raise e

    products = make_products()

    for product in products:
        if overwrite:
            calculate_product(product)
        else:
            try:
                load_product(product)
            except pyspark.sql.utils.AnalysisException:
                calculate_product(product)

    df = spark.read.parquet(path)
    return df


def add_trend_line(
        df: DataFrame,
        value_col: Union[str, List[str]],
        date_col: str = 'date',
        prediction_day: int = 0,
        degree: int = 1,
        cache: bool = True
) -> DataFrame:
    """
    Add trend line to date base data.
    .. versionadded:: 1.3.0
    Parameters
    ----------
    df: DataFrame
        Your dataframe.
    value_col: str or list of str
        Your value column or columns name.
    date_col: str, optional
        Your date column. (default = 'date')
    prediction_day: int, optional
        If prediction_day is greater than 0, it will predict the next days. (default = 0)
    degree: int, optional
         degree of a polynomial trend line. this must be greater than 0. (default = 1)
    cache: bool, optional
        When is True, it will cache the dataframe. (default = True)
"""
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import LinearRegression

    spark = SparkSession.builder.getOrCreate()

    if df.schema[date_col].dataType.simpleString() not in ['date', 'timestamp']:
        raise ValueError(f'{date_col} must be date or timestamp')

    value_cols = [value_col] if isinstance(value_col, str) else value_col
    regression_feature_col = 'regression_feature_col'
    regression_features_list = [(F.col(regression_feature_col) ** i).alias(f'{regression_feature_col}_d{i}') for i in
                                range(1, degree + 1)]
    regression_features_str_list = [f'{regression_feature_col}_d{i}' for i in range(1, degree + 1)]
    regression_features_vector = 'regression_features_vector'

    if cache:
        df = df.cache()

    base_df = df = df.withColumn(date_col, F.col(date_col).cast('date'))
    start_date, end_date = df.agg(F.min(date_col), F.max(date_col)).collect()[0]
    regression_feature_function = F.datediff(date_col, F.lit(start_date).cast('date'))

    df = (df
          .withColumn(regression_feature_col, regression_feature_function)
          .select('*', *regression_features_list)
          )

    feature_assembler = VectorAssembler(inputCols=regression_features_str_list, outputCol=regression_features_vector)
    df = feature_assembler.transform(df).drop(regression_feature_col, *regression_features_str_list)

    regressor = dict()
    for value_col in value_cols:
        regressor[value_col] = LinearRegression(featuresCol=regression_features_vector,
                                                labelCol=value_col,
                                                predictionCol=f'{value_col}_trendline'
                                                ).fit(df)

    prediction_df = (
        spark.createDataFrame(
            pd.DataFrame(pd.date_range(start_date, end_date + datetime.timedelta(days=prediction_day)),
                         columns=[date_col])
        )
        .withColumn(date_col, F.to_date(date_col))
        .withColumn(regression_feature_col, regression_feature_function)
        .select('*', *regression_features_list)

    )
    prediction_df = (feature_assembler
                     .transform(prediction_df)
                     .drop(regression_feature_col, *regression_features_str_list)
                     )

    df = df.join(prediction_df, on=[date_col, regression_features_vector], how='full').fillna(0, value_cols)

    for value_col in value_cols:
        df = regressor[value_col].evaluate(df).predictions

    result_df = (df
                 .drop(regression_feature_col, regression_features_vector, *value_cols)
                 .join(base_df.select(date_col, *value_cols), on=date_col, how='left')
                 )
    return result_df
