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

from pyspark.sql.column import Column
from pyspark.sql.dataframe import DataFrame

import pyspark.sql.functions as F
import bil


def get_cats_with_parents() -> DataFrame:
    """
    Examples
    --------
    >>> df = get_cats_with_parents()
    >>> df.filter(F.col('id').isin([67,68,235])).show()
    +---+--------+------------+-----+----+---------+-----------------+----+---------+-----------------+----+---------+-----------------+
    | id|    name|persian_name|level|cat1|cat1_name|cat1_persian_name|cat2|cat2_name|cat2_persian_name|cat3|cat3_name|cat3_persian_name|
    +---+--------+------------+-----+----+---------+-----------------+----+---------+-----------------+----+---------+-----------------+
    | 67|vehicles|    وسایل نقلیه| cat1|  67| vehicles|      وسایل نقلیه|null|     null|             null|null|     null|             null|
    | 68|    cars|       خودرو| cat2|  67| vehicles|      وسایل نقلیه|  68|     cars|            خودرو| null|     null|             null|
    |235| classic|      کلاسیک| cat3|  67| vehicles|      وسایل نقلیه|  68|     cars|            خودرو | 235|  classic|             کلاسیک|
    +---+--------+------------+-----+----+---------+-----------------+----+---------+-----------------+----+---------+-----------------+
    """
    cats = bil.read.constant.categories.select(
        F.col('old_meta_id').alias('id'),
        F.col('slug').alias('name'),
        F.col('title').alias('persian_name'),
        F.col('parent_slug')
    )
    cat1s = (
        cats
            .filter("parent_slug = 'root'")
            .drop('parent_slug')
            .withColumn('level', F.lit('cat1'))
            .withColumn('cat1', F.col('id'))
            .withColumn('cat1_name', F.col('name'))
            .withColumn('cat1_persian_name', F.col('persian_name'))
    )
    cat2s = (
        cats
            .join(
            cat1s.drop('id', 'name', 'persian_name'),
            F.col('cat1_name') == F.col('parent_slug')
        )
            .drop('parent_slug')
            .withColumn('level', F.lit('cat2'))
            .withColumn('cat2', F.col('id'))
            .withColumn('cat2_name', F.col('name'))
            .withColumn('cat2_persian_name', F.col('persian_name'))
    )
    cat3s = (
        cats
            .join(
            cat2s.drop('id', 'name', 'persian_name'),
            F.col('cat2_name') == F.col('parent_slug')
        )
            .drop('parent_slug')
            .withColumn('level', F.lit('cat3'))
            .withColumn('cat3', F.col('id'))
            .withColumn('cat3_name', F.col('name'))
            .withColumn('cat3_persian_name', F.col('persian_name'))
    )
    all_cats = bil.utils.dataframe.concat(
        cat1s, cat2s, cat3s
    ).select(
        'id', 'name', 'persian_name', 'level',
        'cat1', 'cat1_name', 'cat1_persian_name',
        'cat2', 'cat2_name', 'cat2_persian_name',
        'cat3', 'cat3_name', 'cat3_persian_name',
    )
    return all_cats
