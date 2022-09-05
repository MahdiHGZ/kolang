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

from kolang.column import kolang_column_wrapper


@kolang_column_wrapper
def col(col: Union[Column, str]) -> Column:
    """

    .. versionadded:: 1.0.0

    Parameters
    ----------
    col: str or :class:`Column`

    """
    return F.col(col) if isinstance(col, str) else col


str_to_column = col


@kolang_column_wrapper
def percent(col: Union[Column, str] = 'count',
            partition_by: Union[Column, str, List[Union[Column, str]]] = None,
            r: int = 2) -> Column:
    """
        returns the percent of the value.
    .. versionadded:: 0.1.0

    Parameters
    ----------
    col: str or :class:`Column`, optional
        column containing number values (default = 'count')
    partition_by: str or :class:`Column` or list of (str or :class:`Column`), optional
        partition of data.
    r: int, optional
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
    col = str_to_column(col)

    if partition_by is None:
        w = Window.partitionBy()
    else:
        w = Window.partitionBy(partition_by)
    return F.round(100 * col / F.sum(col).over(w), r)


@kolang_column_wrapper
def median(col: str) -> Column:
    """
       Aggregate function: returns the median of the values in a group.
    .. versionadded:: 0.1.0

    Parameters
    ----------
    col: str
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


@kolang_column_wrapper
def str_array_to_array(col: Union[Column, str]) -> Column:
    """
       convert str_array to pysaprk array.
    .. versionadded:: 0.1.0

    Parameters
    ----------
    col: str or :class:`Column`
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


@kolang_column_wrapper
def number_normalizer(col: Union[Column, str]) -> Column:
    """
       normalize numbers in string to en number.
    .. versionadded:: 0.1.0
    Parameters
    ----------
    col: str or :class:`Column`
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


@kolang_column_wrapper
def cumulative_sum(col: Union[Column, str],
                   on_col: Union[Column, str],
                   ascending: bool = True,
                   partition_by: Union[Column, str, List[Union[Column, str]]] = None) -> Column:
    """
       normalize numbers in string to en number.
    .. versionadded:: 0.1.0
    Parameters
    ----------
    col: str or :class:`Column`
        column containing string.
    on_col: str or :class:`Column`
        order data base on this column.
    ascending: str or :class:`Column`, optional
        type of ordering is ascending. (default = True)
    partition_by: str or :class:`Column` or list of (str or :class:`Column`), optional
        partition of data.
    Examples
    --------
    >>> df = spark.range(0, 5).toDF('id').withColumn('value', F.lit(3))
    >>> df = df.withColumn('cumulative_sum', cumulative_sum('value','id'))
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


@kolang_column_wrapper
def text_cleaner(col: Union[Column, str],
                 accept: str = "") -> Column:
    """
       clean text from emoji and other symbols.(just accept numbers and english and persian letters)
    .. versionadded:: 0.1.0
    Parameters
    ----------
    col: str or :class:`Column`
        column containing string.
    accept: str, optional
        string containing char you want to accept. (default = "")

    Examples
    --------
    >>> df = spark.createDataFrame([("sjkdf sdk❤️❤️fskd j",),
    >>>                             ("۷7۲ 67 gh^&g    df",),
    >>>                             ("۱a%%!. سلام ab😂😂8()",),],
    >>>                             ['string'])
    >>> df = df.withColumn('clean_str', text_cleaner('string'))
    >>> df.show()
    +--------------------+----------------+
    |              string|       clean_str|
    +--------------------+----------------+
    | sjkdf sdk❤️❤️fskd j|sjkdf sdk fskd j|
    |  ۷7۲ 67 gh^&g    df|  772 67 gh g df|
    |۱a%%!. سلام ab😂?...|   1a سلام ab 8 |
    +--------------------+----------------+
    """
    col = number_normalizer(col)
    col = F.translate(col, 'كيأإؤةۀ', 'کیااوهه')
    col = F.regexp_replace(col, f"[^a-zآ-یA-Z0-9 {accept}]", " ")
    col = F.regexp_replace(col, " {2,}", " ")
    return col


@kolang_column_wrapper
def bin(col: Union[Column, str],
        scale: int = 10,
        flooring: bool = True) -> Column:
    """

    .. versionadded:: 0.2.0
    Parameters
    ----------
    col: str or :class:`Column`
        column containing number.
    scale: int, optional
        bin size. (default = 10)
    flooring: bool, optional
        if True uses floor else uses round. (default = True)

    """
    col = str_to_column(col)

    if flooring:
        return F.floor(col / scale) * scale
    else:
        return F.round(col / scale, 0) * scale


@kolang_column_wrapper
def session_id(device_id: Union[Column, str] = 'device_id',
               created_at: Union[Column, str] = 'created_at',
               session_time: int = 30,
               ) -> Column:
    """
        returns session_id for actions.
    .. versionadded:: 0.2.0
    Parameters
    ----------
    device_id: str or :class:`Column`, optional
        column containing user unique key(default = 'device_id')
    created_at: str or :class:`Column`, optional
        column containing long int time.(millisecond)(default = 'created_at')
    session_time: int, optional
        max time of session (minute)(default = 30)

    """
    session_time = session_time * 60000

    device_id = str_to_column(device_id)
    created_at = str_to_column(created_at)

    session_window = Window.partitionBy(device_id).orderBy(created_at)

    col = F.lag(created_at).over(session_window)
    col = F.when((created_at - col > session_time) | col.isNull(), F.monotonically_increasing_id())
    col = F.when(col.isNull(), F.last(col, ignorenulls=True).over(session_window)).otherwise(col)

    return col


@kolang_column_wrapper
def cond_count(cond: Union[Column, str]) -> Column:
    """
        Aggregate function: returns count of rows how accept the condition.
    .. versionadded:: 0.3.0
    Parameters
    ----------
    cond: str or :class:`Column`
        your condition.
    """
    if isinstance(cond, str):
        cond = F.expr(cond)
    return F.count(F.when(cond, True))


condition_count = cond_count


@kolang_column_wrapper
def persian_number(col: Union[Column, str],
                   format: str = '%d') -> Column:
    """
        convert english number to persian number(string)
    .. versionadded:: 0.3.0
    Parameters
    ----------
    col: str or :class:`Column`
        column containing english number.
    format: str, optional
        your final result format. (default = '%d')
    """
    col = F.format_string(format, col)
    col = F.translate(col, "0123456789", "۰۱۲۳۴۵۶۷۸۹")
    return col


@kolang_column_wrapper
def jalali_date(col: Union[Column, str],
                format: str = '%Y-%m-%d') -> Column:
    """
        convert gregorian date to jalali date.
    .. versionadded:: 0.3.0
    Parameters
    ----------
    col: str or :class:`Column`
        column containing gregorian date.
    format: str, optional
        your final result format. (default = '%Y-%m-%d')
        `%d`: Day of the month (29)
        `%fd`: Day of the month with Persian number (۲۹)
        `%m`: Month (03)
        `%fm`: Month with Persian number (۰۳)
        `%y`: Year without century (00)
        `%fy`: Year without century with Persian number (۰۰)
        `%Y`: Year with century (1400)
        `%fY`: Year with century and Persian number (۱۴۰۰)
        `%A`:  Weekday (شنبه)
        `%B`: Month name (خرداد)
        `%C`: Season name (بهار)
    Examples
    --------
    >>> df = (
    ...     spark.createDataFrame([('2021-08-12',)], ['date'])
    ...         .withColumn('j-date-sample1', jalali_date('date'))
    ...         .withColumn('j-date-sample2', jalali_date('date', '%Y-%m'))
    ...         .withColumn('j-date-sample3', jalali_date('date', 'month:%m,day:%d'))
    ... )
    >>> df.show()
    +----------+--------------+--------------+---------------+
    |      date|j-date-sample1|j-date-sample2| j-date-sample3|
    +----------+--------------+--------------+---------------+
    |2021-08-12|    1400-05-21|       1400-05|month:05,day:21|
    +----------+--------------+--------------+---------------+
    """
    j_days_in_month_cum = [0, 31, 62, 93, 124, 155, 186, 216, 246, 276, 306, 336]

    j_month_name = [
        'فروردین',
        'اردیبهشت',
        'خرداد',
        'تیر',
        'مرداد',
        'شهریور',
        'مهر',
        'آبان',
        'آذر',
        'دی',
        'بهمن',
        'اسفند'
    ]

    j_week_name = [
        'شنبه',
        'یکشنبه',
        'دوشنبه',
        'سه شنبه',
        'چهارشنبه',
        'پنجشنبه',
        'جمعه',
    ]

    j_season_name = [
        'بهار',
        'تابستان',
        'پاییز',
        'زمستان',
    ]

    gy = F.year(col) - 1600
    gdy = F.dayofyear(col) - 1
    gwd = F.dayofweek(col)

    g_day_no = 365 * gy + F.floor((gy + 3) / 4) - F.floor((gy + 99) / 100) + F.floor((gy + 399) / 400)
    g_day_no = g_day_no + gdy

    j_day_no = g_day_no - 79

    j_np = F.floor(j_day_no / 12053)
    j_day_no = j_day_no % 12053
    jy = 979 + 33 * j_np + 4 * F.floor(j_day_no / 1461)

    j_day_no = j_day_no % 1461

    jy = F.when(j_day_no >= 366, jy + F.floor((j_day_no - 1) / 365)).otherwise(jy).alias('jy')
    j_day_no = F.when(j_day_no >= 366, (j_day_no - 1) % 365).otherwise(j_day_no).alias('y')

    jm = F.when(j_day_no < j_days_in_month_cum[1], 1)
    for i in range(2, 12):
        jm = jm.when(j_day_no < j_days_in_month_cum[i], i)
    jm = jm.otherwise(12)

    j_days_in_month_cum = F.array([F.lit(element) for element in j_days_in_month_cum])
    jd = j_day_no - j_days_in_month_cum[jm - 1] + 1

    import re
    var_map = {
        '%d': jd,
        '%m': jm,
        '%y': (jy % 100),
        '%Y': jy,
        '%B': F.array([F.lit(element) for element in j_month_name])[jm - 1],
        '%A': F.array([F.lit(element) for element in j_week_name])[gwd % 7],
        '%C': F.array([F.lit(element) for element in j_season_name])[F.floor((jm - 1) / 3)],
        '%fd': persian_number(jd, '%02d'),
        '%fm': persian_number(jm, '%02d'),
        '%fy': persian_number((jy % 100), '%02d'),
        '%fY': persian_number(jy, '%d'),
    }
    str_map = {
        '%d': '%02d',
        '%m': '%02d',
        '%y': '%02d',
        '%Y': '%d',
        '%B': '%s',
        '%A': '%s',
        '%C': '%s',
        '%fd': '%s',
        '%fm': '%s',
        '%fy': '%s',
        '%fY': '%s',
    }

    vars = re.findall('|'.join(var_map.keys()), format)
    vars = list(map(lambda v: var_map[v], vars))

    for key, var in str_map.items():
        format = format.replace(key, var)

    return F.format_string(format, *vars)


@kolang_column_wrapper
def sum_columns(cols: List[Union[Column, str]]) -> Column:
    """
        returns sum of your columns.
    .. versionadded:: 0.4.0
    Parameters
    ----------
    cols: list of (str or :class:`Column`)
        list of columns you want to sum.
    Examples
    --------
    >>> df = (spark.range(0, 5).toDF('a')
    ...         .withColumn('b', F.lit(3))
    ...         .withColumn('c', F.col('a')*2)
    ...         .withColumn('d', F.lit(123))
    ...         .withColumn('e', F.col('a') + 2)
    ...         .withColumn('f', F.col('a')*3 + 2)
    ...         .withColumn('g', F.lit(12))
    ...         .withColumn('h', F.lit(-100))
    ...         .withColumn('i', F.col('a') * -12)
    ...         )
    >>> df = df.withColumn('sum',sum_columns(['a',F.col('b'),'c','d','e','f','g','h','i']))
    >>> df.show()
    +---+---+---+---+---+---+---+----+---+---+
    |  a|  b|  c|  d|  e|  f|  g|   h|  i|sum|
    +---+---+---+---+---+---+---+----+---+---+
    |  0|  3|  0|123|  2|  2| 12|-100|  0| 42|
    |  1|  3|  2|123|  3|  5| 12|-100|-12| 37|
    |  2|  3|  4|123|  4|  8| 12|-100|-24| 32|
    |  3|  3|  6|123|  5| 11| 12|-100|-36| 27|
    |  4|  3|  8|123|  6| 14| 12|-100|-48| 22|
    +---+---+---+---+---+---+---+----+---+---+
    """
    cols = list(map(str_to_column, cols))
    res = F.lit(0)
    for col in cols:
        res = res + col
    return res


@kolang_column_wrapper
def array_contains_column(col: Union[Column, str],
                          array_col: Union[Column, str]) -> Column:
    """

    .. versionadded:: 1.0.0
    Parameters
    ----------
    col: str or :class:`Column`
    array_col: str or :class:`Column`
        column containing array.

    """
    col = str_to_column(col)
    return F.size(F.array_intersect(array_col, F.array([col]))) >= 1


@kolang_column_wrapper
def cumulative_percent(col: Union[Column, str],
                       on_col: Union[Column, str],
                       ascending: bool = True,
                       partition_by: Union[Column, str, List[Union[Column, str]]] = None,
                       r: int = 2) -> Column:
    """
        return the percent of cumulative sum of data.
    .. versionadded:: 1.0.0
    Parameters
    ----------
    col: str or :class:`Column`
        column containing string.
    on_col: str or :class:`Column`
        order data base on this column.
    ascending: str or :class:`Column`, optional
        type of ordering is ascending. (default = True)
    partition_by: str or :class:`Column` or list of (str or :class:`Column`)
    r: int, optional
        rounding a final result base on this (default = 2)
    """
    col = str_to_column(col)
    on_col = str_to_column(on_col)

    on_col = on_col if ascending else F.desc(on_col)

    if partition_by is None:
        w_sum = Window.orderBy(on_col)
        w_percent = w = Window.partitionBy()
    else:
        w_sum = Window.partitionBy(partition_by).orderBy(on_col)
        w_percent = Window.partitionBy(partition_by)

    return F.round(100 * F.sum(col).over(w_sum) / F.sum(col).over(w_percent), r)
