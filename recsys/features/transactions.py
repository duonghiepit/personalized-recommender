import numpy as np
import pandas as pd
import polars as pl
from hopsworks import udf


def convert_article_id_to_str(df: pl.DataFrame) -> pl.Series:
    """
    Chuyển đổi cột 'article_id' sang kiểu dữ liệu chuỗi (string).

    Tham số:
    - df (pl.DataFrame): DataFrame đầu vào chứa cột 'article_id'.

    Trả về:
    - pl.Series: Cột 'article_id' đã được chuyển đổi sang kiểu chuỗi.
    """

    return df["article_id"].cast(pl.Utf8)

def convert_t_dat_to_datetime(df: pl.DataFrame) -> pl.Series:
    """
    Chuyển đổi cột 't_dat' sang kiểu dữ liệu datetime.

    Tham số:
    - df (pl.DataFrame): DataFrame đầu vào chứa cột 't_dat'.

    Trả về:
    - pl.Series: Cột 't_dat' đã được chuyển đổi sang kiểu datetime.
    """

    return pl.from_pandas(pd.to_datetime(df["t_dat"].to_pandas()))

def get_year_feature(df: pl.DataFrame) -> pl.Series:
    """
    Trích xuất năm từ cột 't_dat'.

    Tham số:
    - df (pl.DataFrame): DataFrame đầu vào chứa cột 't_dat'.

    Trả về:
    - pl.Series: Một Series chứa năm được trích xuất từ 't_dat'.
    """

    return df["t_dat"].dt.year()

def get_month_feature(df: pl.DataFrame) -> pl.Series:
    """
    Trích xuất tháng từ cột 't_dat'.

    Tham số:
    - df (pl.DataFrame): DataFrame đầu vào chứa cột 't_dat'.

    Trả về:
    - pl.Series: Một Series chứa tháng được trích xuất từ 't_dat'.
    """

    return df["t_dat"].dt.month()

def get_day_feature(df: pl.DataFrame) -> pl.Series:
    """
    Trích xuất ngày từ cột 't_dat'.

    Tham số:
    - df (pl.DataFrame): DataFrame đầu vào chứa cột 't_dat'.

    Trả về:
    - pl.Series: Một Series chứa ngày được trích xuất từ 't_dat'.
    """

    return df["t_dat"].dt.day()

def get_day_of_week_feature(df: pl.DataFrame) -> pl.Series:
    """
    Trích xuất ngày trong tuần từ cột 't_dat'.

    Tham số:
    - df (pl.DataFrame): DataFrame đầu vào chứa cột 't_dat'.

    Trả về:
    - pl.Series: Một Series chứa thông tin ngày trong tuần được trích xuất từ 't_dat'.
    """

    return df["t_dat"].dt.weekday()

def calculate_month_sin_cos(month: pl.Series) -> pl.DataFrame:
    """
    Tính giá trị sine và cosine của tháng để nắm bắt các đặc trưng chu kỳ.

    Tham số:
    - month (pl.Series): Một Series chứa các giá trị tháng.

    Trả về:
    - pl.DataFrame: Một DataFrame với các cột 'month_sin' và 'month_cos'.
    """

    C = 2 * np.pi / 12
    return pl.DataFrame(
        {
            "month_sin": month.apply(lambda x: np.sin(x*C)),
            "month_cos": month.apply(lambda x: np.cos(x*C)),
        }
    )

def convert_t_dat_to_epoch_millisecond(df: pl.DataFrame) -> pl.Series:
    """
    Chuyển đổi cột 't_dat' sang đơn vị epoch milliseconds.

    Tham số:
    - df (pl.DataFrame): DataFrame đầu vào chứa cột 't_dat'.

    Trả về:
    - pl.Series: Một Series với 't_dat' đã được chuyển đổi sang epoch milliseconds.
    """

    return df["t_dat"].cast(pl.Int64) // 1_000_000

@udf(return_type = float, mode="pandas")
def month_sin(month: pd.Series):
    """
    Hàm chuyển đổi theo yêu cầu, tính giá trị sine của tháng để mã hóa đặc trưng chu kỳ.

    Tham số:
    - month (pd.Series): Một Series của pandas chứa các giá trị tháng.

    Trả về:
    - pd.Series: Giá trị sine của các tháng.
    """

    return np.sin(month * (2*np.pi/12))

@udf(return_type = float, mode="pandas")
def month_cos(month: pd.Series):
    """
    Hàm chuyển đổi theo yêu cầu, tính giá trị cosine của tháng để mã hóa đặc trưng chu kỳ.

    Tham số:
    - month (pd.Series): Một Series của pandas chứa các giá trị tháng.

    Trả về:
    - pd.Series: Giá trị cosine của các tháng.
    """


    return np.cos(month * (2*np.pi/12))

def compute_features_transactions(df: pl.DataFrame) -> pl.DataFrame:
    """
    Chuẩn bị dữ liệu giao dịch bằng cách thực hiện một số bước chuyển đổi dữ liệu.

    Hàm này thực hiện các bước sau:
    1. Chuyển đổi 'article_id' sang kiểu chuỗi (string).
    2. Chuyển đổi 't_dat' sang kiểu ngày giờ (datetime).
    3. Trích xuất năm, tháng, ngày và ngày trong tuần từ 't_dat'.
    4. Tính giá trị sine và cosine của tháng để mã hóa tính chất chu kỳ.
    5. Chuyển đổi 't_dat' sang đơn vị epoch milliseconds.

    Tham số:
    - df (pl.DataFrame): DataFrame đầu vào chứa dữ liệu giao dịch.

    Trả về:
    - pl.DataFrame: DataFrame đã được xử lý với dữ liệu giao dịch đã được chuyển đổi.
    """

    return (
        df.with_columns(
            [
                pl.col("article_id").cast(pl.Utf8).alias("article_id")
            ]
        )
        .with_columns(
            [
                pl.col("t_dat").dt.year().alias("year"),
                pl.col("t_dat").dt.month().alias("month"),
                pl.col("t_dat").dt.day().alias("day"),
                pl.col("t_dat").dt.weekday().alias("day_of_week"),
            ]
        )
        .with_columns([(pl.col("t_dat").cast(pl.Int64) // 1_000_000).alias("t_dat")])
    )
