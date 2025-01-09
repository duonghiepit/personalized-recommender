import random

import polars as pl
from loguru import logger

from recsys.config import CustomerDatasetSize


class DatasetSampler:
    _SIZES = {
        CustomerDatasetSize.LARGE: 50_000,
        CustomerDatasetSize.MEDIUM: 5_000,
        CustomerDatasetSize.SMALL: 1_000,
    }

    def __init__(self, size: CustomerDatasetSize) -> None:
        self._size = size

    @classmethod
    def get_supported_sizes(cls) -> dict:
        return cls._SIZES
    
    def sample(
            self, customers_df: pl.DataFrame, transaction_df: pl.DataFrame
    ) -> dict[str, pl.DataFrame]:
        random.seed(42)

        n_customers = self._SIZES[self._size]
        logger.info(f"Sampling {n_customers} customers.")
        customers_df = customers_df.sample(n=n_customers)

        logger.info(
            f"Number of transactions for all the customers: {transaction_df.height}"
        )
        transaction_df = transaction_df.join(
            customers_df.select("customer_id"), on="customer_id"
        )
        logger.info(
            f"Number of transactions for the {n_customers} sampled customers: {transaction_df.height}"
        )

        return {"customers": customers_df, "transactions": transaction_df}
    
def filling_missing_club_member_status(df: pl.DataFrame) -> pl.DataFrame:
    """
    Điền các giá trị thiếu trong cột 'club_member_status' bằng 'ABSENT'.

    Tham số:
    - df (pl.DataFrame): DataFrame đầu vào chứa cột 'club_member_status'.

    Trả về:
    - pl.DataFrame: DataFrame với cột 'club_member_status' đã được điền giá trị.
    """

    return df.with_columns(pl.col("club_member_status").fill_null("ABSENT"))

def drop_na_age(df: pl.DataFrame) -> pl.DataFrame:
    """
    Loại bỏ các hàng có giá trị null trong cột 'age'.

    Tham số:
    - df (pl.DataFrame): DataFrame đầu vào chứa cột 'age'.

    Trả về:
    - pl.DataFrame: DataFrame đã loại bỏ các hàng có giá trị null trong cột 'age'.
    """

    return df.drop_nulls(subset=['age'])

def create_age_group() -> pl.Expr:
    """
    Tạo một biểu thức để phân loại độ tuổi vào các nhóm.

    Trả về:
    - pl.Expr: Biểu thức Polars phân loại 'age' vào các nhóm độ tuổi đã được định nghĩa trước.
    """
    
    return (
        pl.when(pl.col('age').is_between(0, 18))
        .then(pl.lit("0-18"))
        .when(pl.col('age').is_between(19, 25))
        .then(pl.lit("19-25"))
        .when(pl.col('age').is_between(26, 35))
        .then(pl.lit("26-35"))
        .when(pl.col('age').is_between(36, 45))
        .then(pl.lit("36-45"))
        .when(pl.col('age').is_between(46, 55))
        .then(pl.lit("46-55"))
        .when(pl.col('age').is_between(56, 65))
        .then(pl.lit("56-65"))
        .otherwise(pl.lit("66+"))
    ).alias("age_group")

def compute_features_customers(
        df: pl.DataFrame, drop_null_age: bool = False
) -> pl.DataFrame:
    """
    Chuẩn bị dữ liệu khách hàng bằng cách thực hiện một số bước làm sạch và chuyển đổi dữ liệu.

    Hàm này thực hiện các bước sau:
    1. Kiểm tra các cột cần thiết trong DataFrame đầu vào.
    2. Điền giá trị thiếu trong cột trạng thái thành viên câu lạc bộ bằng 'ABSENT'.
    3. Loại bỏ các hàng có giá trị tuổi bị thiếu.
    4. Tạo một nhóm độ tuổi.
    5. Chuyển đổi cột 'age' sang kiểu dữ liệu Float64.
    6. Chọn và sắp xếp các cột cụ thể trong kết quả đầu ra.

    Tham số:
    - df (pl.DataFrame): DataFrame đầu vào chứa dữ liệu khách hàng.

    Trả về:
    - pl.DataFrame: DataFrame đã được xử lý với dữ liệu khách hàng đã được làm sạch và chuyển đổi.

    Ngoại lệ:
    - ValueError: Nếu bất kỳ cột cần thiết nào bị thiếu trong DataFrame đầu vào.
    """
    required_columns = ["customer_id", "club_member_status", "age", "postal_code"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Columns {', '.join(missing_columns)} not found in the DataFrame"
        )
    
    df = (
        df.pipe(filling_missing_club_member_status)
        .pipe(drop_na_age)
        .with_columns([create_age_group(), pl.col("age").cast(pl.Float64)])
        .select(
            ["customer_id", "club_member_status", "age", "postal_code", "age_group"]
        )
    )

    if drop_null_age is True:
        df = df.drop_nulls(subset=["age"])

    return df
