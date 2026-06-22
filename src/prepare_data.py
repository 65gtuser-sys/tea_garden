"""
prepare_data.py
数据读取、清洗与预处理模块
"""

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """读取原始数据"""
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['date'] = df['Timestamp'].dt.date
    df['hour'] = df['Timestamp'].dt.hour
    df['day'] = df['Timestamp'].dt.day
    df['Plant_ID_str'] = df['Plant_ID'].apply(lambda x: f'植株{int(x):02d}')
    df['健康状态'] = df['Plant_Health_Status'].map({
        'Healthy': '健康',
        'Moderate Stress': '中度胁迫',
        'High Stress': '高度胁迫'
    })
    print(f"数据加载完成：{len(df)} 条记录，{df['Plant_ID'].nunique()} 株植物")
    return df


def detect_outliers(df: pd.DataFrame, col: str) -> pd.Series:
    """IQR方法检测异常值，返回布尔Series"""
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    return (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """数据清洗：异常值处理 + 衍生指标计算"""
    df_clean = df.copy()

    # IQR方法处理异常值
    target_cols = [
        'Soil_Moisture', 'Ambient_Temperature',
        'Soil_Temperature', 'Humidity', 'Light_Intensity'
    ]
    for col in target_cols:
        q1, q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        iqr = q3 - q1
        before = detect_outliers(df_clean, col).sum()
        df_clean[col] = df_clean[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        print(f"  {col}：处理异常值 {before} 条")

    # 衍生指标
    df_clean['Temp_Diff'] = (
        df_clean['Ambient_Temperature'] - df_clean['Soil_Temperature']
    )
    df_clean['Nutrient_Index'] = (
        df_clean['Nitrogen_Level'] +
        df_clean['Phosphorus_Level'] +
        df_clean['Potassium_Level']
    ) / 3

    print(f"数据清洗完成，衍生指标：Temp_Diff、Nutrient_Index")
    return df_clean


def get_summary(df: pd.DataFrame) -> None:
    """打印数据基本信息"""
    print(f"\n时间范围：{df['Timestamp'].min()} 至 {df['Timestamp'].max()}")
    print(f"健康状态分布：\n{df['Plant_Health_Status'].value_counts()}")
    print(f"\n各指标描述统计：\n{df.describe().round(2)}")


if __name__ == '__main__':
    df_raw = load_data('../data/plant_health_data.csv')
    df_clean = clean_data(df_raw)
    get_summary(df_clean)
