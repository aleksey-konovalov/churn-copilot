"""
Функции очистки данных и создания признаков на основе EDA.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Загружает данные и выполняет базовую очистку.
    
    Args:
        filepath: Путь к CSV файлу
        
    Returns:
        Очищенный DataFrame
    """
    df = pd.read_csv(filepath)
    
    # Удаляем идентификаторы и ненужные колонки
    cols_to_drop = ['id', 'CustomerId', 'Surname']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    return df


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует типы данных для оптимизации памяти и корректной работы модели.
    """
    # Столбцы, которые должны быть целыми числами
    int_cols = ['CreditScore', 'Age', 'Tenure', 'NumOfProducts', 
                'HasCrCard', 'IsActiveMember', 'Exited']
    
    for col in int_cols:
        if col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].astype('Int64')
            else:
                df[col] = df[col].astype(int)
    
    return df


def create_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создает бинарные признаки на основе EDA.
    """
    df = df.copy()
    
    # География: Германия имеет аномально высокий отток
    df['is_germany'] = (df['Geography'] == 'Germany').astype(int)
    
    # Количество продуктов: нелинейная зависимость
    df['is_single_product'] = ((df['NumOfProducts'] == 1).fillna(False)).astype(int)
    df['is_3plus_products'] = ((df['NumOfProducts'] >= 3).fillna(False)).astype(int)
    
    # Возраст: резкий рост оттока после 45 лет
    df['is_age_over_45'] = ((df['Age'] > 45).fillna(False)).astype(int)
    
    # Баланс: бимодальное распределение
    df['is_zero_balance'] = ((df['Balance'] <= 10).fillna(False)).astype(int)
    
    # Комбинация высокий баланс + неактивен
    df['is_high_balance_inactive'] = (
        (df['Balance'] > 100_000).fillna(False) & 
        (df['IsActiveMember'] == 0).fillna(False)
    ).astype(int)
    
    return df


def prepare_features(filepath: str, target_col: str = 'Exited') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Полный pipeline подготовки данных: загрузка, очистка, создание признаков.
    
    Args:
        filepath: Путь к CSV файлу
        target_col: Название целевой переменной
        
    Returns:
        X: DataFrame с признаками
        y: Series с целевой переменной
    """
    # Загрузка и очистка
    df = load_and_clean_data(filepath)
    
    # Преобразование типов
    df = convert_dtypes(df)
    
    # Создание бинарных признаков
    df = create_binary_features(df)
    
    # Разделение на признаки и таргет
    if target_col in df.columns:
        # Обучающие / валидационные данные (есть таргет)
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        # Тестовые данные (нет таргета)
        X = df
        y = None
        print(f"Столбец '{target_col}' не найден. Возвращаем только X.")
        
    return X, y


def get_categorical_features() -> list:
    """
    Возвращает список категориальных признаков.
    """
    return ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']


def get_feature_importance_columns() -> list:
    """
    Возвращает список всех признаков (включая созданные) для анализа важности.
    """
    return [
        # Исходные признаки
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        # Созданные бинарные признаки
        'is_germany', 'is_single_product', 'is_3plus_products',
        'is_age_over_45', 'is_zero_balance', 'is_high_balance_inactive'
    ]
