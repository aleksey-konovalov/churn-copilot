
# перевод технических названий фичей в человекопонятные описания
FEATURE_MAP = {
    "client_id": "ID клиента",
    "is_germany": "Клиент из Германии",
    "is_single_product": "У клиента всего один банковский продукт",
    "is_3plus_products": "У клиента три и более банковских продукта",
    "is_age_over_45": "Возраст клиента более 45 лет",
    "is_zero_balance": "У клиента нулевой баланс",
    "CreditScore": "Кредитный рейтинг клиента",
    "is_high_balance_inactive": "У клиента высокий баланс (более 100_000) и отсутствие активности",
    "Geography": "Страна проживания клиента",
    "Gender": "Пол клиента",
    "Age": "Возраст клиента",
    "Tenure": "Количество лет, которое клиент обслуживается в банке",
    "Balance": "Баланс на счёте клиента",
    "NumOfProducts": "Количество банковских продуктов, которыми пользуется клиент",
    "HasCrCard": "Наличие кредитной карты у клиента",
    "IsActiveMember": "Является ли клиент активным членом банка",
    "EstimatedSalary": "Предполагаемая заработная плата клиента"
}

# Список бинарных флагов, которые нужно переводить в Да/Нет
BINARY_FEATURES = [
    "is_germany", "is_single_product", "is_3plus_products", "is_age_over_45",
    "is_zero_balance", "is_high_balance_inactive", "HasCrCard", "IsActiveMember"
]

# Русские значения категориальных фичей: в CSV они записаны сырьём (Female, France)
FEATURE_VALUE_MAPS = {
    "Gender": {"Female": "Женский", "Male": "Мужской"},
    "Geography": {"France": "Франция", "Spain": "Испания", "Germany": "Германия"},
}
