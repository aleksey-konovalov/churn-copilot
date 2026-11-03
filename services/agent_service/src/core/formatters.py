import json
from shared.constants import FEATURE_MAP, BINARY_FEATURES


def format_feature_value(feature_name, value):
    """Форматирует значение в зависимости от типа фичи"""
    if feature_name in BINARY_FEATURES:
        try:
            return "Да" if int(value) == 1 else "Нет"
        except (ValueError, TypeError):
            return value
    return value

def generate_report(json_data):
    """Парсит JSON с прогнозом оттока и формирует текстовый отчет для LLM"""
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data

    client_id = data.get("client_id", "Неизвестно")
    churn_prob = data.get("churn_probability", 0.0)
    
    # Форматируем вероятность (поддерживает как 0.2051, так и 20.51)
    prob_pct = churn_prob * 100 if churn_prob <= 1.0 else churn_prob
    prob_text = f"{prob_pct:.2f}% (вероятность: {churn_prob})"

    lines = [
        f"ПРОГНОЗ ОТТОКА КЛИЕНТА (ID: {client_id})",
        f"Общая вероятность оттока: {prob_text}",
        "",
        "Основные факторы, влияющие на риск оттока (отсортировано по убыванию):"
    ]

    for i, reason in enumerate(data.get("top_reasons", []), start=1):
        feature = reason.get("feature")
        feature_value = reason.get("feature_value")
        risk_pct = reason.get("risk_contribution_pct", 0.0)

        # Получаем человекопонятное описание или используем оригинальное название, если его нет в словаре
        description = FEATURE_MAP.get(feature, feature.replace("_", " "))
        
        # Форматируем значение
        formatted_value = format_feature_value(feature, feature_value)

        lines.append(
            f"{i}. {description} (текущее значение: {formatted_value}) "
            f"— доля в риске оттока: {risk_pct}%"
        )

    return "\n".join(lines)

def build_complex_query_from_reasons(json_data) -> list[str]:
    """Собирает варианты RAG-запроса из top_reasons: по одному на каждую причину.

    Отдельные варианты (а не одна склеенная строка) нужны механизму голосования
    в search_qdrant: чанк, релевантный нескольким причинам, получает буст VOTE_BOOST.
    """
    query_parts = []
    for reason in json_data.get("top_reasons", []):
        feature = reason.get("feature")
        feature_value = reason.get("feature_value")

        description = FEATURE_MAP.get(feature, feature.replace("_", " "))
        formatted_value = format_feature_value(feature, feature_value)

        query_parts.append(f"{description}: {formatted_value}")

    return query_parts