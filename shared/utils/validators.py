def parse_and_validate_client_id(client_id) -> int:
    """
    Проверяет client_id, что это целое число и конвертирует в int.
    Возвращает int или сообщение о неверном формате.
    """
    # Защита от None и булевых значений
    if client_id is None or isinstance(client_id, bool):
        raise ValueError(f"client_id обязателен и должен быть целым числом, получено: {type(client_id).__name__}")

    try:
        # Сначала преобразуем во float, чтобы корректно обработать строки вида "123.0"
        num = float(client_id)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"client_id должен быть числом, получено: '{client_id}'") from exc

    # Проверяем, является ли число целым
    if not num.is_integer():
        raise ValueError(f"client_id должен быть целым числом, получено дробное: {client_id}")

    # Возвращаем как int
    return int(num)
