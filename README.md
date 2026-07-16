# ChurnCoPilot

> **Соединяем ML-предсказания и бизнес-действия через объяснимый ИИ.**

## 📜 Описание проекта
**ChurnCoPilot** — это гибридная система, объединяющая классическое табличное машинное обучение и LLM-агента для интеллектуального анализа оттока клиентов.

Вместо того чтобы просто выдавать "черный ящик" с вероятностью оттока, система объясняет бизнесу *причины* этого прогноза и дает рекомендации по удержанию клиентов в соответствии с политиками компании:
1. Модель **CatBoost** анализирует табличный датасет и предсказывает вероятность оттока для каждого клиента.
2. **LLM-агент** получает доступ к предсказаниям модели, значениям **SHAP** и политикам компании через специальные инструменты (tools).
3. Когда менеджер задает вопрос: *"Почему уйдет клиент X?"*, агент автоматически вызывает нужные tool'ы и генерирует осмысленный, понятный человеку ответ на естественном языке.

## ⚙️ Как это работает
- **ML-модель**: CatBoost (для высокой точности и встроенной поддержки TreeSHAP).
- **Интерпретируемость**: SHAP values (глобальная и локальная интерпретация важности признаков).
- **Агент**: LLM с функцией вызова инструментов (Tool Calling / Function Calling) для безопасного и структурированного доступа к данным.

---

## 📊 Data Source
The churn dataset is sourced from the following open Kaggle competition:  
🔗 [Bank Churn Competition by IPII HS EX MTS](https://www.kaggle.com/competitions/bank-churn-competition-by-ipii-hs-ex-mts/data)
#### ⚖️ License
This dataset are distributed under the:  
🔗 [Creative Commons Attribution 4.0 International (CC BY 4.0) License](https://creativecommons.org/licenses/by/4.0/)

**Note on modifications:** The original train set was structurally split into separate training and validation files for model development purposes. The underlying data records and values themselves have not been altered, modified, or augmented.

---

## License (Code)
The source code of this project is open-sourced under the **Apache License 2.0**.  
See the [LICENSE](LICENSE) file for details.

*Note: The underlying dataset used in this project is governed by its own license (CC BY 4.0), as specified in the "Data Source" section above. This Apache 2.0 license applies strictly to the code, scripts, and model architecture developed for this project.*

---