# 🚀 TradingView Crypto Portfolio Management System

Потужна система управління криптовалютними портфелями з інтеграцією TradingView для отримання real-time даних та аналітики.

## 📋 Огляд Системи

Ця система поєднує:
- **TradingView Data Scraping** - Отримання real-time даних з TradingView
- **Portfolio Management** - Управління криптовалютними портфелями
- **Risk Analysis** - Аналіз ризиків та метрики
- **AI-Powered Insights** - Розумні рекомендації та аналітика
- **Real-time Monitoring** - Моніторинг ринків 24/7

## 🛠 Компоненти Системи

### 1. TradingView Tools (`src/tools/tradingview_tools.py`)
Спеціалізовані інструменти для роботи з TradingView:

```python
from src.tools.tradingview_tools import create_tradingview_tools

# Створення TradingView інструментів
tools = await create_tradingview_tools(session)

# Доступні інструменти:
# - tradingview_crypto_price: Отримання цін криптовалют
# - tradingview_crypto_analysis: Технічний аналіз
# - tradingview_crypto_sentiment: Аналіз настроїв ринку
# - tradingview_crypto_news: Новини криптовалют
# - tradingview_crypto_screener: Скринінг ринків
# - tradingview_realtime_crypto: Real-time дані
```

### 2. Crypto Portfolio Agent (`src/agents/crypto_portfolio_agent.py`)
Інтелектуальний агент для управління портфелями:

```python
from src.agents.crypto_portfolio_agent import CryptoPortfolioAgent

# Ініціалізація агента
agent = CryptoPortfolioAgent(model, session, db)
await agent.initialize()

# Аналіз портфеля
analysis = await agent.analyze_portfolio()

# Моніторинг ринків
market_data = await agent.monitor_markets(['BTCUSD', 'ETHUSD'])

# Генерація звітів
report = await agent.generate_report('daily')
```

### 3. Main System (`src/core/crypto_portfolio_main.py`)
Головна точка входу в систему:

```bash
python src/core/crypto_portfolio_main.py
```

## 🚀 Швидкий Старт

### 1. Встановлення Залежностей

```bash
# Встановлення Python залежностей
pip install -r requirements.txt

# Встановлення Node.js залежностей для Bright Data MCP
npm install -g @brightdata/mcp-server-bright-data
```

### 2. Налаштування Змінних Середовища

```bash
# Створіть .env файл
cp .env.example .env

# Додайте ваші API ключі:
ANTHROPIC_API_KEY=your_anthropic_key
BRIGHT_DATA_API_KEY=your_bright_data_key
```

### 3. Запуск Тестів

```bash
# Базовий тест функціональності
python simple_test.py

# Демо scraping з TradingView
python demo_tradingview_scraping.py

# Повний тест системи
python test_crypto_portfolio_system.py
```

### 4. Запуск Системи

```bash
# Інтерактивна система управління портфелем
python src/core/crypto_portfolio_main.py
```

## 📊 Функціональність

### Аналіз Цін
- Real-time ціни криптовалют
- Історичні дані OHLCV
- Розрахунок змін та волатильності
- Відстань від ATH/ATL

### Технічний Аналіз
- Технічні індикатори (RSI, MACD, MA)
- Сигнали купівлі/продажу
- Рівні підтримки та опору
- Паттерни графіків

### Управління Портфелем
- Трекінг позицій та балансів
- Розрахунок P&L
- Автоматичне оновлення цін
- Історія транзакцій

### Аналіз Ризиків
- Value at Risk (VaR)
- Максимальна просадка
- Коефіцієнт Шарпа
- Концентраційний ризик
- Кореляційний аналіз

### Моніторинг та Алерти
- Real-time моніторинг цін
- Налаштовувані алерти
- Нотифікації про важливі події
- Автоматичні звіти

## 🎯 Приклади Використання

### Аналіз Портфеля

```python
# Створення портфеля
portfolio = {
    "BTCUSD": {"quantity": 0.5, "avg_price": 45000},
    "ETHUSD": {"quantity": 2.0, "avg_price": 3000},
    "ADAUSD": {"quantity": 1000, "avg_price": 1.2}
}

# Аналіз поточної вартості
analysis = await agent.analyze_portfolio()
print(f"Total Value: ${analysis['portfolio_value']:,.2f}")
print(f"Total P&L: ${analysis['total_pnl']:+,.2f}")
```

### Моніторинг Ринків

```python
# Моніторинг топ криптовалют
symbols = ['BTCUSD', 'ETHUSD', 'BNBUSD', 'ADAUSD', 'SOLUSD']
market_data = await agent.monitor_markets(symbols)

# Перевірка алертів
for alert in market_data['alerts']:
    print(f"Alert: {alert}")
```

### Генерація Торгових Сигналів

```python
# Аналіз торгових можливостей
signal = {
    "symbol": "BTCUSD",
    "action": "BUY",
    "price": 50000,
    "confidence": 85
}

# Виконання сигналу з ризик-менеджментом
result = await agent.execute_trade_signal(signal)
print(f"Trade Status: {result['status']}")
```

## 📈 Метрики та KPI

### Портфельні Метрики
- **Total Return**: Загальна прибутковість
- **Sharpe Ratio**: Ризик-скоригована прибутковість
- **Max Drawdown**: Максимальна просадка
- **Win Rate**: Відсоток прибуткових позицій
- **Average Hold Time**: Середній час утримання

### Ризикові Метрики
- **Portfolio Beta**: Кореляція з ринком
- **Value at Risk**: Потенційні втрати
- **Concentration Risk**: Ризик концентрації
- **Correlation Matrix**: Кореляції між активами

### Торгові Метрики
- **Total Trades**: Загальна кількість угод
- **Profit Factor**: Співвідношення прибутку до збитків
- **Average Trade**: Середня угода
- **Maximum Consecutive Losses**: Максимальні послідовні збитки

## 🔧 Налаштування та Конфігурація

### Ризик-Ліміти

```python
risk_limits = {
    "max_position_size": 0.2,    # 20% максимум на позицію
    "max_daily_loss": 0.05,      # 5% максимальний денний збиток
    "min_cash_reserve": 0.1,     # 10% грошовий резерв
    "max_correlation": 0.7,      # Максимальна кореляція між позиціями
    "var_limit": 0.03           # 3% VaR ліміт
}
```

### Алерти та Нотифікації

```python
alerts = [
    {"type": "price", "symbol": "BTCUSD", "condition": ">", "value": 60000},
    {"type": "change", "symbol": "ETHUSD", "condition": "<", "value": -10},
    {"type": "volume", "symbol": "ADAUSD", "condition": ">", "value": 1000000000}
]
```

## 🚀 Розширені Функції

### Machine Learning
- Прогнозування цін
- Класифікація ринкових режимів
- Оптимізація портфеля
- Детекція аномалій

### Автоматизація
- Автоматичне ребалансування
- DCA стратегії
- Stop-loss та take-profit
- Копі-трейдинг

### Інтеграції
- Біржові API (Binance, Coinbase, Kraken)
- DeFi протоколи
- Соціальні мережі
- Новинні джерела

## 📚 Документація

- [Installation Guide](installation.md)
- [API Reference](api.md)
- [Trading Strategies](trading_strategies.md)
- [Risk Management](risk_management.md)
- [Troubleshooting](troubleshooting.md)

## 🤝 Підтримка

Для питань та підтримки:
- GitHub Issues: [DataMCPServerAgent Issues](https://github.com/DimaJoyti/DataMCPServerAgent/issues)
- Email: aws.inspiration@gmail.com
- Telegram: @DimaJoyti

## 📄 Ліцензія

MIT License - див. [LICENSE](../LICENSE) файл для деталей.

---

**🎉 Готові розпочати управління криптовалютним портфелем з TradingView? Запустіть систему та почніть заробляти!**
