# ============================================================
#  bot/nse_holidays.py  —  NSE trading holiday calendar
# ============================================================
"""
Bot checks this before trading each day.
If today is a holiday, bot skips and sends Telegram alert.
Update this list each year in December.
"""

from datetime import date

# NSE Equity market holidays 2025
NSE_HOLIDAYS_2025 = {
    date(2025, 1, 26),   # Republic Day
    date(2025, 2, 26),   # Mahashivratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Id-Ul-Fitr (Ramzan Eid)
    date(2025, 4, 14),   # Dr. Ambedkar Jayanti / Ram Navami
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 27),   # Ganesh Chaturthi
    date(2025, 10, 2),   # Gandhi Jayanti / Dussehra
    date(2025, 10, 20),  # Diwali Laxmi Puja (Muhurat trading — check NSE)
    date(2025, 10, 21),  # Diwali Balipratipada
    date(2025, 11, 5),   # Prakash Gurpurb
    date(2025, 12, 25),  # Christmas
}

# NSE Equity market holidays 2026
NSE_HOLIDAYS_2026 = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 3),    # Mahashivratri (tentative)
    date(2026, 3, 20),   # Holi (tentative)
    date(2026, 4, 3),    # Good Friday (tentative)
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 8, 15),   # Independence Day
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 11, 3),   # Diwali (tentative — check NSE circular)
    date(2026, 12, 25),  # Christmas
}

ALL_HOLIDAYS = NSE_HOLIDAYS_2025 | NSE_HOLIDAYS_2026


def is_holiday_today() -> bool:
    """Returns True if today is an NSE holiday."""
    today = date.today()
    # Also skip weekends (Saturday=5, Sunday=6)
    if today.weekday() >= 5:
        return True
    return today in ALL_HOLIDAYS


def get_holiday_name(d: date) -> str:
    """Returns holiday name if date is a holiday."""
    names = {
        date(2025, 1, 26): "Republic Day",
        date(2025, 2, 26): "Mahashivratri",
        date(2025, 3, 14): "Holi",
        date(2025, 3, 31): "Ramzan Eid",
        date(2025, 4, 14): "Dr. Ambedkar Jayanti",
        date(2025, 4, 18): "Good Friday",
        date(2025, 5, 1):  "Maharashtra Day",
        date(2025, 8, 15): "Independence Day",
        date(2025, 8, 27): "Ganesh Chaturthi",
        date(2025, 10, 2): "Gandhi Jayanti",
        date(2025, 10, 20):"Diwali Laxmi Puja",
        date(2025, 10, 21):"Diwali Balipratipada",
        date(2025, 11, 5): "Prakash Gurpurb",
        date(2025, 12, 25):"Christmas",
        date(2026, 1, 26): "Republic Day",
        date(2026, 4, 3):  "Good Friday",
        date(2026, 4, 14): "Dr. Ambedkar Jayanti",
        date(2026, 5, 1):  "Maharashtra Day",
        date(2026, 8, 15): "Independence Day",
        date(2026, 10, 2): "Gandhi Jayanti",
        date(2026, 12, 25):"Christmas",
    }
    return names.get(d, "NSE Holiday")


def next_trading_day() -> date:
    """Returns the next trading day (skips weekends + holidays)."""
    from datetime import timedelta
    d = date.today() + timedelta(days=1)
    while d.weekday() >= 5 or d in ALL_HOLIDAYS:
        d += timedelta(days=1)
    return d


if __name__ == "__main__":
    today = date.today()
    if is_holiday_today():
        print(f"Today ({today}) is a holiday: {get_holiday_name(today)}")
        print(f"Next trading day: {next_trading_day()}")
    else:
        print(f"Today ({today}) is a trading day.")
        print(f"Next trading day: {next_trading_day()}")
