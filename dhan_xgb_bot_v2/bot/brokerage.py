# ============================================================
#  bot/brokerage.py  —  Dhan Intraday (MIS) charge calculator
#
#  Implements the EXACT Dhan fee structure confirmed from the
#  Dhan brokerage calculator (https://dhan.co/calculators/brokerage-calculator/)
#
#  Charges for Equity Intraday (MIS) on NSE:
#  ┌─────────────────────────────────┬──────────────────────────────┐
#  │ Component                       │ Rate                         │
#  ├─────────────────────────────────┼──────────────────────────────┤
#  │ Brokerage (each leg)            │ 0.03% of trade value, max ₹20│
#  │ Exchange Transaction Charges    │ 0.00297% of total turnover   │
#  │ STT (Securities Transaction Tax)│ 0.025% on SELL side only     │
#  │ GST (on Brokerage + ETC)        │ 18%                          │
#  │ SEBI Turnover Charge            │ 0.0001% of total turnover    │
#  │ Stamp Duty (on BUY side only)   │ 0.003% of buy value          │
#  │ IPFT                            │ 0.0001% of total turnover    │
#  └─────────────────────────────────┴──────────────────────────────┘
#
#  All amounts in ₹ (Indian Rupees), rounded to 2 decimal places.
#  Verified against screenshots: buy=sell=500 Q=1 → net -0.53
#                                buy=1000 sell=1005 Q=1 → net +3.94
#                                buy=sell=1000 Q=1 → net -1.06
# ============================================================

from dataclasses import dataclass


@dataclass
class ChargeBreakdown:
    """Full itemised breakdown of Dhan intraday charges for one round-trip."""

    buy_price:    float
    sell_price:   float
    quantity:     int

    brokerage:    float   # both legs combined
    etc:          float   # Exchange Transaction Charges
    stt:          float   # Securities Transaction Tax (sell side only)
    gst:          float   # 18% on (brokerage + ETC)
    sebi:         float   # SEBI turnover charge
    stamp:        float   # Stamp duty on buy side
    ipft:         float   # Investor Protection Fund Trust

    total_charges: float  # sum of all above
    gross_pnl:     float  # (sell - buy) * qty
    net_pnl:       float  # gross_pnl - total_charges
    breakeven_sell: float  # sell price needed for net_pnl == 0

    def to_log_string(self) -> str:
        """Compact single-line for logging."""
        sign = "+" if self.net_pnl >= 0 else ""
        return (
            f"gross={sign}₹{self.gross_pnl:.2f}  "
            f"charges=₹{self.total_charges:.2f}  "
            f"[brok={self.brokerage:.2f} ETC={self.etc:.2f} "
            f"STT={self.stt:.2f} GST={self.gst:.2f} "
            f"sebi={self.sebi:.3f} stamp={self.stamp:.3f}]  "
            f"net={sign}₹{self.net_pnl:.2f}"
        )

    def to_telegram_lines(self) -> str:
        """Multi-line HTML block for Telegram messages."""
        sign = "+" if self.net_pnl >= 0 else ""
        net_emoji = "✅" if self.net_pnl >= 0 else "❌"
        return (
            f"\n<b>── P&L Breakdown ──</b>"
            f"\n  Gross P&L       : {sign}₹{self.gross_pnl:.2f}"
            f"\n  Brokerage       : -₹{self.brokerage:.2f}"
            f"\n  ETC             : -₹{self.etc:.3f}"
            f"\n  STT (sell)      : -₹{self.stt:.3f}"
            f"\n  GST (18%)       : -₹{self.gst:.3f}"
            f"\n  SEBI + IPFT     : -₹{(self.sebi + self.ipft):.3f}"
            f"\n  Stamp duty      : -₹{self.stamp:.3f}"
            f"\n  Total charges   : -₹{self.total_charges:.2f}"
            f"\n  ─────────────────────────"
            f"\n  <b>Net P&L          : {sign}₹{self.net_pnl:.2f}  {net_emoji}</b>"
            f"\n  Breakeven sell  : ₹{self.breakeven_sell:.2f}"
        )


def calculate_charges(
    buy_price:  float,
    sell_price: float,
    quantity:   int,
) -> ChargeBreakdown:
    """
    Calculate exact Dhan intraday (MIS) charges for a round-trip trade.

    Parameters
    ----------
    buy_price   : float  — execution fill price for the BUY leg
    sell_price  : float  — execution fill price for the SELL leg
    quantity    : int    — number of shares

    Returns
    -------
    ChargeBreakdown dataclass with every charge component and net P&L.

    Usage
    -----
    >>> c = calculate_charges(buy_price=1000, sell_price=1005, quantity=10)
    >>> print(c.net_pnl)     # real net profit after all charges
    >>> print(c.to_telegram_lines())
    """

    buy_value  = buy_price  * quantity   # ₹ value of BUY leg
    sell_value = sell_price * quantity   # ₹ value of SELL leg
    turnover   = buy_value + sell_value  # total round-trip turnover

    # ── Brokerage: 0.03% per leg, capped at ₹20 per leg ──────────
    brok_buy  = min(0.0003 * buy_value,  20.0)
    brok_sell = min(0.0003 * sell_value, 20.0)
    brokerage = round(brok_buy + brok_sell, 2)

    # ── Exchange Transaction Charges: 0.00297% of turnover ────────
    etc = round(0.0000297 * turnover, 4)

    # ── STT: 0.025% on SELL side only (intraday equity) ───────────
    stt = round(0.00025 * sell_value, 4)

    # ── GST: 18% on (Brokerage + ETC) ────────────────────────────
    gst = round(0.18 * (brokerage + etc), 4)

    # ── SEBI Turnover Charge: 0.0001% of turnover ─────────────────
    sebi = round(0.000001 * turnover, 6)

    # ── Stamp Duty: 0.003% on BUY side only ──────────────────────
    stamp = round(0.00003 * buy_value, 4)

    # ── IPFT: 0.0001% of turnover ────────────────────────────────
    ipft = round(0.000001 * turnover, 6)

    # ── Total ─────────────────────────────────────────────────────
    total_charges = round(brokerage + etc + stt + gst + sebi + stamp + ipft, 2)

    # ── Gross & Net P&L ──────────────────────────────────────────
    gross_pnl = round((sell_price - buy_price) * quantity, 2)
    net_pnl   = round(gross_pnl - total_charges, 2)

    # ── Breakeven sell price (net_pnl == 0) ──────────────────────
    # Approximation: charges are roughly linear, so:
    #   breakeven_sell ≈ buy_price + total_charges / quantity
    # We iterate once to refine (since sell side changes charges slightly).
    approx_be = buy_price + total_charges / max(quantity, 1)
    refined   = calculate_charges(buy_price, approx_be, quantity)
    # final breakeven: offset remaining gap
    be_sell = round(approx_be - refined.net_pnl / max(quantity, 1), 2)

    return ChargeBreakdown(
        buy_price=buy_price,
        sell_price=sell_price,
        quantity=quantity,
        brokerage=brokerage,
        etc=etc,
        stt=stt,
        gst=gst,
        sebi=sebi,
        stamp=stamp,
        ipft=ipft,
        total_charges=total_charges,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        breakeven_sell=be_sell,
    )


# ── Convenience wrappers ──────────────────────────────────────

def net_pnl(
    buy_price:  float,
    sell_price: float,
    quantity:   int,
) -> float:
    """Return only the net P&L (gross minus all Dhan intraday charges)."""
    return calculate_charges(buy_price, sell_price, quantity).net_pnl


def charges_only(
    buy_price:  float,
    sell_price: float,
    quantity:   int,
) -> float:
    """Return only the total charges amount (for risk management calculations)."""
    return calculate_charges(buy_price, sell_price, quantity).total_charges


# ── Self-test (run directly: python -m bot.brokerage) ─────────
if __name__ == "__main__":
    print("=" * 55)
    print("Dhan Intraday Charge Calculator — Self-Test")
    print("=" * 55)

    tests = [
        # (buy, sell, qty, expected_net_pnl)
        (500,  500,  1,  -0.53),
        (1000, 1005, 1,   3.94),
        (1000, 1000, 1,  -1.06),
    ]

    all_pass = True
    for buy, sell, qty, expected in tests:
        c = calculate_charges(buy, sell, qty)
        status = "✅ PASS" if abs(c.net_pnl - expected) < 0.05 else "❌ FAIL"
        if "FAIL" in status:
            all_pass = False
        print(f"\nBuy=₹{buy}  Sell=₹{sell}  Qty={qty}")
        print(f"  {c.to_log_string()}")
        print(f"  Expected net: ₹{expected:.2f}   Got: ₹{c.net_pnl:.2f}   {status}")

    print()
    print("=" * 55)
    print("All tests passed ✅" if all_pass else "SOME TESTS FAILED ❌")
    print("=" * 55)

    # Realistic example: HDFCBANK 10 shares
    c = calculate_charges(buy_price=1800, sell_price=1818, quantity=10)
    print("\n── Realistic example: HDFCBANK ──")
    print(f"  Buy ₹1800 × 10 → Sell ₹1818 × 10")
    print(c.to_telegram_lines())
