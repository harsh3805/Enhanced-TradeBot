"""
Indian Market Cost Model — Full transaction cost calculator for NSE/BSE.
Calculates STT, stamp duty, exchange charges, SEBI charges, GST, brokerage,
and slippage by liquidity tier for Indian equity markets.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LiquidityTier(Enum):
    """Stock liquidity classification based on market cap."""
    LARGE_CAP = "large"    # market_cap > 1,00,000 Cr
    MID_CAP = "mid"        # market_cap 10,000 - 1,00,000 Cr
    SMALL_CAP = "small"    # market_cap < 10,000 Cr


class TradeType(Enum):
    """Trade classification for tax calculation."""
    DELIVERY = "delivery"  # Holdings (buy today, sell later)
    INTRADAY = "intraday"  # Same-day square-off


@dataclass
class TradeCosts:
    """Detailed cost breakdown for a single trade leg."""
    brokerage: float = 0.0
    stt: float = 0.0
    stamp_duty: float = 0.0
    exchange_charges: float = 0.0
    sebi_charges: float = 0.0
    gst: float = 0.0
    slippage: float = 0.0
    total: float = 0.0


@dataclass
class CostBreakdown:
    """Complete cost analysis for a round-trip trade."""
    buy_costs: TradeCosts = None
    sell_costs: TradeCosts = None
    total_trade_value: float = 0.0
    round_trip: float = 0.0
    break_even_pct: float = 0.0
    break_even_rupees: float = 0.0
    net_return_pct: float = 0.0
    costs_as_pct: float = 0.0


# ── Current SEBI/NSE/Government Rates (as of 2025-26) ──────────

# Securities Transaction Tax
STT_DELIVERY_SELL = 0.1       # 0.1% on delivery sell side
STT_INTRADAY_BUY = 0.025      # 0.025% on intraday buy
STT_INTRADAY_SELL = 0.025     # 0.025% on intraday sell

# Stamp Duty (varies by state; using average Maharashtra rate)
STAMP_DUTY_BUY = 0.015        # 0.015% on buy side only

# Exchange Transaction Charges
EXCHANGE_NSE_EQUITY = 0.00345  # 0.00345% NSE turnover
EXCHANGE_BSE_EQUITY = 0.00375  # 0.00375% BSE turnover

# SEBI Charges
SEBI_PER_CRORE = 10.0         # ₹10 per crore of turnover

# GST
GST_PCT = 18.0                # 18% on (brokerage + exchange charges)

# Slippage by Liquidity Tier (base estimates)
SLIPPAGE_TIERS = {
    LiquidityTier.LARGE_CAP: 0.05,
    LiquidityTier.MID_CAP: 0.15,
    LiquidityTier.SMALL_CAP: 0.30,
}

# Default brokerage (discount broker)
DEFAULT_BROKERAGE_PCT = 0.03  # 0.03% per trade (e.g., Zerodha, Angel)


class IndianMarketCostModel:
    """
    Complete Indian equity market cost calculator.

    Usage:
        model = IndianMarketCostModel()
        costs = model.calculate_round_trip_costs(
            entry_price=1500.0, exit_price=1550.0,
            quantity=100, trade_type="delivery",
            market_cap_cr=800000  # Reliance-sized
        )
        print(costs.break_even_pct)  # ~0.34%
    """

    def __init__(self, brokerage_pct: float = DEFAULT_BROKERAGE_PCT, exchange: str = "NSE"):
        self.brokerage_pct = brokerage_pct
        self.exchange_charges_pct = EXCHANGE_NSE_EQUITY if exchange.upper() == "NSE" else EXCHANGE_BSE_EQUITY

    def calculate_leg_costs(
        self,
        price: float,
        quantity: int,
        side: str,              # "BUY" or "SELL"
        trade_type: str = "delivery",
        slippage_pct: float = 0.05,
        include_slippage: bool = True,
    ) -> TradeCosts:
        """
        Calculate all costs for a single trade leg.

        Args:
            price: Execution price per share
            quantity: Number of shares
            side: "BUY" or "SELL"
            trade_type: "delivery" or "intraday"
            slippage_pct: Estimated slippage as percentage
            include_slippage: Whether to include slippage cost

        Returns:
            TradeCosts dataclass with all cost components
        """
        turnover = price * quantity
        costs = TradeCosts()

        # Brokerage
        costs.brokerage = turnover * (self.brokerage_pct / 100)

        # Securities Transaction Tax
        if side.upper() == "SELL":
            if trade_type == "delivery":
                costs.stt = turnover * (STT_DELIVERY_SELL / 100)
            else:
                costs.stt = turnover * (STT_INTRADAY_SELL / 100)
        else:  # BUY
            if trade_type == "intraday":
                costs.stt = turnover * (STT_INTRADAY_BUY / 100)
            # Delivery buy has NO STT

        # Stamp Duty (buy side only)
        if side.upper() == "BUY":
            costs.stamp_duty = turnover * (STAMP_DUTY_BUY / 100)

        # Exchange Transaction Charges
        costs.exchange_charges = turnover * (self.exchange_charges_pct / 100)

        # SEBI Charges
        costs.sebi_charges = (turnover / 10000000) * SEBI_PER_CRORE

        # GST on (brokerage + exchange charges)
        taxable = costs.brokerage + costs.exchange_charges
        costs.gst = taxable * (GST_PCT / 100)

        # Slippage (market impact cost)
        if include_slippage:
            costs.slippage = turnover * (slippage_pct / 100)

        costs.total = sum([
            costs.brokerage, costs.stt, costs.stamp_duty,
            costs.exchange_charges, costs.sebi_charges,
            costs.gst, costs.slippage,
        ])

        return costs

    def calculate_round_trip_costs(
        self,
        entry_price: float,
        exit_price: float,
        quantity: int,
        trade_type: str = "delivery",
        market_cap_cr: Optional[float] = None,
    ) -> CostBreakdown:
        """
        Calculate total cost for a complete buy+sell cycle.

        Args:
            entry_price: Buy price per share
            exit_price: Sell price per share
            quantity: Number of shares
            trade_type: "delivery" or "intraday"
            market_cap_cr: Market cap in crores (for liquidity tier detection)

        Returns:
            CostBreakdown with full cost analysis
        """
        tier = self.detect_liquidity_tier(market_cap_cr) if market_cap_cr else LiquidityTier.MID_CAP
        slippage = SLIPPAGE_TIERS.get(tier, 0.15)

        buy_costs = self.calculate_leg_costs(
            entry_price, quantity, "BUY", trade_type, slippage
        )
        sell_costs = self.calculate_leg_costs(
            exit_price, quantity, "SELL", trade_type, slippage
        )

        total_trade_value = (entry_price + exit_price) * quantity
        round_trip_total = buy_costs.total + sell_costs.total

        # Break-even analysis
        entry_value = entry_price * quantity
        break_even_move = round_trip_total
        break_even_pct = (break_even_move / entry_value) * 100 if entry_value else 0

        # Net return after costs
        gross_return = (exit_price - entry_price) * quantity
        net_return = gross_return - round_trip_total
        net_return_pct = (net_return / entry_value) * 100 if entry_value else 0

        costs_as_pct = (round_trip_total / entry_value) * 100 if entry_value else 0

        return CostBreakdown(
            buy_costs=buy_costs,
            sell_costs=sell_costs,
            total_trade_value=round(total_trade_value, 2),
            round_trip=round(round_trip_total, 2),
            break_even_pct=round(break_even_pct, 3),
            break_even_rupees=round(break_even_move, 2),
            net_return_pct=round(net_return_pct, 2),
            costs_as_pct=round(costs_as_pct, 3),
        )

    def break_even_move(self, entry_price: float, quantity: int, trade_type: str = "delivery",
                        market_cap_cr: Optional[float] = None) -> float:
        """
        Minimum percentage price move needed to cover all costs.
        For delivery: typically 0.3-0.5%
        For intraday: typically 0.10-0.15%
        """
        # Calculate rough break-even without needing exit price
        tier = self.detect_liquidity_tier(market_cap_cr) if market_cap_cr else LiquidityTier.MID_CAP
        slippage = SLIPPAGE_TIERS.get(tier, 0.15)

        # Total cost rate (in percentage) for round-trip
        stt_rate = 0
        if trade_type == "delivery":
            stt_rate = STT_DELIVERY_SELL / 2  # averaged over buy+sell
        else:
            stt_rate = (STT_INTRADAY_BUY + STT_INTRADAY_SELL) / 2  # averaged over buy+sell

        # Stamp duty on buy only (half of round trip)
        stamp_rate = STAMP_DUTY_BUY / 2

        # All other costs apply both sides
        brokerage_rate = self.brokerage_pct
        exchange_rate = self.exchange_charges_pct
        sebi_rate = SEBI_PER_CRORE / 10000000  # per rupee
        gst_rate = GST_PCT / 100 * (brokerage_rate + exchange_rate) / 100

        total_cost_rate = (
            brokerage_rate + stt_rate + stamp_rate +
            exchange_rate + sebi_rate * 100 + gst_rate +
            slippage / 2 * 100  # slippage on average
        )

        return round(total_cost_rate, 3)

    def estimate_slippage(self, price: float, volume: float, avg_volume: float,
                          market_cap_cr: Optional[float] = None) -> float:
        """
        Estimate slippage based on liquidity tier and volume ratio.

        Adjusts base slippage:
        - If current volume > 2x average → lower slippage (liquid)
        - If current volume < 0.5x average → higher slippage (thin)
        - Market cap provides base tier
        """
        tier = self.detect_liquidity_tier(market_cap_cr) if market_cap_cr else LiquidityTier.MID_CAP
        base_slippage = SLIPPAGE_TIERS.get(tier, 0.15)

        if avg_volume <= 0:
            return base_slippage

        vol_ratio = volume / avg_volume
        adjustment = 1.0
        if vol_ratio > 2.0:
            adjustment = 0.7
        elif vol_ratio > 1.5:
            adjustment = 0.85
        elif vol_ratio < 0.3:
            adjustment = 1.5
        elif vol_ratio < 0.5:
            adjustment = 1.3

        return round(base_slippage * adjustment, 4)

    @staticmethod
    def detect_liquidity_tier(market_cap_cr: Optional[float]) -> LiquidityTier:
        """Determine liquidity tier from market cap in crores."""
        if market_cap_cr is None:
            return LiquidityTier.MID_CAP
        if market_cap_cr >= 100000:  # ≥ 1 Lakh Cr
            return LiquidityTier.LARGE_CAP
        elif market_cap_cr >= 10000:  # ≥ 10,000 Cr
            return LiquidityTier.MID_CAP
        return LiquidityTier.SMALL_CAP

    def cost_summary_for_signal(self, price: float, expected_return_pct: float,
                                quantity: int, trade_type: str = "delivery",
                                market_cap_cr: Optional[float] = None) -> dict:
        """
        Quick summary for displaying alongside trading signals.

        Args:
            price: Current/entry price
            expected_return_pct: Expected return percentage (e.g., 5.0 for 5%)
            quantity: Planned quantity
            trade_type: "delivery" or "intraday"

        Returns:
            dict with signals/compatible summary
        """
        breakeven = self.break_even_move(price, quantity, trade_type, market_cap_cr)
        exit_price = price * (1 + expected_return_pct / 100)
        costs = self.calculate_round_trip_costs(price, exit_price, quantity, trade_type, market_cap_cr)

        return {
            "break_even_pct": breakeven,
            "costs_as_pct": costs.costs_as_pct,
            "round_trip_costs": costs.round_trip,
            "profitable": expected_return_pct > breakeven,
            "net_if_target_hit": round(expected_return_pct - costs.costs_as_pct, 2),
        }
