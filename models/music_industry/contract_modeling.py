"""Music industry contract modeling: deal structures, advance recoupment, cross-collateralization.

Models the financial mechanics of recording contracts, publishing deals,
distribution agreements, and 360 deals with recoupment schedules,
cross-collateralization, and break-even analysis.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum


class DealType(Enum):
    TRADITIONAL_RECORD = "traditional_record"
    DISTRIBUTION_ONLY = "distribution_only"
    NET_PROFIT = "net_profit"
    LICENSING = "licensing"
    PUBLISHING_FULL = "publishing_full"
    PUBLISHING_COPUB = "publishing_copub"
    PUBLISHING_ADMIN = "publishing_admin"
    THREE_SIXTY = "360"


@dataclass
class RoyaltyRate:
    base_rate: float
    escalation_thresholds: List[int] = field(default_factory=list)
    escalated_rates: List[float] = field(default_factory=list)

    def effective_rate(self, units_sold: int) -> float:
        rate = self.base_rate
        for threshold, esc_rate in zip(self.escalation_thresholds, self.escalated_rates):
            if units_sold >= threshold:
                rate = esc_rate
        return rate


@dataclass
class RecordDeal:
    deal_type: DealType
    advance: float
    royalty_rate: RoyaltyRate
    packaging_deduction: float = 0.0
    free_goods_pct: float = 0.0
    recording_costs_recoupable: bool = True
    marketing_costs_recoupable: bool = False
    cross_collateralized: bool = False
    term_albums: int = 1
    option_albums: int = 0

    def net_royalty_rate(self) -> float:
        base = self.royalty_rate.base_rate
        after_packaging = base * (1.0 - self.packaging_deduction)
        after_free_goods = after_packaging * (1.0 - self.free_goods_pct)
        return after_free_goods

    def units_to_recoup(self, unit_price: float,
                        recording_costs: float = 0.0,
                        marketing_costs: float = 0.0) -> float:
        recoupable = self.advance
        if self.recording_costs_recoupable:
            recoupable += recording_costs
        if self.marketing_costs_recoupable:
            recoupable += marketing_costs
        per_unit = unit_price * self.net_royalty_rate()
        if per_unit <= 0:
            return float('inf')
        return recoupable / per_unit


@dataclass
class PublishingDeal:
    deal_type: DealType
    advance: float
    writer_share: float
    publisher_share: float
    admin_fee_pct: float = 0.0
    term_years: int = 1
    retention_years: int = 0

    def effective_writer_income(self, gross_publishing: float) -> float:
        if self.deal_type == DealType.PUBLISHING_ADMIN:
            return gross_publishing * (1.0 - self.admin_fee_pct)
        return gross_publishing * self.writer_share


@dataclass
class ThreeSixtyDeal:
    advance: float
    recording_pct: float
    publishing_pct: float
    touring_pct: float
    merch_pct: float
    endorsement_pct: float = 0.0
    recording_royalty: RoyaltyRate = field(default_factory=lambda: RoyaltyRate(0.15))

    def total_label_share(self, recording_rev: float, publishing_rev: float,
                          touring_rev: float, merch_rev: float,
                          endorsement_rev: float = 0.0) -> float:
        return (recording_rev * self.recording_pct +
                publishing_rev * self.publishing_pct +
                touring_rev * self.touring_pct +
                merch_rev * self.merch_pct +
                endorsement_rev * self.endorsement_pct)


@dataclass
class RecoupmentState:
    advance_balance: float
    earned_royalties: float = 0.0
    recouped: bool = False
    periods: List[float] = field(default_factory=list)
    balances: List[float] = field(default_factory=list)

    def apply_earnings(self, amount: float) -> float:
        self.earned_royalties += amount
        self.advance_balance = max(0.0, self.advance_balance - amount)
        self.recouped = self.advance_balance <= 0
        payout = max(0.0, amount - (self.advance_balance + amount - self.earned_royalties)
                     if not self.recouped else amount)
        if self.recouped:
            payout = amount
        else:
            payout = 0.0
        self.balances.append(self.advance_balance)
        return payout


def simulate_recoupment(deal: RecordDeal, unit_price: float,
                        monthly_units: np.ndarray,
                        recording_costs: float = 0.0,
                        marketing_costs: float = 0.0) -> Dict:
    recoupable = deal.advance
    if deal.recording_costs_recoupable:
        recoupable += recording_costs
    if deal.marketing_costs_recoupable:
        recoupable += marketing_costs

    balance = recoupable
    cumulative_units = 0
    cumulative_earnings = 0.0
    balances = []
    artist_payouts = []
    recoup_month = -1

    for month, units in enumerate(monthly_units):
        cumulative_units += int(units)
        rate = deal.royalty_rate.effective_rate(cumulative_units)
        net_rate = rate * (1.0 - deal.packaging_deduction) * (1.0 - deal.free_goods_pct)
        earnings = float(units) * unit_price * net_rate
        cumulative_earnings += earnings

        if balance > 0:
            balance = max(0.0, balance - earnings)
            artist_payout = 0.0
            if balance == 0:
                overage = earnings - (recoupable - (cumulative_earnings - earnings))
                artist_payout = max(0.0, overage)
                if recoup_month < 0:
                    recoup_month = month
        else:
            artist_payout = earnings

        balances.append(balance)
        artist_payouts.append(artist_payout)

    return {
        "recoup_month": recoup_month,
        "recouped": balance <= 0,
        "final_balance": balance,
        "cumulative_earnings": cumulative_earnings,
        "cumulative_units": cumulative_units,
        "balances": np.array(balances),
        "artist_payouts": np.array(artist_payouts),
        "total_artist_income": float(np.sum(artist_payouts)) + deal.advance,
    }


def cross_collateralize(deals: List[RecordDeal],
                        earnings_per_deal: List[float]) -> List[float]:
    total_advance = sum(d.advance for d in deals)
    total_earnings = sum(earnings_per_deal)
    if total_earnings >= total_advance:
        return [e - d.advance for d, e in zip(deals, earnings_per_deal)]
    shortfalls = []
    for d, e in zip(deals, earnings_per_deal):
        shortfalls.append(max(0.0, d.advance - e))
    total_shortfall = sum(shortfalls)
    payouts = []
    for d, e in zip(deals, earnings_per_deal):
        surplus = max(0.0, e - d.advance)
        deduction = min(surplus, total_shortfall) if total_shortfall > 0 else 0.0
        payouts.append(surplus - deduction)
        total_shortfall = max(0.0, total_shortfall - deduction)
    return payouts


def deal_npv(cashflows: np.ndarray, discount_rate: float = 0.08) -> float:
    periods = np.arange(len(cashflows))
    return float(np.sum(cashflows / (1 + discount_rate) ** periods))


def deal_irr(cashflows: np.ndarray, tol: float = 1e-6, max_iter: int = 100) -> float:
    r = 0.1
    for _ in range(max_iter):
        periods = np.arange(len(cashflows))
        npv = np.sum(cashflows / (1 + r) ** periods)
        dnpv = np.sum(-periods * cashflows / (1 + r) ** (periods + 1))
        if abs(dnpv) < 1e-15:
            break
        r_new = r - npv / dnpv
        if abs(r_new - r) < tol:
            return r_new
        r = r_new
    return r


def compare_deals(deals: List[Dict], projected_revenue: np.ndarray) -> List[Dict]:
    results = []
    for deal_info in deals:
        deal_type = deal_info.get("type", "unknown")
        advance = deal_info.get("advance", 0.0)
        royalty = deal_info.get("royalty_rate", 0.15)
        costs = deal_info.get("costs", 0.0)

        earnings = projected_revenue * royalty
        recoupable = advance + costs
        cumulative = np.cumsum(earnings)
        recoup_idx = np.searchsorted(cumulative, recoupable)
        recoup_month = int(recoup_idx) if recoup_idx < len(cumulative) else -1

        cashflows = np.concatenate([[advance], earnings])
        artist_cashflows = np.concatenate([[-advance], earnings])
        artist_cashflows_payout = np.where(np.cumsum(earnings) > recoupable,
                                           earnings, 0.0)
        artist_total = advance + float(np.sum(artist_cashflows_payout))

        results.append({
            "deal_type": deal_type,
            "advance": advance,
            "total_artist_income": artist_total,
            "recoup_month": recoup_month,
            "npv_artist": deal_npv(np.concatenate([[advance], artist_cashflows_payout])),
            "total_earnings": float(np.sum(earnings)),
        })
    return sorted(results, key=lambda x: x["npv_artist"], reverse=True)


@dataclass
class MechanicalRoyalty:
    statutory_rate_per_unit: float = 0.0917
    long_song_rate_per_minute: float = 0.0175
    duration_threshold_minutes: float = 5.0

    def compute(self, duration_seconds: float, units: int) -> float:
        minutes = duration_seconds / 60.0
        if minutes <= self.duration_threshold_minutes:
            rate = self.statutory_rate_per_unit
        else:
            rate = self.long_song_rate_per_minute * minutes
        return rate * units


@dataclass
class PerformanceRoyalty:
    writer_share: float = 0.5
    publisher_share: float = 0.5

    def split(self, gross: float) -> Tuple[float, float]:
        return gross * self.writer_share, gross * self.publisher_share


def sync_license_value(track_popularity: float, placement_type: str,
                       exclusivity: bool = False) -> Tuple[float, float]:
    base_ranges = {
        "major_film": (15000, 300000),
        "indie_film": (1000, 10000),
        "tv_series": (5000, 50000),
        "commercial_national": (25000, 500000),
        "commercial_local": (500, 5000),
        "video_game": (5000, 50000),
        "trailer": (10000, 100000),
        "web_content": (200, 2000),
    }
    lo, hi = base_ranges.get(placement_type, (1000, 10000))
    popularity_factor = 0.2 + 0.8 * track_popularity
    value = lo + (hi - lo) * popularity_factor
    if exclusivity:
        value *= 1.5
    master_fee = value * 0.5
    sync_fee = value * 0.5
    return master_fee, sync_fee
