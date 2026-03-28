"""Simple order book simulator for microstructure research."""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Order:
    """Represents a single order in the order book.

    Attributes:
        side: 'buy' or 'sell'.
        size: Number of units.
        price: Limit price (None for market orders).
        order_type: 'limit' or 'market'.
        timestamp: When the order was placed.
        order_id: Unique identifier (auto-generated).
    """

    side: str
    size: float
    price: float | None
    order_type: str
    timestamp: pd.Timestamp
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Fill:
    """Represents an order fill (execution report).

    Attributes:
        order_id: ID of the filled order.
        filled_size: Quantity actually filled.
        avg_price: Volume-weighted average fill price.
        timestamp: Fill timestamp.
        cost: Total market impact cost incurred.
    """

    order_id: str
    filled_size: float
    avg_price: float
    timestamp: pd.Timestamp
    cost: float


class OrderBook:
    """Simulates a limit order book for a single asset.

    Maintains bid and ask sides as sorted price levels and supports
    market and limit orders, cancellations, and market-data-driven updates.

    Attributes:
        ticker: Asset identifier.
        tick_size: Minimum price increment.
        bids: Dict mapping price → list of Orders (buy side).
        asks: Dict mapping price → list of Orders (sell side).
        orders: Registry of all active orders by order_id.
    """

    def __init__(self, ticker: str, tick_size: float = 0.01) -> None:
        """Initializes OrderBook.

        Args:
            ticker: Asset ticker symbol.
            tick_size: Minimum price increment for rounding.
        """
        self.ticker = ticker
        self.tick_size = tick_size
        self.bids: dict[float, list[Order]] = defaultdict(list)
        self.asks: dict[float, list[Order]] = defaultdict(list)
        self.orders: dict[str, Order] = {}

    def _round_price(self, price: float) -> float:
        """Rounds a price to the nearest tick size."""
        return round(round(price / self.tick_size) * self.tick_size, 10)

    def add_limit_order(self, side: str, size: float, price: float) -> str:
        """Adds a limit order to the book.

        Args:
            side: 'buy' or 'sell'.
            size: Order quantity.
            price: Limit price.

        Returns:
            Unique order ID string.
        """
        rounded = self._round_price(price)
        order = Order(
            side=side,
            size=size,
            price=rounded,
            order_type="limit",
            timestamp=pd.Timestamp.now(),
        )
        self.orders[order.order_id] = order
        if side == "buy":
            self.bids[rounded].append(order)
        else:
            self.asks[rounded].append(order)
        return order.order_id

    def add_market_order(self, side: str, size: float) -> Fill:
        """Executes a market order immediately at best available price.

        Args:
            side: 'buy' (takes from asks) or 'sell' (takes from bids).
            size: Order quantity.

        Returns:
            Fill dataclass with execution details.
        """
        order_id = str(uuid.uuid4())
        fill_price = self.simulate_fill(side, size).avg_price
        return Fill(
            order_id=order_id,
            filled_size=size,
            avg_price=fill_price,
            timestamp=pd.Timestamp.now(),
            cost=0.0,
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancels an existing limit order by ID.

        Args:
            order_id: The order ID to cancel.

        Returns:
            True if the order was found and cancelled, False otherwise.
        """
        if order_id not in self.orders:
            return False
        order = self.orders.pop(order_id)
        book_side = self.bids if order.side == "buy" else self.asks
        if order.price is not None and order.price in book_side:
            book_side[order.price] = [
                o for o in book_side[order.price] if o.order_id != order_id
            ]
            if not book_side[order.price]:
                del book_side[order.price]
        return True

    def get_bid(self) -> float | None:
        """Returns the best (highest) bid price.

        Returns:
            Best bid price, or None if no bids exist.
        """
        if not self.bids:
            return None
        return max(self.bids.keys())

    def get_ask(self) -> float | None:
        """Returns the best (lowest) ask price.

        Returns:
            Best ask price, or None if no asks exist.
        """
        if not self.asks:
            return None
        return min(self.asks.keys())

    def get_spread(self) -> float | None:
        """Returns the bid-ask spread.

        Returns:
            Spread (ask - bid), or None if either side is empty.
        """
        bid = self.get_bid()
        ask = self.get_ask()
        if bid is None or ask is None:
            return None
        return ask - bid

    def simulate_fill(
        self, side: str, size: float, market_impact: float = 0.0
    ) -> Fill:
        """Simulates a fill against the order book with optional market impact.

        Args:
            side: 'buy' or 'sell'.
            size: Quantity to fill.
            market_impact: Additional price impact in dollars to add to
                the fill price (e.g. from an Almgren-Chriss model).

        Returns:
            Fill dataclass representing the simulated execution.
        """
        order_id = str(uuid.uuid4())
        if side == "buy":
            base_price = self.get_ask()
        else:
            base_price = self.get_bid()

        if base_price is None:
            # No quotes; use a placeholder price of 100
            base_price = 100.0

        if side == "buy":
            fill_price = base_price + market_impact
        else:
            fill_price = base_price - market_impact

        cost = abs(market_impact) * size

        return Fill(
            order_id=order_id,
            filled_size=size,
            avg_price=fill_price,
            timestamp=pd.Timestamp.now(),
            cost=cost,
        )

    def _update_book(self, prices: pd.Series, volumes: pd.Series) -> None:
        """Updates the order book from a snapshot of market prices and volumes.

        Clears existing orders and repopulates both sides of the book
        based on the current mid-price and estimated spread.

        Args:
            prices: Series of recent prices to infer mid-price and spread.
            volumes: Series of recent volumes (unused in this implementation
                but available for future LOB reconstruction).
        """
        self.bids.clear()
        self.asks.clear()
        self.orders.clear()

        if prices.empty:
            return

        mid = float(prices.iloc[-1])
        # Estimate spread as 0.1% of mid price
        half_spread = mid * 0.001
        bid = self._round_price(mid - half_spread)
        ask = self._round_price(mid + half_spread)

        default_size = float(volumes.mean()) if not volumes.empty else 1000.0
        self.add_limit_order("buy", default_size, bid)
        self.add_limit_order("sell", default_size, ask)
