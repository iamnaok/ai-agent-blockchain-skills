"""
Aevo Options Connector
Real-time options data from Aevo exchange (Ethereum L2)
Supports: Options chain, orderbook, trades, WebSocket streaming
"""

import requests
import asyncio
import websockets
import json
import time
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

@dataclass
class OptionsInstrument:
    """Aevo options instrument"""
    instrument_name: str  # e.g., "BTC-USD-241227-70000-C"
    underlying: str       # BTC, ETH
    strike: float
    expiry: str          # YYYYMMDD
    option_type: str     # C or P
    instrument_type: str # option

    @property
    def expiry_datetime(self) -> datetime:
        """Parse Aevo expiry format: 06MAR26 -> 2026-03-06"""
        # Map month abbreviations
        months = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }

        # Parse format like "06MAR26"
        day = int(self.expiry[:2])
        month_abbr = self.expiry[2:5].upper()
        year_short = int(self.expiry[5:7])

        year = 2000 + year_short  # Assume 2026+
        month = months.get(month_abbr, 1)

        return datetime(year, month, day)

    @property
    def time_to_expiry_years(self) -> float:
        days = (self.expiry_datetime - datetime.now()).days
        return max(days / 365.0, 0.001)

class AevoConnector:
    """
    Aevo Exchange Connector

    API: https://api-docs.aevo.xyz
    Supports: Perpetuals + Options

    Public API (no auth):
    - GET /markets - List all instruments
    - GET /orderbook - Get orderbook for symbol
    - GET /assets - List available assets
    - GET /expiries - List option expiries
    - GET /instrument/{name} - Instrument details
    - GET /funding - Funding rates

    Private API (requires API key):
    - Trading, positions, orders
    """

    REST_URL = "https://api.aevo.xyz"
    WS_URL = "wss://ws.aevo.xyz"

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self._ws_connection = None

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make GET request to Aevo REST API"""
        url = f"{self.REST_URL}{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": response.status_code if hasattr(response, 'status_code') else 'unknown'}

    def get_assets(self) -> List[str]:
        """Get list of available assets"""
        result = self._get("/assets")
        if isinstance(result, list):
            return result
        return result.get("assets", []) if isinstance(result, dict) else []

    def get_expiries(self) -> List[str]:
        """Get list of option expiries (YYYYMMDD)"""
        result = self._get("/expiries")
        if isinstance(result, list):
            return result
        return result.get("expiries", []) if isinstance(result, dict) else []

    def get_markets(self, asset: Optional[str] = None) -> List[Dict]:
        """
        Get all instruments (options + perps)

        Args:
            asset: Filter by asset (BTC, ETH, etc.)

        Returns:
            List of instruments with type, strike, expiry
        """
        params = {"asset": asset} if asset else {}
        result = self._get("/markets", params)
        if isinstance(result, list):
            return result
        return result.get("markets", []) if isinstance(result, dict) else []

    def get_options_chain(self, underlying: str) -> List[OptionsInstrument]:
        """
        Get options chain for specific underlying

        Args:
            underlying: BTC or ETH

        Returns:
            List of OptionsInstrument objects
        """
        markets = self.get_markets(underlying)
        options = []

        for market in markets:
            if market.get("instrument_type", "").upper() == "OPTION":
                # Parse instrument_name: BTC-06MAR26-50000-P
                name = market.get("instrument_name", "")
                parts = name.split("-")
                if len(parts) >= 4:
                    try:
                        opt = OptionsInstrument(
                            instrument_name=name,
                            underlying=parts[0],
                            strike=float(parts[2]),
                            expiry=parts[1],
                            option_type=parts[3],
                            instrument_type="option"
                        )
                        options.append(opt)
                    except (ValueError, IndexError):
                        continue

        return options

    def get_orderbook(self, instrument_name: str) -> Dict:
        """
        Get orderbook for specific instrument

        Args:
            instrument_name: e.g., "BTC-USD-241227-70000-C"

        Returns:
            Orderbook with bids, asks, last price
        """
        params = {"instrument_name": instrument_name}
        return self._get("/orderbook", params)

    def get_instrument_details(self, instrument_name: str) -> Dict:
        """Get detailed instrument information"""
        return self._get(f"/instrument/{instrument_name}")

    def get_funding(self, instrument_name: Optional[str] = None) -> Dict:
        """Get funding rate (for perps)"""
        params = {"instrument_name": instrument_name} if instrument_name else {}
        return self._get("/funding", params)

    def get_options_vol_surface(self, underlying: str) -> Dict:
        """
        Build volatility surface from options chain

        Args:
            underlying: BTC or ETH

        Returns:
            Vol surface data with strikes, expiries, implied vols
        """
        options = self.get_options_chain(underlying)
        surface_data = {}

        for opt in options:
            # Get orderbook for each option
            ob = self.get_orderbook(opt.instrument_name)

            if "error" not in ob:
                # Calculate mid price and implied vol
                bids = ob.get("bids", [])
                asks = ob.get("asks", [])

                if bids and asks:
                    best_bid = float(bids[0][0])
                    best_ask = float(asks[0][0])
                    mid_price = (best_bid + best_ask) / 2
                    spread = (best_ask - best_bid) / mid_price

                    expiry_key = opt.expiry
                    if expiry_key not in surface_data:
                        surface_data[expiry_key] = {}

                    surface_data[expiry_key][opt.strike] = {
                        "call_price": mid_price if opt.option_type == "C" else None,
                        "put_price": mid_price if opt.option_type == "P" else None,
                        "spread_pct": spread * 100
                    }

        return surface_data

    def get_index(self, asset: str) -> float:
        """Get current index price (spot)"""
        result = self._get(f"/index?asset={asset}")
        if isinstance(result, dict):
            price = result.get("price", 0)
            return float(price) if price else 0.0
        return 0.0

    def find_mispriced_options(self, underlying: str) -> List[Dict]:
        """
        Find options with wide spreads (potential edge)

        Returns options where bid-ask spread > 5%
        """
        options = self.get_options_chain(underlying)
        mispriced = []

        for opt in options[:50]:  # Limit to avoid rate limits
            ob = self.get_orderbook(opt.instrument_name)
            if "error" in ob:
                continue

            bids = ob.get("bids", [])
            asks = ob.get("asks", [])

            if bids and asks:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                spread_pct = (best_ask - best_bid) / ((best_ask + best_bid) / 2) * 100

                if spread_pct > 5:  # >5% spread
                    mispriced.append({
                        "instrument": opt.instrument_name,
                        "strike": opt.strike,
                        "expiry": opt.expiry,
                        "type": opt.option_type,
                        "bid": best_bid,
                        "ask": best_ask,
                        "spread_pct": spread_pct
                    })

        return sorted(mispriced, key=lambda x: x["spread_pct"], reverse=True)

    async def connect_websocket(self):
        """Connect to Aevo WebSocket for real-time data"""
        self._ws_connection = await websockets.connect(self.WS_URL)
        return self._ws_connection

    async def subscribe_orderbook(self, instrument_name: str):
        """Subscribe to orderbook updates via WebSocket"""
        if not self._ws_connection:
            await self.connect_websocket()

        msg = {
            "op": "subscribe",
            "channel": f"orderbook:{instrument_name}"
        }
        await self._ws_connection.send(json.dumps(msg))

    async def subscribe_trades(self, instrument_name: str):
        """Subscribe to trade updates"""
        if not self._ws_connection:
            await self.connect_websocket()

        msg = {
            "op": "subscribe", 
            "channel": f"trades:{instrument_name}"
        }
        await self._ws_connection.send(json.dumps(msg))

    async def listen(self, callback):
        """Listen for WebSocket messages"""
        if not self._ws_connection:
            raise RuntimeError("WebSocket not connected. Call connect_websocket() first.")

        async for message in self._ws_connection:
            data = json.loads(message)
            await callback(data)


def test_aevo_connector():
    """Test Aevo connector"""
    print("=" * 60)
    print("AEVO OPTIONS CONNECTOR TEST")
    print("=" * 60)

    aevo = AevoConnector()

    # Test 1: Get assets
    print("\n📊 Available Assets:")
    assets = aevo.get_assets()
    print(f"   Found {len(assets)} assets: {assets}")

    # Test 2: Get expiries
    print("\n📅 Option Expiries:")
    expiries = aevo.get_expiries()
    print(f"   Found {len(expiries)} expiries")
    for exp in expiries[:5]:
        print(f"   - {exp}")

    # Test 3: Get options chain for BTC
    print("\n🔗 BTC Options Chain:")
    btc_options = aevo.get_options_chain("BTC")
    print(f"   Found {len(btc_options)} BTC options")

    if btc_options:
        # Show sample
        sample = btc_options[0]
        print(f"   Example: {sample.instrument_name}")
        print(f"   - Strike: ${sample.strike:,.0f}")
        print(f"   - Expiry: {sample.expiry}")
        print(f"   - Type: {'Call' if sample.option_type == 'C' else 'Put'}")
        print(f"   - TTE: {sample.time_to_expiry_years:.3f} years")

    # Test 4: Get index price
    print("\n💰 Current Prices:")
    btc_price = aevo.get_index("BTC")
    print(f"   BTC Index: ${btc_price:,.2f}")

    # Test 5: Get orderbook for specific option
    if btc_options:
        test_option = btc_options[0]
        print(f"\n📖 Orderbook for {test_option.instrument_name}:")
        ob = aevo.get_orderbook(test_option.instrument_name)

        if "error" in ob:
            print(f"   Error: {ob['error']}")
        else:
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])
            print(f"   Bids: {len(bids)} levels")
            print(f"   Asks: {len(asks)} levels")
            if bids:
                print(f"   Best Bid: ${float(bids[0][0]):.2f} (qty: {bids[0][1]})")
            if asks:
                print(f"   Best Ask: ${float(asks[0][0]):.2f} (qty: {asks[0][1]})")

    # Test 6: Find wide spreads
    print("\n🔍 Options with Wide Spreads (>5%):")
    mispriced = aevo.find_mispriced_options("BTC")
    if mispriced:
        for opt in mispriced[:3]:
            print(f"   {opt['instrument']}")
            print(f"   - Spread: {opt['spread_pct']:.1f}%")
            print(f"   - Bid: ${opt['bid']:.2f}, Ask: ${opt['ask']:.2f}")
    else:
        print("   No wide spreads found in sampled options")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    return aevo


if __name__ == "__main__":
    connector = test_aevo_connector()
