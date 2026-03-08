"""
Premia Subgraph Connector
===========================
Query decentralized options data from Premia V3.

IMPORTANT: The Graph hosted service has been deprecated.
Subgraphs are now on the decentralized network or Arbitrum-specific endpoints.

The Graph Explorer: https://thegraph.com/explorer/subgraphs/3nXfK3RbFrj6mhkGdoKRowEEti2WvmUdxmz73tben6Mb

Premia Docs: https://docs.premia.blue/developer-center/apis/subgraph-api
"""

import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger('PremiaConnector')


@dataclass
class OptionMarket:
    """Premia option market data."""
    id: str
    asset: str
    strike: float
    expiry: datetime
    option_type: str  # 'call' or 'put'
    is_buy: bool      # True = buy side, False = sell side
    iv: float         # Implied volatility
    price: float      # Option price
    bid: float        # Best bid
    ask: float        # Best ask
    liquidity: float  # Pool liquidity


@dataclass
class VolSurfacePoint:
    """Volatility surface data point."""
    strike: float
    expiry_days: int
    iv: float
    price: float
    delta: Optional[float] = None


class PremiaSubgraphConnector:
    """
    Connect to Premia V3 subgraph for options data.

    NOTE: Hosted service deprecated. Use decentralized network or containerized API.

    Premia offers:
    1. Subgraph API (TheGraph decentralized network - requires API key)
    2. Containerized API ("all-in-one" solution for trading)
    3. Direct contract interaction (ethers.js/web3.py)

    Docs: https://docs.premia.blue/developer-center/apis
    """

    # Deprecated hosted service URL (no longer works)
    DEPRECATED_URL = "https://api.thegraph.com/subgraphs/name/premiafinance/premia-v3-arbitrum"

    # Alternative: Arbitrum Goerli testnet (for development)
    TESTNET_URL = "https://api.thegraph.com/subgraphs/name/premian-labs/premia-blue"

    # Recommended: Containerized API (no key needed, direct access)
    # Contact Premia for containerized API access
    CONTAINERIZED_API = "https://api.premia.blue/containerized"  # Example

    # Decentralized network (requires paid API key from TheGraph)
    DECENTRALIZED_NETWORK_URL = "https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/3nXfK3RbFrj6mhkGdoKRowEEti2WvmUdxmz73tben6Mb"

    def __init__(self, api_key: Optional[str] = None, use_testnet: bool = False):
        self.session = requests.Session()
        self.api_key = api_key

        if api_key:
            self.subgraph_url = self.DECENTRALIZED_NETWORK_URL.format(API_KEY=api_key)
        elif use_testnet:
            self.subgraph_url = self.TESTNET_URL
        else:
            # Will fail - need valid endpoint
            self.subgraph_url = self.DEPRECATED_URL

    def _query(self, query: str, variables: Optional[Dict] = None) -> Dict:
        """Execute GraphQL query."""
        try:
            response = self.session.post(
                self.subgraph_url,
                json={"query": query, "variables": variables or {}},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            if 'errors' in data:
                logger.error(f"GraphQL errors: {data['errors']}")
                return {}
            return data.get('data', {})
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {}

    def check_status(self) -> Dict:
        """Check if the subgraph endpoint is working."""
        test_query = "{ _meta { block { number } } }"
        result = self._query(test_query)
        return {
            'working': bool(result),
            'url': self.subgraph_url,
            'requires_api_key': not self.api_key and self.subgraph_url == self.DEPRECATED_URL
        }

    def get_pools(self, asset: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get options pools from Premia."""
        where_clause = f'where: {{base: "{asset}"}}' if asset else ''

        query = f"""
        {{
            pools(
                first: {limit}
                orderBy: totalValueLocked
                orderDirection: desc
                {where_clause}
            ) {{
                id
                base
                quote
                isCall
                strike
                maturity
                totalValueLocked
                totalVolume
                basePrice
                lastPrice
                impliedVolatility
                token0 {{ symbol }}
                token1 {{ symbol }}
            }}
        }}
        """

        result = self._query(query)
        return result.get('pools', [])

    def get_vol_surface(self, asset: str) -> List[VolSurfacePoint]:
        """Build volatility surface for an asset."""
        pools = self.get_pools(asset, limit=1000)

        surface = []
        now = datetime.now()

        for pool in pools:
            if not pool.get('impliedVolatility'):
                continue

            expiry = datetime.fromtimestamp(int(pool['maturity']))
            days_to_expiry = (expiry - now).days

            if days_to_expiry <= 0:
                continue

            surface.append(VolSurfacePoint(
                strike=float(pool['strike']) / 1e18,
                expiry_days=days_to_expiry,
                iv=float(pool['impliedVolatility']) / 1e4,
                price=float(pool.get('lastPrice', 0)) / 1e18
            ))

        return sorted(surface, key=lambda x: (x.expiry_days, x.strike))

    def get_alternative_access_methods(self) -> Dict:
        """
        Return available access methods since hosted service is deprecated.

        Options:
        1. Containerized API (Premia-hosted, similar to CEX API)
        2. TheGraph decentralized network (requires paid API key)
        3. Direct contract interaction (ethers.js/web3.py)
        4. Premia SDK
        """
        return {
            'containerized_api': {
                'description': 'Premia-hosted "all-in-one" API for trading',
                'pros': ['No blockchain knowledge needed', 'Fast', 'Similar to CEX'],
                'cons': ['Requires Premia access', 'Centralized'],
                'docs': 'https://docs.premia.blue/developer-center/apis/containerized-api'
            },
            'decentralized_network': {
                'description': 'TheGraph decentralized network with paid API key',
                'pros': ['Decentralized', 'Reliable', 'Standard GraphQL'],
                'cons': ['Requires API key (~$50-200/month)', 'Paid service'],
                'docs': 'https://thegraph.com/studio/'
            },
            'direct_contracts': {
                'description': 'Direct smart contract interaction',
                'pros': ['No API needed', 'Fully decentralized', 'Free'],
                'cons': ['Requires Web3 knowledge', 'Slower', 'Gas costs'],
                'docs': 'https://docs.premia.blue/developer-center/contracts'
            },
            'premia_sdk': {
                'description': 'Official Premia SDK',
                'pros': ['Easy to use', 'TypeScript support'],
                'cons': ['Limited to TS/JS', 'May still need RPC'],
                'docs': 'https://docs.premia.blue/developer-center/sdk'
            }
        }


def main():
    """Test Premia connector status."""
    print("=" * 60)
    print("PREMIA SUBGRAPH CONNECTOR STATUS")
    print("=" * 60)

    # Test with deprecated endpoint (will fail)
    print("\n--- Testing deprecated hosted service ---")
    connector = PremiaSubgraphConnector()
    status = connector.check_status()
    print(f"Status: {'✅ Working' if status['working'] else '❌ FAILED'}")
    print(f"Requires API key: {status['requires_api_key']}")

    if not status['working']:
        print("\n⚠️  The Graph hosted service has been deprecated.")
        print("\nAlternative access methods:")

        alternatives = connector.get_alternative_access_methods()
        for name, info in alternatives.items():
            print(f"\n{name.upper().replace('_', ' ')}:")
            print(f"  Description: {info['description']}")
            print(f"  Pros: {', '.join(info['pros'])}")
            print(f"  Cons: {', '.join(info['cons'])}")
            print(f"  Docs: {info['docs']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
