"""
APEX CorrRank Selector
======================
Portfolio correlation ranking for top-1 selection.

Selects the best asset by risk-adjusted edge,
accounting for correlation to reduce concentration risk.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AssetScore:
    """Asset score with correlation adjustment."""
    asset: str
    edge_score: float
    risk_score: float
    correlation_penalty: float
    final_score: float
    rank: int

    def to_dict(self) -> Dict:
        return {
            'asset': self.asset,
            'edge_score': self.edge_score,
            'risk_score': self.risk_score,
            'correlation_penalty': self.correlation_penalty,
            'final_score': self.final_score,
            'rank': self.rank
        }


class CorrRankSelector:
    """
    Correlation-based asset ranking.

    Selects top assets while accounting for correlations
to avoid concentrated exposure to similar moves.
    """

    def __init__(self, 
                 correlation_threshold: float = 0.7,
                 risk_weight: float = 0.3,
                 edge_weight: float = 0.7):
        """
        Args:
            correlation_threshold: Min correlation for penalty
            risk_weight: Weight for risk score
            edge_weight: Weight for edge score
        """
        self.correlation_threshold = correlation_threshold
        self.risk_weight = risk_weight
        self.edge_weight = edge_weight
        self.history: List[AssetScore] = []

    def calculate_correlation_matrix(self, 
                                     returns_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Calculate correlation matrix from returns.

        Args:
            returns_dict: Dict of asset -> returns series

        Returns:
            Correlation DataFrame
        """
        # Align all series
        df = pd.DataFrame(returns_dict)
        return df.corr().fillna(0)

    def calculate_risk_score(self, 
                            returns: pd.Series) -> float:
        """
        Calculate risk-adjusted score.

        Lower volatility = higher score
        """
        if len(returns) < 10:
            return 0.5

        vol = returns.std()
        sharpe = returns.mean() / vol if vol > 0 else 0

        # Normalize to 0-1
        score = min(max(sharpe + 1, 0), 2) / 2
        return score

    def calculate_correlation_penalty(self,
                                     asset: str,
                                     correlation_matrix: pd.DataFrame,
                                     selected_assets: List[str]) -> float:
        """
        Calculate correlation penalty for an asset.

        Higher correlation with selected assets = higher penalty
        """
        if not selected_assets or asset not in correlation_matrix.columns:
            return 0.0

        # Get correlations with selected assets
        corrs = correlation_matrix.loc[asset, selected_assets]

        # Penalty for high correlations
        penalties = [c - self.correlation_threshold for c in corrs if c > self.correlation_threshold]

        if penalties:
            return np.mean(penalties)
        return 0.0

    def rank_assets(self,
                   edge_scores: Dict[str, float],
                   returns_dict: Dict[str, pd.Series],
                   top_n: int = 5) -> List[AssetScore]:
        """
        Rank assets by risk-adjusted edge with correlation penalty.

        Args:
            edge_scores: Dict of asset -> edge score
            returns_dict: Dict of asset -> returns series
            top_n: Number of top assets to return

        Returns:
            List of AssetScore (sorted by rank)
        """
        # Calculate correlation matrix
        correlation_matrix = self.calculate_correlation_matrix(returns_dict)

        scores = []
        selected = []

        # First pass: calculate base scores
        for asset, edge in edge_scores.items():
            if asset not in returns_dict:
                continue

            returns = returns_dict[asset]

            # Calculate components
            risk_score = self.calculate_risk_score(returns)
            corr_penalty = self.calculate_correlation_penalty(
                asset, correlation_matrix, selected
            )

            # Combined score
            base_score = (self.edge_weight * edge + 
                         self.risk_weight * risk_score)

            final_score = base_score * (1 - corr_penalty * 0.5)

            scores.append({
                'asset': asset,
                'edge_score': edge,
                'risk_score': risk_score,
                'correlation_penalty': corr_penalty,
                'final_score': final_score
            })

            selected.append(asset)

        # Sort by final score
        scores.sort(key=lambda x: x['final_score'], reverse=True)

        # Create AssetScore objects with ranks
        results = []
        for i, s in enumerate(scores[:top_n]):
            results.append(AssetScore(
                asset=s['asset'],
                edge_score=s['edge_score'],
                risk_score=s['risk_score'],
                correlation_penalty=s['correlation_penalty'],
                final_score=s['final_score'],
                rank=i+1
            ))

        self.history.extend(results)
        return results

    def select_top(self,
                  edge_scores: Dict[str, float],
                  returns_dict: Dict[str, pd.Series]) -> Optional[AssetScore]:
        """
        Select top-1 asset.

        Returns:
            Top AssetScore or None if no valid assets
        """
        ranked = self.rank_assets(edge_scores, returns_dict, top_n=1)
        return ranked[0] if ranked else None

    def get_selection_history(self, 
                             asset: str = None) -> List[AssetScore]:
        """Get selection history."""
        if asset:
            return [s for s in self.history if s.asset == asset]
        return self.history

    def get_summary(self) -> Dict:
        """Get selector summary."""
        if not self.history:
            return {}

        assets = set(s.asset for s in self.history)
        return {
            'total_selections': len(self.history),
            'unique_assets': len(assets),
            'top_assets': [s.asset for s in sorted(
                {a: max((s.final_score for s in self.history if s.asset == a), default=0) 
                 for a in assets}.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]]
        }


if __name__ == "__main__":
    print("APEX CorrRank Selector Demo")
    print("=" * 60)

    selector = CorrRankSelector()

    # Generate synthetic data for 5 assets
    np.random.seed(42)
    assets = ['ETH', 'BTC', 'SOL', 'AVAX', 'MATIC']
    returns_dict = {}

    # Create correlated returns
    common_factor = np.random.randn(100) * 0.01
    for asset in assets:
        specific = np.random.randn(100) * 0.015
        returns_dict[asset] = pd.Series(common_factor * 0.7 + specific)

    # Edge scores (from signal engine)
    edge_scores = {
        'ETH': 0.75,
        'BTC': 0.65,
        'SOL': 0.55,
        'AVAX': 0.45,
        'MATIC': 0.35
    }

    print(f"\nAsset Rankings:")
    ranked = selector.rank_assets(edge_scores, returns_dict, top_n=3)

    for score in ranked:
        print(f"  #{score.rank} {score.asset}:")
        print(f"     Edge: {score.edge_score:.2f}")
        print(f"     Risk: {score.risk_score:.2f}")
        print(f"     Corr Penalty: {score.correlation_penalty:.3f}")
        print(f"     Final: {score.final_score:.3f}")

    top = selector.select_top(edge_scores, returns_dict)
    print(f"\nTop Selection: {top.asset} (score: {top.final_score:.3f})")
