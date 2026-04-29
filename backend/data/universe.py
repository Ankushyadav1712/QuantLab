from __future__ import annotations

from config import SECTOR_MAP, UNIVERSE


class UniverseManager:
    def __init__(
        self,
        tickers: list[str] | None = None,
        sector_map: dict[str, str] | None = None,
    ):
        self._tickers = list(tickers if tickers is not None else UNIVERSE)
        self._sector_map = dict(sector_map if sector_map is not None else SECTOR_MAP)

    def get_tickers(self) -> list[str]:
        return list(self._tickers)

    def get_sector(self, ticker: str) -> str:
        try:
            return self._sector_map[ticker]
        except KeyError as exc:
            raise KeyError(f"Unknown ticker: {ticker}") from exc
