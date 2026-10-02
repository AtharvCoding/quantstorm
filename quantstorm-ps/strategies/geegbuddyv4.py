# Name: Atharv
# College: KJSCE, Mumbai
# Roll Number: 16010123088

import math
import random
from functools import lru_cache


@lru_cache(maxsize=None)
def _score_pmf(m: int) -> dict:
    """Exact PMF of a sum of m independent fair +/-1 coins."""
    if m <= 0:
        return {0: 1.0}
    denom = 1 << m
    return {
        x: math.comb(m, (x + m) // 2) / denom
        for x in range(-m, m + 1, 2)
    }


class Bot:
    name = "geegbuddyv4"

    # Deliberately aggressive research parameters.
    CLOSE_EARLY_BONUS = 1.00
    DIRECTIONAL_BONUS_CAP = 1.75
    BUY_RISK_THRESHOLD = -0.75
    SELL_RISK_THRESHOLD = -0.75

    # First-price auction shading.
    TE_SHADE = 0.72

    def reset(self, seat: int, config, seed: int) -> None:
        self.seat = seat
        self.config = config
        self.rng = random.Random(seed)

        # Current / most recent FORESIGHT sample.
        self.foresight_known = []
        self._last_ingested_round = 0
        self._last_seen_revealed = ()

    # ------------------------------------------------------------------
    # Information model
    # ------------------------------------------------------------------

    def _ingest(self, obs) -> None:
        """
        Update persistent information while avoiding Foresight
        double-counting.

        A non-prefix change in our revealed hand indicates a Transform
        swap, so stale Foresight information is discarded.
        """
        current_revealed = tuple(obs.my_revealed)

        if self._last_seen_revealed:
            prev = self._last_seen_revealed

            if current_revealed[:len(prev)] != prev:
                self.foresight_known = []
                self._last_ingested_round = 0

        self._last_seen_revealed = current_revealed

        # Store the latest leak rather than accumulating overlapping samples.
        if obs.foresight and obs.round != self._last_ingested_round:
            self.foresight_known = list(obs.foresight)
            self._last_ingested_round = obs.round

    def _belief(self, obs):
        """
        Return:
            center  = posterior mean of S
            unknown = number of score-contributing coins still unknown
            pmf     = exact residual PMF
        """
        self._ingest(obs)

        known = obs.k_mine + sum(self.foresight_known)

        unknown = max(
            0,
            self.config.N_COINS
            - len(obs.my_revealed)
            - len(self.foresight_known),
        )

        pmf = _score_pmf(unknown)

        return known, unknown, pmf

    def _directional_bonus(self, center: float, unknown: int) -> float:
        """
        Give an additional utility bonus to taking exposure in the
        direction supported by our posterior.
        """
        if abs(center) < 1e-9:
            return 0.0

        sigma = max(1.0, math.sqrt(unknown))
        z = min(1.5, abs(center) / sigma)

        return self.DIRECTIONAL_BONUS_CAP * (z / 1.5)

    def _contract_ev(
        self,
        direction: str,
        price: float,
        center: int,
        pmf: dict,
        has_substitute: bool,
        sub_cap: float,
    ) -> float:
        """Exact expected settlement PnL for one contract."""
        total = 0.0

        for residual, prob in pmf.items():
            score = center + residual

            raw = (
                score - price
                if direction == "long"
                else price - score
            )

            if has_substitute:
                raw = max(raw, -sub_cap)

            total += prob * raw

        return total

    # ------------------------------------------------------------------
    # Power valuation / auction policy
    # ------------------------------------------------------------------

    def _power_value(self, name: str, obs, offered) -> float:
        """
        Aggressive tick-value estimate used only to size TE bids.
        """

        round_no = obs.round
        rounds_left = self.config.N_ROUNDS - round_no + 1

        if name == "FORESIGHT":
            # Earlier information has more time to compound.
            return 2.00 + 0.20 * max(0, rounds_left - 2)

        if name == "SUBSTITUTE":
            center, unknown, pmf = self._belief(obs)

            sigma = math.sqrt(max(1, unknown))

            # Tail probability that a contract can fall more than 2 ticks
            # against us.
            tail = sum(
                p
                for x, p in pmf.items()
                if x < -2
            )

            directional = min(
                0.8,
                abs(center) / max(1.0, sigma),
            )

            return 1.35 + 0.90 * tail + 0.35 * directional

        if name == "TRICK_ROOM":
            value = 1.45

            if "STEALTH_ROCK" in obs.powers_mine:
                value += 1.20

            return value

        if name == "STEALTH_ROCK":
            # Persistent value increases with remaining rounds.
            future = max(0, self.config.N_ROUNDS - round_no)

            value = 1.15 + 0.45 * min(3, future)

            if "TRICK_ROOM" in obs.powers_mine:
                value += 1.20

            return value

        if name == "TRANSFORM":
            center = obs.k_mine

            if obs.foresight:
                other_visible = sum(obs.foresight)
                gap = abs(other_visible) - abs(center)

                return 0.75 + max(
                    0.0,
                    min(1.50, 0.30 * gap),
                )

            if abs(center) <= 1:
                return 1.25

            if abs(center) <= 2:
                return 0.65

            return 0.20

        return 0.0

    def bid(self, obs, offered: list) -> dict:
        if not offered or obs.te_mine <= 0:
            return {}

        self._ingest(obs)

        remaining_rounds = (
            self.config.N_ROUNDS - obs.round + 1
        )

        bids = {}

        for power in offered:
            value = self._power_value(
                power,
                obs,
                offered,
            )

            if value <= 0:
                continue

            # Convert tick value into TE-equivalent reservation value.
            reservation_te = (
                value / self.config.TE_SALVAGE
            )

            bid = int(
                reservation_te * self.TE_SHADE
            )

            # Deliberately aggressive ceilings.
            if power == "FORESIGHT":
                per_round_limit = 10

            elif power in (
                "STEALTH_ROCK",
                "TRICK_ROOM",
            ):
                per_round_limit = 8

            elif power == "SUBSTITUTE":
                per_round_limit = 7

            else:
                per_round_limit = 7

            # Keep some TE alive for later auctions.
            reserve = max(
                2,
                remaining_rounds - 1,
            )

            bid = min(
                bid,
                per_round_limit,
                max(0, obs.te_mine - reserve),
            )

            if bid > 0:
                bids[power] = bid

        # Defensive scaling if multiple powers are ever offered.
        total = sum(bids.values())

        if total > obs.te_mine:
            scale = obs.te_mine / total

            bids = {
                name: int(amount * scale)
                for name, amount in bids.items()
                if int(amount * scale) > 0
            }

        return bids

    # ------------------------------------------------------------------
    # Maker
    # ------------------------------------------------------------------

    def quote(self, obs) -> tuple:
        center, unknown, _ = self._belief(obs)

        # Aggressive regime:
        # use the tightest legal width.
        width = obs.final_cap

        bid = center - width // 2
        ask = center + (width - width // 2)

        return int(bid), int(ask)

    # ------------------------------------------------------------------
    # Taker / negotiation
    # ------------------------------------------------------------------

    def respond(self, obs, quote: tuple, turn: int):
        center, unknown, pmf = self._belief(obs)

        bid, ask = quote

        has_sub = (
            "SUBSTITUTE" in obs.powers_mine
        )

        sub_cap = float(
            self.config.POWERS["SUBSTITUTE"]["magnitude"]
        )

        long_ev = self._contract_ev(
            "long",
            ask,
            center,
            pmf,
            has_sub,
            sub_cap,
        )

        short_ev = self._contract_ev(
            "short",
            bid,
            center,
            pmf,
            has_sub,
            sub_cap,
        )

        direction_bonus = self._directional_bonus(
            center,
            unknown,
        )

        # Give extra utility to taking exposure in the direction
        # supported by our posterior.
        if center > 0:

            long_score = (
                long_ev
                + self.CLOSE_EARLY_BONUS
                + direction_bonus
            )

            short_score = (
                short_ev
                + self.CLOSE_EARLY_BONUS
            )

        elif center < 0:

            long_score = (
                long_ev
                + self.CLOSE_EARLY_BONUS
            )

            short_score = (
                short_ev
                + self.CLOSE_EARLY_BONUS
                + direction_bonus
            )

        else:

            long_score = (
                long_ev
                + self.CLOSE_EARLY_BONUS
            )

            short_score = (
                short_ev
                + self.CLOSE_EARLY_BONUS
            )

        # --------------------------------------------------------------
        # Final turn
        # --------------------------------------------------------------

        if turn == self.config.N_TURNS:

            midpoint = (bid + ask) // 2

            shift = 0

            if "TRICK_ROOM" in obs.powers_mine:
                shift += self.config.POWERS[
                    "TRICK_ROOM"
                ]["magnitude"]

            if "STEALTH_ROCK" in obs.powers_mine:
                shift += self.config.POWERS[
                    "STEALTH_ROCK"
                ]["magnitude"]

            force_ev = (
                self._contract_ev(
                    "short",
                    midpoint + shift,
                    center,
                    pmf,
                    has_sub,
                    sub_cap,
                )
                - self.config.FORCED_FILL_FEE
            )

            best_accept = max(
                long_score,
                short_score,
            )

            # Force only if it is materially better than closing now.
            if force_ev > best_accept + 0.35:

                current_width = ask - bid

                new_width = max(
                    obs.final_cap,
                    current_width
                    - self.config.MIN_REDUCTION,
                )

                new_bid = max(
                    bid,
                    min(
                        round(center),
                        ask - new_width,
                    ),
                )

                return (
                    "COUNTER",
                    new_bid,
                    new_bid + new_width,
                )

            return (
                "ACCEPT_BUY"
                if long_score >= short_score
                else "ACCEPT_SELL"
            )

        # --------------------------------------------------------------
        # Early-close regime
        # --------------------------------------------------------------

        buy_threshold = self.BUY_RISK_THRESHOLD
        sell_threshold = self.SELL_RISK_THRESHOLD

        preferred_long = center > 1
        preferred_short = center < -1

        # Aggressively take mildly negative trades in our preferred
        # directional regime.
        if (
            preferred_long
            and long_ev >= buy_threshold
            and long_score >= short_score
        ):
            return "ACCEPT_BUY"

        if (
            preferred_short
            and short_ev >= sell_threshold
            and short_score > long_score
        ):
            return "ACCEPT_SELL"

        # If either side is plainly positive, take it immediately.
        if (
            long_ev >= 0.0
            and long_ev >= short_ev
        ):
            return "ACCEPT_BUY"

        if (
            short_ev >= 0.0
            and short_ev > long_ev
        ):
            return "ACCEPT_SELL"

        # Otherwise make a large legal reduction toward our directional
        # center instead of wasting all six turns.
        current_width = ask - bid

        reduction = max(
            self.config.MIN_REDUCTION,
            current_width // 2,
        )

        new_width = max(
            obs.final_cap,
            current_width - reduction,
        )

        new_bid = max(
            bid,
            min(
                round(center),
                ask - new_width,
            ),
        )

        return (
            "COUNTER",
            new_bid,
            new_bid + new_width,
        )

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def use_transform(self, obs) -> bool:
        self._ingest(obs)

        my_abs = abs(obs.k_mine)

        if obs.foresight:

            other_abs = abs(
                sum(obs.foresight)
            )

            # Aggressive threshold:
            # take the opponent's materially stronger visible hand.
            return other_abs >= my_abs + 2

        # Without Foresight, transform only when our current hand
        # is basically uninformative.
        return my_abs <= 1