# Name: Atharv
# College: KJSCE, Mumbai
# Roll Number: 16010123088

import math
import random
from functools import lru_cache


# ════════════════════════════════════════════════════════════════════
#  Pure math -- module level on purpose. This caches on `m` (an int)
#  only, never on anything from a specific deal, so sharing it across
#  Bot instances (and across deals) leaks nothing and just saves time.
# ════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=None)
def _score_pmf(m: int) -> dict:
    """P(X = x) for X = sum of m independent fair +/-1 coins."""
    if m <= 0:
        return {0: 1.0}
    denom = 1 << m
    return {x: math.comb(m, (x + m) // 2) / denom for x in range(-m, m + 1, 2)}


class Bot:
    name = "GeegBuddyV3"

    # ------------------------------------------------------------------
    # V1 research priors. These are intentionally explicit so they can be
    # swept later; they are not claimed to be optimal.
    # ------------------------------------------------------------------
    CLOSE_EARLY_BONUS = 0.10   # small tie-break for resolving early
    TE_SHADE = 0.50            # first-price auction bid shading

    def reset(self, seat: int, config, seed: int) -> None:
        self.seat = seat
        self.config = config
        self.rng = random.Random(seed)

        # The one piece of state the engine will NOT hand back to us.
        self.foresight_known = []
        self._last_ingested_round = 0
        self._last_seen_revealed = ()  # for detecting a TRANSFORM swap

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _ingest(self, obs) -> None:
        """Pull this round's FORESIGHT leak (if any) into permanent memory,
        and detect a TRANSFORM swap (ours or the opponent's) so stale
        opponent-leak data doesn't get folded into the new hand.

        Two correctness traps this guards against:

          1. FORESIGHT resamples the opponent's FULL revealed set from
             scratch every round (no memory of prior draws). Naively
             extending a running list double-counts coins seen more than
             once. Since rounds 1-4 always hand back the COMPLETE set
             (n_sample == len(opp_revealed) whenever 4*round <= 16), each
             round's leak supersedes the last -- so we REPLACE, keeping
             whichever is larger, instead of accumulating.

          2. apply_transform() swaps BOTH hands unconditionally, whoever
             fired it. If either side swaps, our own obs.my_revealed
             stops being an extension of what we saw last call -- that's
             the only way it can happen, since normal reveals only
             append. Any previously stored opponent-leak values are now
             either already folded into our new k_mine or about a hand
             that no longer exists -- either way, stale.
        """
        cur = tuple(obs.my_revealed)
        if self._last_seen_revealed and cur[: len(self._last_seen_revealed)] != self._last_seen_revealed:
            self.foresight_known = []
        self._last_seen_revealed = cur

        if not obs.foresight or obs.round == self._last_ingested_round:
            return
        self._last_ingested_round = obs.round
        leak = list(obs.foresight)
        if len(leak) >= len(self.foresight_known):
            self.foresight_known = leak

    def _k_and_m(self, obs):
        """Our current posterior: k_known + m unseen coins."""
        k_known = obs.k_mine + sum(self.foresight_known)
        m = self.config.N_COINS - len(obs.my_revealed) - len(self.foresight_known)
        return k_known, max(0, m)

    def _contract_ev(self, direction, price, k_known, pmf, has_substitute, cap):
        """Exact expected PnL of holding one side of a contract at `price`,
        given our posterior pmf over the residual X (S = k_known + X).
        direction: 'long' or 'short'. SUBSTITUTE floors the loss at -cap.
        """
        total = 0.0
        for x, p in pmf.items():
            s = k_known + x
            raw = (s - price) if direction == "long" else (price - s)
            if has_substitute:
                raw = max(raw, -cap)
            total += p * raw
        return total

    # ------------------------------------------------------------------
    # Power valuation (ticks).
    # Start from the measured harness averages shipped with the starter
    # materials, then apply only simple state-aware adjustments where the
    # mechanics justify them. These are reservation values, not bids.
    # ------------------------------------------------------------------

    def _power_value(self, name, obs, offered):
        # Measured reference averages from the shipped starter materials.
        # We use them as conservative priors, then make the obvious
        # state-dependent adjustments rather than inventing a round-linear
        # formula.
        if name == "FORESIGHT":
            base = 1.22
            # Earlier information has more chances to affect later rounds.
            rounds_left = self.config.N_ROUNDS - obs.round
            return base + 0.12 * rounds_left

        if name == "TRICK_ROOM":
            base = 1.37
            # A second forced-fill shift makes the already-owned shift power
            # more useful, because the effects stack on forced fills.
            if "STEALTH_ROCK" in obs.powers_mine:
                base += 1.00
            return base

        if name == "SUBSTITUTE":
            # Start from the measured average, then scale up when our current
            # posterior makes the round's contract particularly risky.
            k_known, m = self._k_and_m(obs)
            pmf = _score_pmf(m)
            cap = self.config.POWERS["SUBSTITUTE"]["magnitude"]
            # Approximate the probability of a >=2-tick loss on a typical
            # floor-width acceptance around our current estimate.
            p_bad = 0.0
            for x, p in pmf.items():
                if x < -cap:
                    p_bad += p
            risk_multiplier = 1.0 + 0.75 * p_bad
            return 1.07 * risk_multiplier

        if name == "STEALTH_ROCK":
            base = 1.47
            rounds_left_after = self.config.N_ROUNDS - obs.round
            # Its persistence makes it less attractive near the end.
            time_factor = rounds_left_after / 3.0 if rounds_left_after else 0.0
            value = base * min(1.0, max(0.0, time_factor))
            if "TRICK_ROOM" in obs.powers_mine:
                value += 1.00
            return value

        if name == "TRANSFORM":
            # No fixed magnitude is defensible. Use a conservative denial
            # value, then add value only when our current visible hand is
            # weak enough that swapping information could materially help.
            flat = abs(obs.k_mine) <= 1
            value = 0.30
            if flat:
                value += 0.70
            if "FORESIGHT" in obs.powers_mine:
                value += 0.25
            return value

        return 0.0

    # ------------------------------------------------------------------
    # Required methods
    # ------------------------------------------------------------------

    def bid(self, obs, offered: list) -> dict:
        if not offered or obs.te_mine <= 0:
            return {}
        self._ingest(obs)

        remaining_rounds = self.config.N_ROUNDS - obs.round + 1
        # Don't blow the whole budget on one round; allow up to ~2x an
        # even per-round share so FORESIGHT can be pressed hard early.
        round_cap = max(1, int(obs.te_mine * 2 / max(1, remaining_rounds)))

        bids = {}
        for name in offered:
            value = self._power_value(name, obs, offered)
            if value <= 0:
                continue
            reservation = value / self.config.TE_SALVAGE
            shaded = reservation * self.TE_SHADE
            amt = int(min(shaded, round_cap, obs.te_mine))
            if amt > 0:
                bids[name] = amt

        total = sum(bids.values())
        if total > obs.te_mine:
            scale = obs.te_mine / total
            bids = {k: int(v * scale) for k, v in bids.items() if int(v * scale) > 0}
        return bids

    def quote(self, obs) -> tuple:
        self._ingest(obs)
        k_known, m = self._k_and_m(obs)

        best_w, best_score = obs.final_cap, None
        for w in range(obs.final_cap, obs.spread_cap + 1):
            p_true = self.config.straddle_prob(obs.round, w, unseen=m)
            p_base = self.config.straddle_prob(obs.round, w)
            edge = self.config.MAKER_OBLIGATION * (p_true - p_base)
            premium = self.config.WIDTH_PREMIUM * max(0, w - obs.final_cap)
            score = edge - premium
            if best_score is None or score > best_score:
                best_score, best_w = score, w

        v, w = k_known, best_w
        return (v - w // 2, v + (w - w // 2))

    def respond(self, obs, quote: tuple, turn: int):
        self._ingest(obs)
        k_known, m = self._k_and_m(obs)
        bid, ask = quote
        pmf = _score_pmf(m)
        has_sub = "SUBSTITUTE" in obs.powers_mine
        sub_cap = self.config.POWERS["SUBSTITUTE"]["magnitude"]

        buy_ev = self._contract_ev("long", ask, k_known, pmf, has_sub, sub_cap) + self.CLOSE_EARLY_BONUS
        sell_ev = self._contract_ev("short", bid, k_known, pmf, has_sub, sub_cap) + self.CLOSE_EARLY_BONUS

        if turn == self.config.N_TURNS:
            mid = (bid + ask) // 2
            shift = 0
            if "TRICK_ROOM" in obs.powers_mine:
                shift += self.config.POWERS["TRICK_ROOM"]["magnitude"]
            if "STEALTH_ROCK" in obs.powers_mine:
                shift += self.config.POWERS["STEALTH_ROCK"]["magnitude"]
            force_ev = (
                self._contract_ev("short", mid + shift, k_known, pmf, has_sub, sub_cap)
                - self.config.FORCED_FILL_FEE
            )

            options = {"ACCEPT_BUY": buy_ev, "ACCEPT_SELL": sell_ev, "FORCE": force_ev}
            best = max(options, key=options.get)
            if best == "ACCEPT_BUY":
                return "ACCEPT_BUY"
            if best == "ACCEPT_SELL":
                return "ACCEPT_SELL"
            # FORCE wins: any legal counter triggers it, values are moot.
            cur_w = ask - bid
            new_w = max(obs.final_cap, cur_w - self.config.MIN_REDUCTION)
            center = max(bid, min(round(k_known), ask - new_w))
            return ("COUNTER", center, center + new_w)

        # Non-final turn: accept if it clears zero, else counter hard
        # toward our own estimate (aggressive closer -> shrink by more
        # than the minimum so rounds resolve fast).
        if buy_ev > 0 and buy_ev >= sell_ev:
            return "ACCEPT_BUY"
        if sell_ev > 0 and sell_ev > buy_ev:
            return "ACCEPT_SELL"

        cur_w = ask - bid
        shrink = max(self.config.MIN_REDUCTION, cur_w // 2)
        new_w = max(obs.final_cap, cur_w - shrink)
        center = max(bid, min(round(k_known), ask - new_w))
        return ("COUNTER", center, center + new_w)

    def use_transform(self, obs) -> bool:
        self._ingest(obs)
        if obs.foresight:
            # We're peeking this round -- an informed decision, not a guess.
            their_visible_sum = sum(obs.foresight)
            return abs(their_visible_sum) > abs(obs.k_mine) + 1
        # Blind decision: only worth firing if our own hand is flat,
        # per the "extreme hand = keep, flat hand = nothing to lose" logic.
        return abs(obs.k_mine) <= 1