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
    name = "InfoAwareCloserV1"

    # ------------------------------------------------------------------
    # Tunables. TODO: sweep these against the backtester rather than
    # trusting the starting values.
    # ------------------------------------------------------------------
    CLOSE_EARLY_BONUS = 0.30   # tick bonus assigned to accepting now
    TE_SHADE = 0.5             # fraction of reservation price we actually bid

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
    # Power valuation (ticks), conditioned on what we already hold.
    # TODO: these baselines are reasoned estimates, not backtested
    # numbers -- validate them once the bot can actually play matches.
    # ------------------------------------------------------------------

    def _power_value(self, name, obs, offered):
        if name == "FORESIGHT":
            rounds_left = self.config.N_ROUNDS - obs.round + 1
            return 1.1 + 0.15 * rounds_left  # compounds: earlier is worth more

        if name == "SUBSTITUTE":
            k_known, m = self._k_and_m(obs)
            pmf = _score_pmf(m)
            cap = self.config.POWERS["SUBSTITUTE"]["magnitude"]
            d = obs.final_cap / 2.0  # assumed typical aggressiveness this round
            return sum(p * max(0.0, (d - cap) - x) for x, p in pmf.items())

        if name == "STEALTH_ROCK":
            rounds_left_after = max(0, self.config.N_ROUNDS - obs.round)
            value = 0.4 * rounds_left_after  # persistent -> more rounds to pay off
            if "TRICK_ROOM" in obs.powers_mine:
                value += 1.5  # stack clears the forcing fee
            return value

        if name == "TRICK_ROOM":
            value = 0.5  # single round, modest solo value
            if "STEALTH_ROCK" in obs.powers_mine:
                value += 1.5
            return value

        if name == "TRANSFORM":
            if "FORESIGHT" in obs.powers_mine or "FORESIGHT" in offered:
                return 1.2  # peek-then-decide combo
            return 0.3  # denial value only, decision would be blind

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