"""Chunk -> fixed-length feature vector for Poker44 bot detection.

CRITICAL: this module is imported by BOTH the trainer and the live miner, so
features are identical at train and serve time (no skew). The benchmark API and
the validator both deliver hands in the same sanitized, miner-visible form:
    - metadata.bb == 0.02, sb == 0.01, hole_cards == null, board_cards == []
    - outcome zeroed, seats re-aliased, actions down-sampled to ~5-12/hand
    - NO decision-time fields  -> timing features are impossible, do not invent them

A "group" (== one entry of DetectionSynapse.chunks) is the set of hands for ONE
labeled player, the "hero". `metadata.hero_seat` marks the hero's aliased seat in
each hand, so we isolate the hero's own actions -- the strongest available signal.

Units: `normalized_amount_bb` is size in big blinds (scale-free); `amount`,
`pot_before`, `pot_after`, `raise_to` are visible chip units, so ratios between
them are unit-free (bb cancels).
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

STREETS = ("preflop", "flop", "turn", "river")
ACTION_TYPES = ("fold", "check", "call", "bet", "raise")
_AGGR_FACTOR_CAP = 20.0


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _f(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _stats(values: Sequence[float], prefix: str) -> Dict[str, float]:
    keys = ("mean", "std", "med", "p25", "p75", "max", "cv")
    arr = [float(v) for v in values]
    if not arr:
        return {f"{prefix}_{k}": 0.0 for k in keys}
    n = len(arr)
    srt = sorted(arr)
    mean = sum(arr) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in arr) / n)

    def quant(p: float) -> float:
        if n == 1:
            return srt[0]
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        return srt[lo] + (srt[hi] - srt[lo]) * (idx - lo)

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_med": quant(0.5),
        f"{prefix}_p25": quant(0.25),
        f"{prefix}_p75": quant(0.75),
        f"{prefix}_max": max(arr),
        f"{prefix}_cv": _safe_div(std, abs(mean)),  # low CV == robotic sizing
    }


def _mean_std(values: Sequence[float], prefix: str) -> Dict[str, float]:
    arr = [float(v) for v in values]
    if not arr:
        return {f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0}
    n = len(arr)
    mean = sum(arr) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in arr) / n)
    return {f"{prefix}_mean": mean, f"{prefix}_std": std}


def _entropy(counts: Sequence[float]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    if len(probs) <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in probs)
    return h / math.log(len(counts))  # normalized to [0, 1]


def _feature_dict(group: List[dict]) -> Dict[str, float]:
    hands = [h for h in (group or []) if isinstance(h, dict)]
    n_hands = len(hands)

    hero_type_counts = {t: 0 for t in ACTION_TYPES}
    hero_actions_total = 0
    all_actions_total = 0

    hero_size_bb: List[float] = []      # normalized_amount_bb for bet/raise
    hero_bet_to_pot: List[float] = []   # amount / pot_before
    hero_raise_to_pot: List[float] = [] # raise_to / pot_before

    players_per_hand: List[float] = []
    hero_actions_per_hand: List[float] = []
    all_actions_per_hand: List[float] = []
    pot_before_vals: List[float] = []
    pot_after_vals: List[float] = []

    street_hero_counts = {s: 0 for s in STREETS}
    street_hero_aggr = {s: 0 for s in STREETS}

    per_hand_aggr: List[float] = []
    per_hand_fold: List[float] = []
    hands_reaching = {s: 0 for s in STREETS}
    vpip_hands = 0
    pfr_hands = 0

    for hand in hands:
        meta = hand.get("metadata") if isinstance(hand.get("metadata"), dict) else {}
        hero_seat = meta.get("hero_seat")
        players = hand.get("players") if isinstance(hand.get("players"), list) else []
        players_per_hand.append(len(players))

        actions = [a for a in (hand.get("actions") or []) if isinstance(a, dict)]
        all_actions_total += len(actions)
        all_actions_per_hand.append(len(actions))
        for a in actions:
            pot_before_vals.append(_f(a.get("pot_before")))
            pot_after_vals.append(_f(a.get("pot_after")))

        hero_actions = [a for a in actions if a.get("actor_seat") == hero_seat]
        hero_actions_per_hand.append(len(hero_actions))

        h_hero = 0
        h_aggr = 0
        h_fold = 0
        vpip = False
        pfr = False
        streets_seen = set()
        for a in hero_actions:
            at = str(a.get("action_type") or "").lower()
            if at not in hero_type_counts:
                continue
            hero_type_counts[at] += 1
            hero_actions_total += 1
            h_hero += 1

            street = str(a.get("street") or "preflop").lower()
            if street not in street_hero_counts:
                street = "preflop"
            street_hero_counts[street] += 1
            streets_seen.add(street)

            size_bb = _f(a.get("normalized_amount_bb"))
            pot_b = _f(a.get("pot_before"))
            amount = _f(a.get("amount"))
            raise_to = a.get("raise_to")

            if at == "fold":
                h_fold += 1
            if at in ("call", "bet", "raise") and street == "preflop":
                vpip = True
            if at == "raise" and street == "preflop":
                pfr = True
            if at in ("bet", "raise"):
                street_hero_aggr[street] += 1
                h_aggr += 1
                if size_bb > 0:
                    hero_size_bb.append(size_bb)
                if pot_b > 0 and amount > 0:
                    hero_bet_to_pot.append(amount / pot_b)
                if pot_b > 0 and raise_to is not None:
                    rt = _f(raise_to)
                    if rt > 0:
                        hero_raise_to_pot.append(rt / pot_b)

        for s in streets_seen:
            hands_reaching[s] += 1
        if h_hero > 0:
            per_hand_aggr.append(h_aggr / h_hero)
            per_hand_fold.append(h_fold / h_hero)
        if vpip:
            vpip_hands += 1
        if pfr:
            pfr_hands += 1

    bet = hero_type_counts["bet"]
    raise_ = hero_type_counts["raise"]
    call = hero_type_counts["call"]
    check = hero_type_counts["check"]
    aggr_num = bet + raise_
    aggr_den = bet + raise_ + call + check

    feats: Dict[str, float] = {}

    # --- volume ---
    feats["n_hands"] = float(n_hands)
    feats.update(_mean_std(hero_actions_per_hand, "heroact_per_hand"))
    feats["allact_per_hand_mean"] = _safe_div(all_actions_total, n_hands)
    feats["hero_share"] = _safe_div(hero_actions_total, all_actions_total)

    # --- hero action-type rates ---
    for t in ACTION_TYPES:
        feats[f"htr_{t}"] = _safe_div(hero_type_counts[t], hero_actions_total)

    # --- aggression ---
    feats["aggr_freq"] = _safe_div(aggr_num, aggr_den)
    feats["aggr_factor"] = min(_safe_div(aggr_num, call) if call else float(aggr_num), _AGGR_FACTOR_CAP)
    feats["action_entropy"] = _entropy([hero_type_counts[t] for t in ACTION_TYPES])

    # --- preflop selectivity ---
    feats["vpip_rate"] = _safe_div(vpip_hands, n_hands)
    feats["pfr_rate"] = _safe_div(pfr_hands, n_hands)
    feats["pfr_over_vpip"] = _safe_div(pfr_hands, vpip_hands)

    # --- continuation (fraction of hands hero acts on street) ---
    feats["cont_flop"] = _safe_div(hands_reaching["flop"], n_hands)
    feats["cont_turn"] = _safe_div(hands_reaching["turn"], n_hands)
    feats["cont_river"] = _safe_div(hands_reaching["river"], n_hands)

    # --- per-street aggression & share of hero actions ---
    for s in STREETS:
        feats[f"saggr_{s}"] = _safe_div(street_hero_aggr[s], street_hero_counts[s])
        feats[f"sshare_{s}"] = _safe_div(street_hero_counts[s], hero_actions_total)

    # --- bet sizing (the discipline / robotic tells) ---
    feats.update(_stats(hero_size_bb, "betbb"))
    feats.update(_stats(hero_bet_to_pot, "bet2pot"))
    feats.update(_stats(hero_raise_to_pot, "raise2pot"))

    # --- table structure ---
    feats.update(_mean_std(players_per_hand, "nplayers"))
    feats.update(_mean_std(pot_before_vals, "potbefore"))
    feats["potafter_mean"] = _safe_div(sum(pot_after_vals), len(pot_after_vals))

    # --- cross-hand consistency (bots are more uniform) ---
    feats.update(_mean_std(per_hand_aggr, "perhand_aggr"))
    feats.update(_mean_std(per_hand_fold, "perhand_fold"))

    return feats


# --- v3 extension: sizing-discreteness, pot-pressure reactions, per-street sizing ---
_VISIBLE_BB = 0.02
_B2P_EDGES = (0.33, 0.5, 0.66, 0.75, 1.0, 1.5)
_ROUND_FRACS = (0.5, 0.66, 0.75, 1.0)


def _pair_mean_std(values: Sequence[float]) -> tuple:
    arr = [float(v) for v in values]
    if not arr:
        return (0.0, 0.0)
    n = len(arr)
    mean = sum(arr) / n
    return (mean, math.sqrt(sum((x - mean) ** 2 for x in arr) / n))


def _bucket_entropy(values: Sequence[float], edges: Sequence[float]) -> float:
    if not values:
        return 0.0
    counts = [0] * (len(edges) + 1)
    for v in values:
        i = 0
        while i < len(edges) and v > edges[i]:
            i += 1
        counts[i] += 1
    total = sum(counts)
    probs = [c / total for c in counts if c > 0]
    if len(probs) <= 1:
        return 0.0
    return -sum(p * math.log(p) for p in probs) / math.log(len(counts))


def _extra_dict(group: List[dict]) -> Dict[str, float]:
    hands = [h for h in (group or []) if isinstance(h, dict)]
    bet2pot: List[float] = []
    sizes_bb: List[float] = []
    call2pot: List[float] = []
    raise2pot: List[float] = []
    pot_bb: List[float] = []
    acts: List[tuple] = []
    pre_calls = 0
    pre_raises = 0
    sb2p = {s: [] for s in STREETS}
    per_hand_aggr: List[int] = []
    multibarrel = 0

    for hand in hands:
        meta = hand.get("metadata") if isinstance(hand.get("metadata"), dict) else {}
        hero_seat = meta.get("hero_seat")
        aggr_in_hand = 0
        for a in (hand.get("actions") or []):
            if not isinstance(a, dict) or a.get("actor_seat") != hero_seat:
                continue
            at = str(a.get("action_type") or "").lower()
            pot = _f(a.get("pot_before"))
            pbb = pot / _VISIBLE_BB if pot > 0 else 0.0
            street = str(a.get("street") or "preflop").lower()
            if street not in sb2p:
                street = "preflop"
            amount = _f(a.get("amount"))
            size_bb = _f(a.get("normalized_amount_bb"))
            acts.append((at, pbb))
            pot_bb.append(pbb)
            if at in ("bet", "raise"):
                aggr_in_hand += 1
                if size_bb > 0:
                    sizes_bb.append(size_bb)
                if pot > 0 and amount > 0:
                    ratio = amount / pot
                    bet2pot.append(ratio)
                    sb2p[street].append(ratio)
                rt = a.get("raise_to")
                if pot > 0 and rt is not None and _f(rt) > 0:
                    raise2pot.append(_f(rt) / pot)
            if at == "call":
                ct = a.get("call_to")
                if pot > 0 and ct is not None and _f(ct) > 0:
                    call2pot.append(_f(ct) / pot)
            if street == "preflop" and at == "call":
                pre_calls += 1
            if street == "preflop" and at == "raise":
                pre_raises += 1
        per_hand_aggr.append(aggr_in_hand)
        if aggr_in_hand >= 2:
            multibarrel += 1

    nb = max(1, len(bet2pot))
    call_m, call_s = _pair_mean_std(call2pot)
    rr_m, rr_s = _pair_mean_std(raise2pot)
    fold_hi = fold_lo = aggr_hi = aggr_lo = nhi = nlo = 0
    if pot_bb:
        med = sorted(pot_bb)[len(pot_bb) // 2]
        for at, pb in acts:
            hi = pb >= med
            nhi += hi
            nlo += (not hi)
            if at == "fold":
                fold_hi += hi
                fold_lo += (not hi)
            if at in ("bet", "raise"):
                aggr_hi += hi
                aggr_lo += (not hi)

    d: Dict[str, float] = {}
    d["x_b2p_distinct_rate"] = _safe_div(len({round(v, 1) for v in bet2pot}), nb)
    d["x_sbb_distinct_rate"] = _safe_div(len({round(v, 0) for v in sizes_bb}), max(1, len(sizes_bb)))
    d["x_b2p_entropy"] = _bucket_entropy(bet2pot, _B2P_EDGES)
    d["x_b2p_round_frac"] = _safe_div(sum(1 for v in bet2pot if any(abs(v - r) <= 0.08 for r in _ROUND_FRACS)), nb)
    d["x_call2pot_mean"] = call_m
    d["x_call2pot_std"] = call_s
    d["x_raise2pot_mean2"] = rr_m
    d["x_raise2pot_std2"] = rr_s
    d["x_fold_hi"] = _safe_div(fold_hi, nhi)
    d["x_fold_lo"] = _safe_div(fold_lo, nlo)
    d["x_aggr_hi"] = _safe_div(aggr_hi, nhi)
    d["x_aggr_lo"] = _safe_div(aggr_lo, nlo)
    d["x_fold_pressure_delta"] = _safe_div(fold_hi, nhi) - _safe_div(fold_lo, nlo)
    d["x_aggr_pressure_delta"] = _safe_div(aggr_hi, nhi) - _safe_div(aggr_lo, nlo)
    d["x_pre_raise_mix"] = _safe_div(pre_raises, pre_calls + pre_raises)
    d["x_b2p_distinct_count"] = float(len({round(v, 1) for v in bet2pot}))
    for s in STREETS:
        m, sd = _pair_mean_std(sb2p[s])
        d[f"x_sb2p_{s}_mean"] = m
        d[f"x_sb2p_{s}_std"] = sd
    pm, psd = _pair_mean_std(per_hand_aggr)
    d["x_perhand_aggrcnt_mean"] = pm
    d["x_perhand_aggrcnt_std"] = psd
    d["x_perhand_aggrcnt_max"] = float(max(per_hand_aggr) if per_hand_aggr else 0.0)
    d["x_multibarrel_rate"] = _safe_div(multibarrel, len(hands))
    return d


# Stable, canonical feature order (empty group yields every key with 0.0).
_BASE_NAMES: List[str] = list(_feature_dict([]).keys())
_EXTRA_NAMES: List[str] = list(_extra_dict([]).keys())
FEATURE_NAMES: List[str] = _BASE_NAMES + _EXTRA_NAMES


def extract_features(group: List[dict]) -> List[float]:
    """Return the feature vector for one chunk group, ordered by FEATURE_NAMES."""
    base = _feature_dict(group)
    extra = _extra_dict(group)
    return [base[name] for name in _BASE_NAMES] + [extra[name] for name in _EXTRA_NAMES]


def extract_matrix(groups: List[List[dict]]) -> List[List[float]]:
    return [extract_features(g) for g in groups]


if __name__ == "__main__":
    print(f"{len(FEATURE_NAMES)} features:")
    for name in FEATURE_NAMES:
        print("  ", name)
