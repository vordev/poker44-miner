"""Asynchronous forward loop for the Poker44 validator."""
## poker44/validator/forward.py

from __future__ import annotations

import asyncio
import os
import traceback
from typing import Any, Dict, List, Sequence, Tuple

import bittensor as bt
import numpy as np

from poker44.score.scoring import reward
from poker44.utils.model_manifest import manifest_digest, normalize_model_manifest
from poker44.validator.integrity import (
    chunk_fingerprint,
    evaluate_manifest_compliance,
    evaluate_manifest_suspicion,
    persist_json_registry,
    record_served_chunks,
    update_compliance_registry,
    update_suspicion_registry,
)
from poker44.validator.synapse import DetectionSynapse
from poker44.validator.payload_view import prepare_hand_for_miner

from poker44.validator.constants import (
    BURN_EMISSIONS,
    BURN_FRACTION,
    KEEP_FRACTION,
    UID_ZERO,
)


async def forward(validator) -> None:
    """Entry point invoked by :class:`neurons.validator.Validator`."""
    try:
        await _run_forward_cycle(validator)
    except Exception:
        wandb_helper = getattr(validator, "wandb_helper", None)
        if wandb_helper is not None:
            wandb_helper.log_error(
                "forward_cycle_unexpected",
                traceback.format_exc(),
            )
        bt.logging.error(f"Unexpected error in forward cycle:\n{traceback.format_exc()}")


async def _run_forward_cycle(validator) -> None:
    validator.forward_count = getattr(validator, "forward_count", 0) + 1
    bt.logging.info(f"[Forward #{validator.forward_count}] start")
    wandb_helper = getattr(validator, "wandb_helper", None)

    if hasattr(validator.provider, "refresh_if_due"):
        validator.provider.refresh_if_due()

    # Fetch the full stable dataset snapshot selected by the backend.
    batches = validator.provider.fetch_hand_batch()
    if not batches:
        bt.logging.info("No hands fetched from dataset; sleeping.")
        if wandb_helper is not None:
            wandb_helper.log_forward_summary(
                forward_count=validator.forward_count,
                chunk_count=0,
                total_hands=0,
                miner_count=0,
                responded_count=0,
                successful_miners=0,
                dataset_hash=getattr(validator.provider, "dataset_hash", ""),
                dataset_stats=getattr(validator.provider, "stats", {}),
                extra={"forward/status": "no_batches"},
            )
        await asyncio.sleep(validator.poll_interval)
        return
    
    miner_uids, axons = _get_candidate_miners(validator)
    responses: Dict[int, List[float]] = {uid: [] for uid in miner_uids}

    if not miner_uids:
        bt.logging.info("No eligible miner UIDs available for this cycle.")
        if wandb_helper is not None:
            wandb_helper.log_forward_summary(
                forward_count=validator.forward_count,
                chunk_count=len(batches),
                total_hands=sum(len(batch.hands) for batch in batches),
                miner_count=0,
                responded_count=0,
                successful_miners=0,
                dataset_hash=getattr(validator.provider, "dataset_hash", ""),
                dataset_stats=getattr(validator.provider, "stats", {}),
                extra={"forward/status": "no_eligible_miners"},
            )
        _finalize_provider_cycle(validator, evaluation_completed=False)
        await asyncio.sleep(validator.poll_interval)
        return
    
    # Prepare chunks and labels
    chunks = []  # List of batches (each batch is a list of hand dicts)
    batch_labels = []  # One label per batch
    
    for batch in batches:
        # Convert HandHistory objects to dicts
        chunk_dicts = []
        for hand in batch.hands:
            hand_payload: Dict[str, Any]
            if isinstance(hand, dict):
                hand_payload = hand
            else:
                # Assume hand has a to_payload() or to_dict() method
                try:
                    hand_payload = hand.to_payload()
                except AttributeError:
                    # Fallback: convert dataclass to dict
                    import dataclasses
                    if dataclasses.is_dataclass(hand):
                        hand_payload = dataclasses.asdict(hand)
                    else:
                        hand_payload = hand.__dict__

            chunk_dicts.append(prepare_hand_for_miner(hand_payload))
        
        chunks.append(chunk_dicts)
        
        # batch.is_human is False for bots, True for humans
        # We need: 1=bot, 0=human
        batch_label = 0 if batch.is_human else 1
        batch_labels.append(batch_label)
    
    bt.logging.info(f"Processing {len(chunks)} chunks with labels: {batch_labels} (1=bot, 0=human)")
    bt.logging.info(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")
    validator.current_eval_sample_count = len(chunks)
    _record_served_chunk_fingerprints(
        validator,
        chunks=chunks,
        dataset_hash=getattr(validator.provider, "dataset_hash", ""),
    )
    if wandb_helper is not None:
        provider_stats = getattr(validator.provider, "stats", {})
        wandb_helper.log_dataset_state(
            dataset_hash=getattr(validator.provider, "dataset_hash", ""),
            stats=provider_stats,
        )
    
    # Create synapse with all chunks (now as list of dicts)
    synapse = DetectionSynapse(chunks=chunks)
    
    # Larger canonical snapshots require more miner-side processing time.
    timeout = 180.0
    if hasattr(validator.config, "neuron") and hasattr(validator.config.neuron, "timeout"):
        try:
            timeout = float(validator.config.neuron.timeout)
        except (ValueError, TypeError):
            timeout = 180.0
    try:
        timeout = float(os.getenv("POKER44_MINER_QUERY_TIMEOUT_SECONDS", str(timeout)))
    except (ValueError, TypeError):
        timeout = 180.0
    timeout = max(30.0, timeout)
    
    total_hands = sum(len(chunk) for chunk in chunks)
    bt.logging.info(f"Querying {len(axons)} miners with {len(chunks)} chunks ({total_hands} total hands)...")
    
    synapse_responses = await _dendrite_with_retries(
        validator.dendrite,
        axons=axons,
        synapse=synapse,
        timeout=timeout,
        attempts=3,
    )
    bt.logging.info(f"Received {len(synapse_responses)} responses from miners")

    expected_chunk_count = len(chunks)
    response_metadata: Dict[int, Dict[str, Any]] = {}
    
    for uid, resp in zip(miner_uids, synapse_responses):
        if resp is None:
            bt.logging.debug(f"Miner {uid} returned None response")
            response_metadata[uid] = {
                "coverage_rate": 0.0,
                "latency_seconds": None,
            }
            continue

        _record_model_manifest(
            validator,
            uid,
            getattr(resp, "model_manifest", None),
            dataset_hash=getattr(validator.provider, "dataset_hash", ""),
        )
            
        scores = getattr(resp, "risk_scores", None)
        if scores is None:
            bt.logging.debug(f"Miner {uid} returned no risk_scores")
            response_metadata[uid] = {
                "coverage_rate": 0.0,
                "latency_seconds": _extract_latency_seconds(resp),
            }
            continue
            
        try:
            scores_f = [float(s) for s in scores]
            
            if len(scores_f) != len(chunks):
                bt.logging.warning(
                    f"Miner {uid} returned {len(scores_f)} scores but expected {len(chunks)} "
                    "(one per chunk); discarding incomplete response."
                )
                response_metadata[uid] = {
                    "coverage_rate": 0.0,
                    "latency_seconds": _extract_latency_seconds(resp),
                }
                validator.coverage_buffer.setdefault(uid, []).append(0.0)
                continue
            effective_labels = batch_labels

            coverage_rate = (
                float(len(scores_f)) / float(expected_chunk_count)
                if expected_chunk_count > 0
                else 0.0
            )
            latency_seconds = _extract_latency_seconds(resp)
            response_metadata[uid] = {
                "coverage_rate": coverage_rate,
                "latency_seconds": latency_seconds,
            }
            validator.coverage_buffer.setdefault(uid, []).append(coverage_rate)
            if latency_seconds is not None:
                validator.latency_buffer.setdefault(uid, []).append(latency_seconds)
            
            responses[uid].extend(scores_f)
            
            # Store predictions and labels (one per chunk)
            if not hasattr(validator, "prediction_buffer"):
                validator.prediction_buffer = {}
            if not hasattr(validator, "label_buffer"):
                validator.label_buffer = {}
            
            validator.prediction_buffer.setdefault(uid, []).extend(scores_f)
            validator.label_buffer.setdefault(uid, []).extend(effective_labels)
            
            bt.logging.info(f"Miner {uid} scored {len(scores_f)} chunks successfully")
        except Exception as e:
            bt.logging.warning(f"Error processing response from miner {uid}: {e}")
            import traceback
            bt.logging.debug(traceback.format_exc())
            response_metadata[uid] = {
                "coverage_rate": 0.0,
                "latency_seconds": _extract_latency_seconds(resp),
            }
            continue

    for uid in miner_uids:
        if uid in response_metadata:
            continue
        response_metadata[uid] = {
            "coverage_rate": 0.0,
            "latency_seconds": None,
        }
        validator.coverage_buffer.setdefault(uid, []).append(0.0)
    
    if not any(responses.values()):
        bt.logging.info("No miner responses this cycle.")
        if wandb_helper is not None:
            wandb_helper.log_forward_summary(
                forward_count=validator.forward_count,
                chunk_count=len(chunks),
                total_hands=total_hands,
                miner_count=len(axons),
                responded_count=len(synapse_responses),
                successful_miners=0,
                dataset_hash=getattr(validator.provider, "dataset_hash", ""),
                dataset_stats=getattr(validator.provider, "stats", {}),
                extra={
                    "forward/status": "no_valid_scores",
                    "forward/human_chunk_count": sum(1 for label in batch_labels if label == 0),
                    "forward/bot_chunk_count": sum(1 for label in batch_labels if label == 1),
                },
            )
        _finalize_provider_cycle(validator, evaluation_completed=False)
        await asyncio.sleep(validator.poll_interval)
        return
    
    rewards_array, metrics = _compute_windowed_rewards(validator, miner_uids)
    reward_map = dict(zip(miner_uids, rewards_array.tolist()))
    metrics_map = {uid: metric for uid, metric in zip(miner_uids, metrics)}
    validator.competition_scores_payload = _build_competition_scores_payload(
        validator,
        miner_uids=miner_uids,
        metrics_map=metrics_map,
        response_metadata=response_metadata,
    )
    record_audit_report = getattr(validator, "_record_audit_report", None)
    if callable(record_audit_report):
        try:
            record_audit_report(
                total_hands=total_hands,
                chunk_count=len(chunks),
                human_chunk_count=sum(1 for label in batch_labels if label == 0),
                bot_chunk_count=sum(1 for label in batch_labels if label == 1),
            )
        except Exception as exc:
            bt.logging.warning(f"Audit report recording failed: {exc}")
    report_competition_scores = getattr(validator, "_report_competition_scores", None)
    if callable(report_competition_scores):
        try:
            report_competition_scores()
        except Exception as exc:
            bt.logging.warning(f"Competition score reporting failed: {exc}")
    bt.logging.info(f"Reward map by UID: {reward_map}")
    bt.logging.info(f"Reward metrics by UID: {metrics_map}")
    winner_uids, winner_rewards = _select_weight_targets(reward_map)

    validator.update_scores(winner_rewards, winner_uids)
    if wandb_helper is not None:
        successful_miners = sum(1 for scores in responses.values() if scores)
        wandb_helper.log_forward_summary(
            forward_count=validator.forward_count,
            chunk_count=len(chunks),
            total_hands=total_hands,
            miner_count=len(axons),
            responded_count=len(synapse_responses),
            successful_miners=successful_miners,
            dataset_hash=getattr(validator.provider, "dataset_hash", ""),
            dataset_stats=getattr(validator.provider, "stats", {}),
            extra={
                "forward/status": "ok",
                "forward/human_chunk_count": sum(1 for label in batch_labels if label == 0),
                "forward/bot_chunk_count": sum(1 for label in batch_labels if label == 1),
            },
        )
        wandb_helper.log_reward_summary(
            reward_map=reward_map,
            metrics_map=metrics_map,
            winner_uids=[int(uid) for uid in winner_uids],
            winner_rewards=[float(weight) for weight in winner_rewards],
        )
    bt.logging.info(f"Rewards issued for {len(winner_rewards)} UID(s).")
    _finalize_provider_cycle(validator, evaluation_completed=True)
    bt.logging.info(
        f"[Forward #{validator.forward_count}] complete. Sleeping {validator.poll_interval}s before next tick.",
    )
    await asyncio.sleep(validator.poll_interval)


def _finalize_provider_cycle(validator, *, evaluation_completed: bool) -> None:
    if not evaluation_completed:
        bt.logging.info(
            "Skipping provider-runtime finalization because the cycle did not produce usable evaluation results."
        )
        return
    mark_evaluated = getattr(validator.provider, "mark_last_batch_evaluated", None)
    if not callable(mark_evaluated):
        return
    try:
        mark_evaluated()
    except Exception as exc:
        bt.logging.warning(f"Provider runtime finalization failed: {exc}")


def _record_model_manifest(
    validator,
    uid: int,
    manifest: Dict[str, Any] | None,
    *,
    dataset_hash: str,
) -> None:
    normalized = normalize_model_manifest(manifest)
    suspicion_reasons = evaluate_manifest_suspicion(normalized if normalized else None)
    _record_suspicion(
        validator,
        uid,
        reasons=suspicion_reasons,
        dataset_hash=dataset_hash,
    )
    _record_compliance(
        validator,
        uid,
        manifest=normalized if normalized else None,
        dataset_hash=dataset_hash,
    )

    if not normalized:
        bt.logging.debug(f"Miner {uid} did not provide a model manifest.")
        return

    digest = manifest_digest(normalized)
    registry = getattr(validator, "model_manifest_registry", None)
    if registry is None:
        registry = {}
        validator.model_manifest_registry = registry

    previous = registry.get(uid)
    previous_digest = previous.get("manifest_digest") if previous else None
    if previous_digest == digest:
        return

    entry = {
        "uid": int(uid),
        "manifest_digest": digest,
        "model_manifest": normalized,
    }
    registry[uid] = entry

    bt.logging.info(
        f"Miner {uid} manifest updated | "
        f"open_source={normalized.get('open_source')} "
        f"model={normalized.get('model_name', '')} "
        f"version={normalized.get('model_version', '')} "
        f"repo={normalized.get('repo_url', '')} "
        f"commit={normalized.get('repo_commit', '')}"
    )
    _persist_model_manifest_registry(getattr(validator, "model_manifest_path", None), registry)


def _persist_model_manifest_registry(path: str | Path | None, registry: Dict[int, Dict[str, Any]]) -> None:
    def _uid_sort_key(uid: Any) -> tuple[int, str]:
        try:
            return (0, f"{int(uid):010d}")
        except (TypeError, ValueError):
            return (1, str(uid))

    payload = {
        str(uid): registry[uid]
        for uid in sorted(registry, key=_uid_sort_key)
    }
    persist_json_registry(path, payload)


def _record_served_chunk_fingerprints(validator, *, chunks: List[List[dict]], dataset_hash: str) -> None:
    registry = getattr(validator, "served_chunk_registry", None)
    if registry is None:
        registry = {"chunk_index": {}, "recent_cycles": [], "summary": {}}
        validator.served_chunk_registry = registry

    chunk_hashes = [chunk_fingerprint(chunk) for chunk in chunks]
    summary = record_served_chunks(
        registry,
        chunk_hashes=chunk_hashes,
        forward_count=int(getattr(validator, "forward_count", 0)),
        dataset_hash=dataset_hash,
    )
    persist_json_registry(getattr(validator, "served_chunk_registry_path", None), registry)

    if summary["repeated_count"] > 0:
        bt.logging.warning(
            f"Forward #{getattr(validator, 'forward_count', 0)} reused "
            f"{summary['repeated_count']} chunk fingerprints; "
            f"{summary['unique_count']} unique chunk fingerprints tracked so far."
        )


def _record_suspicion(
    validator,
    uid: int,
    *,
    reasons: List[str],
    dataset_hash: str,
) -> None:
    registry = getattr(validator, "suspicion_registry", None)
    if registry is None:
        registry = {"miners": {}, "summary": {}}
        validator.suspicion_registry = registry

    event = update_suspicion_registry(
        registry,
        uid=int(uid),
        reasons=reasons,
        forward_count=int(getattr(validator, "forward_count", 0)),
        dataset_hash=dataset_hash,
    )
    if event is None:
        return

    bt.logging.warning(f"Miner {uid} integrity suspicion flags: {', '.join(reasons)}")
    persist_json_registry(getattr(validator, "suspicion_registry_path", None), registry)


def _record_compliance(
    validator,
    uid: int,
    *,
    manifest: Dict[str, Any] | None,
    dataset_hash: str,
) -> None:
    registry = getattr(validator, "compliance_registry", None)
    if registry is None:
        registry = {"miners": {}, "summary": {}}
        validator.compliance_registry = registry

    compliance = evaluate_manifest_compliance(manifest)
    digest = manifest_digest(manifest or {})
    entry = update_compliance_registry(
        registry,
        uid=int(uid),
        compliance=compliance,
        manifest_digest=digest,
        forward_count=int(getattr(validator, "forward_count", 0)),
        dataset_hash=dataset_hash,
    )
    persist_json_registry(getattr(validator, "compliance_registry_path", None), registry)

    if entry.get("status_changed"):
        bt.logging.info(
            f"Miner {uid} compliance status changed to {entry['status']} "
            f"(missing_fields={entry['missing_fields']})"
        )


def _get_candidate_miners(validator) -> Tuple[List[int], List]:
    miner_uids: List[int] = []
    axons: List = []
    target_uids_env = os.getenv("POKER44_TARGET_MINER_UIDS", "").strip()
    miners_per_cycle_env = os.getenv("POKER44_MINERS_PER_CYCLE", "16").strip()
    min_validator_stake_env = os.getenv("POKER44_MIN_VALIDATOR_STAKE", "17000").strip()
    miners_per_cycle = 16
    min_validator_stake = 17000.0
    target_uids = None
    if target_uids_env:
        try:
            target_uids = {
                int(uid.strip())
                for uid in target_uids_env.split(",")
                if uid.strip() != ""
            }
            bt.logging.info(f"Restricting miner queries to target UIDs: {sorted(target_uids)}")
        except ValueError:
            bt.logging.warning(
                f"Invalid POKER44_TARGET_MINER_UIDS={target_uids_env!r}; ignoring filter."
            )
            target_uids = None
    try:
        miners_per_cycle = int(miners_per_cycle_env)
    except ValueError:
        bt.logging.warning(
            f"Invalid POKER44_MINERS_PER_CYCLE={miners_per_cycle_env!r}; defaulting to 16."
        )
        miners_per_cycle = 16
    try:
        min_validator_stake = float(min_validator_stake_env)
    except ValueError:
        bt.logging.warning(
            f"Invalid POKER44_MIN_VALIDATOR_STAKE={min_validator_stake_env!r}; defaulting to 17000."
        )
        min_validator_stake = 17000.0

    for uid, axon in enumerate(validator.metagraph.axons):
        if uid == UID_ZERO:
            continue
        if target_uids is not None and uid not in target_uids:
            continue
        stake = 0.0
        try:
            stake = float(validator.metagraph.S[uid])
        except Exception:
            stake = 0.0
        if bool(validator.metagraph.validator_permit[uid]) and stake >= min_validator_stake:
            continue
        ip = str(getattr(axon, "ip", "") or "")
        port = int(getattr(axon, "port", 0) or 0)
        if ip in {"", "0.0.0.0", "::", "[::]"} or port <= 0:
            continue
        miner_uids.append(uid)
        axons.append(axon)

    if target_uids is None and miners_per_cycle > 0 and len(miner_uids) > miners_per_cycle:
        # Rotate deterministically through the eligible set so coverage expands over time
        # without blasting every miner on each cycle.
        offset = ((getattr(validator, "forward_count", 1) - 1) * miners_per_cycle) % len(miner_uids)
        rotated = list(zip(miner_uids, axons))
        rotated = rotated[offset:] + rotated[:offset]
        selected = rotated[:miners_per_cycle]
        miner_uids = [uid for uid, _ in selected]
        axons = [axon for _, axon in selected]
        bt.logging.info(
            f"Sampling {miners_per_cycle} miners this cycle from {len(rotated)} eligible miners "
            f"(rotation offset={offset})."
        )

    bt.logging.info(f"Eligible miners this cycle: {miner_uids}")
    return miner_uids, axons


def _compute_windowed_rewards(validator, miner_uids: List[int]) -> tuple[np.ndarray, list]:
    current_sample_count = int(getattr(validator, "current_eval_sample_count", 0) or 0)
    window = current_sample_count
    if window <= 0:
        window = 1
    rewards: List[float] = []
    metrics: List[dict] = []

    for uid in miner_uids:
        pred_buf = validator.prediction_buffer.get(uid, [])
        label_buf = validator.label_buffer.get(uid, [])
        coverage_buf = validator.coverage_buffer.get(uid, [])
        latency_buf = validator.latency_buffer.get(uid, [])
        coverage_rate = float(np.mean(coverage_buf[-window:])) if coverage_buf else 0.0
        latency_mean_seconds = (
            float(np.mean(latency_buf[-window:])) if latency_buf else None
        )

        if len(pred_buf) < window or len(label_buf) < window:
            current_sample_count = int(
                min(len(pred_buf), len(label_buf), max(0, window))
            )
            rewards.append(0.0)
            metrics.append(
                {
                    "fpr": 1.0,
                    "bot_recall": 0.0,
                    "ap_score": 0.0,
                    "human_safety_penalty": 0.0,
                    "base_score": 0.0,
                    "reward": 0.0,
                    "coverage_rate": coverage_rate,
                    "latency_mean_seconds": latency_mean_seconds,
                    "sample_count": current_sample_count,
                }
            )
            continue

        preds_window = np.asarray(pred_buf[-window:], dtype=float)
        labels_window = np.asarray(label_buf[-window:], dtype=bool)
        rew, metric = reward(preds_window, labels_window)
        metric["coverage_rate"] = coverage_rate
        metric["latency_mean_seconds"] = latency_mean_seconds
        metric["sample_count"] = int(len(labels_window))
        rewards.append(rew)
        metrics.append(metric)

    rewards_array = np.asarray(rewards, dtype=np.float32)
    
    return rewards_array, metrics


def _extract_latency_seconds(resp: Any) -> float | None:
    dendrite_info = getattr(resp, "dendrite", None)
    process_time = getattr(dendrite_info, "process_time", None)
    if process_time is None:
        return None
    try:
        value = float(process_time)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _build_competition_scores_payload(
    validator,
    *,
    miner_uids: List[int],
    metrics_map: Dict[int, Dict[str, Any]],
    response_metadata: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    manifests = getattr(validator, "model_manifest_registry", {}) or {}
    hotkeys = list(getattr(validator.metagraph, "hotkeys", []) or [])

    for uid in miner_uids:
        metric = metrics_map.get(uid, {})
        response_meta = response_metadata.get(uid, {})
        manifest_entry = manifests.get(uid) or manifests.get(str(uid)) or {}
        manifest = manifest_entry.get("model_manifest") or {}

        rows.append(
            {
                "uid": int(uid),
                "hotkey": hotkeys[uid] if uid < len(hotkeys) else "",
                "manifest_digest": manifest_entry.get("manifest_digest"),
                "implementation_sha256": manifest.get("implementation_sha256"),
                "model_name": manifest.get("model_name"),
                "model_version": manifest.get("model_version"),
                "repo_url": manifest.get("repo_url"),
                "repo_commit": manifest.get("repo_commit"),
                "open_source": manifest.get("open_source"),
                "reward": float(metric.get("reward", 0.0) or 0.0),
                "ap_score": float(metric.get("ap_score", 0.0) or 0.0),
                "bot_recall": float(metric.get("bot_recall", 0.0) or 0.0),
                "human_safety_penalty": float(
                    metric.get("human_safety_penalty", 0.0) or 0.0
                ),
                "coverage_rate": float(
                    metric.get(
                        "coverage_rate",
                        response_meta.get("coverage_rate", 0.0),
                    )
                    or 0.0
                ),
                "latency_mean_seconds": metric.get(
                    "latency_mean_seconds",
                    response_meta.get("latency_seconds"),
                ),
                "sample_count": int(metric.get("sample_count", 0) or 0),
            }
        )

    return rows


def _select_weight_targets(reward_map: Dict[int, float]) -> tuple[List[int], np.ndarray]:
    if not reward_map:
        bt.logging.info("No eligible rewards computed; assigning 100%% to UID 0.")
        return [UID_ZERO], np.asarray([1.0], dtype=np.float32)

    sorted_rewards = sorted(reward_map.items(), key=lambda item: (-item[1], item[0]))
    winner_uid, winner_reward = sorted_rewards[0]

    if winner_reward <= 0.0:
        bt.logging.info("No miner achieved positive reward; assigning 100%% to UID 0.")
        return [UID_ZERO], np.asarray([1.0], dtype=np.float32)

    if BURN_EMISSIONS:
        bt.logging.info(
            f"Winner-take-all burn enabled: UID 0 gets {BURN_FRACTION * 100:.2f}%, "
            f"winner UID {winner_uid} gets {KEEP_FRACTION * 100:.2f}%."
        )
        return [UID_ZERO, winner_uid], np.asarray(
            [BURN_FRACTION, KEEP_FRACTION], dtype=np.float32
        )

    bt.logging.info(f"Winner-take-all enabled: winner UID {winner_uid} gets 100%.")
    return [winner_uid], np.asarray([1.0], dtype=np.float32)

async def _dendrite_with_retries(
    dendrite: bt.dendrite,
    *,
    axons: Sequence,
    synapse: DetectionSynapse,
    timeout: float,
    attempts: int = 3,
):
    """
    Simple retry loop around dendrite calls to avoid transient failures.
    """
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return await dendrite(
                axons=axons,
                synapse=synapse,
                timeout=timeout,
            )
        except Exception as exc:
            last_exc = exc
            bt.logging.warning(f"dendrite attempt {attempt}/{attempts} failed: {exc}")
            await asyncio.sleep(0.5)
    bt.logging.error(f"dendrite retries exhausted: {last_exc}")
    return [None] * len(axons)
