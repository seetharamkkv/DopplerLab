"""Leakage-free train / test / val split builders for pass-by audio.

Protocols
---------
scene
    Hold out *scenes* ``vehicle|integer_speed_kmh``. Same car may appear in
    train and test at different speeds. Synth twins of a held-out scene never
    enter training. This matches the SE-ResNet downstream protocol.

lovo
    Leave-one-vehicle-out folds. Strongest unknown-car claim.

speed_stratified
    Clip-level 70/30 (etc.) stratified by speed bins. Cars may appear in both
    sets — easier mixed-fleet claim; do **not** call this LOVO or scene-holdout.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .catalog import (
    condition_key,
    discover_uids,
    normalize_uid,
    uid_to_speed,
    uid_to_vehicle,
)

ProtocolName = Literal["scene", "lovo", "speed_stratified"]


@dataclass
class SplitBundle:
    """Serializable split artifact."""

    protocol: ProtocolName
    meta: dict[str, Any]
    train_uids: list[str]
    test_uids: list[str]
    # Optional synth partitions (scene protocol)
    synth_train_uids: list[str] = field(default_factory=list)
    synth_test_uids: list[str] = field(default_factory=list)
    synth_test_forced_uids: list[str] = field(default_factory=list)
    synth_test_extra_uids: list[str] = field(default_factory=list)
    # Inner early-stopping (subset of train scenes / uids)
    fit_uids: list[str] = field(default_factory=list)
    val_uids: list[str] = field(default_factory=list)
    train_scene_keys: list[str] = field(default_factory=list)
    test_scene_keys: list[str] = field(default_factory=list)
    fit_scene_keys: list[str] = field(default_factory=list)
    val_scene_keys: list[str] = field(default_factory=list)
    fold_id: int | None = None
    held_out_vehicle: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Aliases consumed by the Prith SE-ResNet scripts
        d["real_train_uids"] = list(self.train_uids)
        d["real_test_uids"] = list(self.test_uids)
        d["blocked_condition_keys"] = list(self.test_scene_keys)
        d["paired_condition_keys"] = sorted(
            {condition_key(u) for u in self.synth_test_forced_uids}
        )
        d["s_train_uids"] = list(self.synth_train_uids)
        d["s_blocked_uids"] = list(self.synth_test_forced_uids)
        d["counts"] = {
            "n_real_train": len(self.train_uids),
            "n_real_test": len(self.test_uids),
            "n_synth_train": len(self.synth_train_uids),
            "n_synth_test": len(self.synth_test_uids),
            "n_fit": len(self.fit_uids),
            "n_val": len(self.val_uids),
            "n_test_scenes": len(self.test_scene_keys),
        }
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SplitBundle":
        train = data.get("train_uids") or data.get("real_train_uids") or []
        test = data.get("test_uids") or data.get("real_test_uids") or []
        protocol = data.get("protocol") or (data.get("meta") or {}).get(
            "split_scheme", "scene"
        )
        if protocol in (
            "scene_level_paired_holdout",
            "scene_level_paired_holdout_v2",
            "scene_level_paired_holdout_v4",
        ):
            protocol = "scene"
        return cls(
            protocol=protocol,  # type: ignore[arg-type]
            meta=dict(data.get("meta") or {}),
            train_uids=[normalize_uid(u) for u in train],
            test_uids=[normalize_uid(u) for u in test],
            synth_train_uids=[
                normalize_uid(u)
                for u in (data.get("synth_train_uids") or data.get("s_train_uids") or [])
            ],
            synth_test_uids=[normalize_uid(u) for u in (data.get("synth_test_uids") or [])],
            synth_test_forced_uids=[
                normalize_uid(u)
                for u in (
                    data.get("synth_test_forced_uids") or data.get("s_blocked_uids") or []
                )
            ],
            synth_test_extra_uids=[
                normalize_uid(u) for u in (data.get("synth_test_extra_uids") or [])
            ],
            fit_uids=[normalize_uid(u) for u in (data.get("fit_uids") or [])],
            val_uids=[normalize_uid(u) for u in (data.get("val_uids") or [])],
            train_scene_keys=list(
                data.get("train_scene_keys")
                or sorted({condition_key(u) for u in train})
            ),
            test_scene_keys=list(
                data.get("test_scene_keys")
                or data.get("blocked_condition_keys")
                or sorted({condition_key(u) for u in test})
            ),
            fit_scene_keys=list(data.get("fit_scene_keys") or []),
            val_scene_keys=list(data.get("val_scene_keys") or []),
            fold_id=data.get("fold_id"),
            held_out_vehicle=data.get("held_out_vehicle"),
        )


def save_split(path: str | Path, split: SplitBundle | dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = split.to_dict() if isinstance(split, SplitBundle) else split
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


def load_split(path: str | Path) -> SplitBundle:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SplitBundle.from_dict(data)


def _stratified_by_vehicle(
    uids: list[str], train_frac: float, seed: int
) -> tuple[list[str], list[str]]:
    from sklearn.model_selection import StratifiedShuffleSplit

    by_vehicle = Counter(uid_to_vehicle(u) for u in uids)
    rare = {v for v, n in by_vehicle.items() if n < 2}
    main = [u for u in uids if uid_to_vehicle(u) not in rare]
    rare_uids = [u for u in uids if uid_to_vehicle(u) in rare]

    if not main:
        rng = np.random.RandomState(seed)
        order = list(uids)
        rng.shuffle(order)
        n_train = max(1, int(round(train_frac * len(order))))
        return sorted(order[:n_train]), sorted(order[n_train:])

    y = np.array([uid_to_vehicle(u) for u in main])
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    tr_idx, te_idx = next(sss.split(main, y))
    train = [main[i] for i in tr_idx] + rare_uids
    test = [main[i] for i in te_idx]
    return sorted(train), sorted(test)


def adjacent_twin_uids(
    train_uids: list[str],
    test_uids: list[str],
    *,
    min_speed_gap: int = 2,
) -> list[str]:
    """Test UIDs that have a same-vehicle train speed within ``min_speed_gap``.

    ``min_speed_gap=2`` flags ±1 km/h twins (the usual near-duplicate case).
    """
    if min_speed_gap <= 1:
        return []
    from collections import defaultdict

    train_speeds: dict[str, set[int]] = defaultdict(set)
    for u in train_uids:
        train_speeds[uid_to_vehicle(u)].add(uid_to_speed(u))

    bad: list[str] = []
    for u in test_uids:
        v, s = uid_to_vehicle(u), uid_to_speed(u)
        for ts in train_speeds.get(v, ()):
            if abs(ts - s) < min_speed_gap:
                bad.append(u)
                break
    return sorted(bad)


def enforce_min_speed_gap(
    train_uids: list[str],
    test_uids: list[str],
    *,
    min_speed_gap: int = 2,
) -> tuple[list[str], list[str], list[str]]:
    """Demote conflicting test clips into train until the gap holds.

    Returns ``(train, test, demoted_test_uids)``.
    With ``min_speed_gap <= 1`` this is a no-op (exact scene keys already unique).
    """
    if min_speed_gap <= 1:
        return sorted(train_uids), sorted(test_uids), []

    train = set(train_uids)
    test = set(test_uids)
    demoted: list[str] = []

    # Iterate until stable — demoting one clip can clear several conflicts.
    while True:
        bad = adjacent_twin_uids(sorted(train), sorted(test), min_speed_gap=min_speed_gap)
        if not bad:
            break
        # Demote one conflict per round (deterministic order from adjacent_twin_uids)
        u = bad[0]
        test.remove(u)
        train.add(u)
        demoted.append(u)

    if not test:
        raise ValueError(
            f"min_speed_gap={min_speed_gap} removed every test clip; "
            "lower the gap or use a different seed/train_frac"
        )
    return sorted(train), sorted(test), demoted


def build_scene_split(
    real_root: str | Path,
    synth_root: str | Path | None = None,
    *,
    train_frac: float = 0.85,
    seed: int = 42,
    val_frac: float = 0.15,
    min_speed_gap: int = 1,
) -> SplitBundle:
    """Strict scene-level paired holdout (real ± optional synth).

    Parameters
    ----------
    min_speed_gap
        Minimum ``|speed_train - speed_test|`` allowed for the *same* vehicle.
        Default ``1`` = only exact ``vehicle|speed`` blocking (SE-ResNet default).
        Pass ``2`` to also ban ±1 km/h near-twins (opt-in “spread” mode).
    """
    if min_speed_gap < 1:
        raise ValueError("min_speed_gap must be >= 1")

    real_uids = discover_uids(real_root)
    real_train, real_test = _stratified_by_vehicle(real_uids, train_frac, seed)
    demoted: list[str] = []
    if min_speed_gap > 1:
        real_train, real_test, demoted = enforce_min_speed_gap(
            real_train, real_test, min_speed_gap=min_speed_gap
        )

    real_train_keys = {condition_key(u) for u in real_train}
    real_test_keys = {condition_key(u) for u in real_test}
    if real_train_keys & real_test_keys:
        raise ValueError("real train/test scene-key overlap")

    synth_forced: list[str] = []
    synth_for_train: list[str] = []
    orphans: list[str] = []
    if synth_root is not None:
        for u in discover_uids(synth_root):
            key = condition_key(u)
            if key in real_test_keys:
                synth_forced.append(u)
            elif key in real_train_keys:
                synth_for_train.append(u)
            else:
                orphans.append(u)

    orphan_train, orphan_test = (
        _stratified_by_vehicle(orphans, train_frac, seed + 1) if orphans else ([], [])
    )
    if min_speed_gap > 1 and orphan_test:
        # Spread orphan synth test the same way (against synth train pool)
        orphan_train, orphan_test, _ = enforce_min_speed_gap(
            orphan_train, orphan_test, min_speed_gap=min_speed_gap
        )

    synth_train = sorted(set(synth_for_train) | set(orphan_train))
    synth_extra = sorted(orphan_test)
    synth_test = sorted(set(synth_forced) | set(synth_extra))
    test_keys = sorted(real_test_keys | {condition_key(u) for u in synth_extra})
    train_keys = sorted(real_train_keys | {condition_key(u) for u in synth_train})
    if set(train_keys) & set(test_keys):
        raise ValueError("train/test scene-key overlap after synth assign")

    # Remaining adjacent conflicts must be zero when gap > 1
    still_bad = adjacent_twin_uids(real_train, real_test, min_speed_gap=min_speed_gap)
    if still_bad:
        raise RuntimeError(f"min_speed_gap enforcement failed, e.g. {still_bad[0]}")

    fit_keys, val_keys, fit_uids, val_uids = build_inner_scene_val(
        real_train, val_frac=val_frac, seed=seed
    )

    claim = (
        "Held-out scenes (vehicle|speed). Same vehicle may appear in "
        "train and test at different speeds — not LOVO."
    )
    if min_speed_gap > 1:
        claim += (
            f" Additionally, same-vehicle train/test speeds differ by "
            f">= {min_speed_gap} km/h (no +/-{min_speed_gap - 1} near-twins)."
        )

    return SplitBundle(
        protocol="scene",
        meta={
            "split_scheme": "scene_level_paired_holdout",
            "scene_key": "condition_key = vehicle|integer_speed_kmh",
            "real_root": str(Path(real_root).resolve()),
            "synth_root": str(Path(synth_root).resolve()) if synth_root else None,
            "seed": seed,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "min_speed_gap": min_speed_gap,
            "n_demoted_for_speed_gap": len(demoted),
            "demoted_uids_for_speed_gap": demoted,
            "claim": claim,
            "notes": (
                "Synth twins of real_test always go to synth_test; synth twins of "
                "real_train never enter synth_test. Mel stats must be fit on "
                "fit_uids only (never val/test). "
                "min_speed_gap=1 is the default SE-ResNet-style split; "
                "set min_speed_gap>=2 for opt-in near-twin spreading."
            ),
        },
        train_uids=real_train,
        test_uids=real_test,
        synth_train_uids=synth_train,
        synth_test_uids=synth_test,
        synth_test_forced_uids=sorted(synth_forced),
        synth_test_extra_uids=synth_extra,
        fit_uids=fit_uids,
        val_uids=val_uids,
        train_scene_keys=train_keys,
        test_scene_keys=test_keys,
        fit_scene_keys=sorted(fit_keys),
        val_scene_keys=sorted(val_keys),
    )


def build_inner_scene_val(
    train_uids: list[str],
    *,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[set[str], set[str], list[str], list[str]]:
    """Carve early-stopping val from *train* scenes only."""
    from sklearn.model_selection import ShuffleSplit

    by_scene: dict[str, str] = {}
    for u in train_uids:
        by_scene.setdefault(condition_key(u), u)
    scenes = sorted(by_scene)
    if len(scenes) < 2:
        keys = set(scenes)
        return keys, set(), list(train_uids), []

    ss = ShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    fit_i, val_i = next(ss.split(scenes))
    fit_keys = {scenes[i] for i in fit_i}
    val_keys = {scenes[i] for i in val_i}
    fit_uids = [u for u in train_uids if condition_key(u) in fit_keys]
    val_uids = [u for u in train_uids if condition_key(u) in val_keys]
    return fit_keys, val_keys, sorted(fit_uids), sorted(val_uids)


def build_lovo_folds(real_root: str | Path) -> list[SplitBundle]:
    """One fold per vehicle: train = all other cars, test = held-out car."""
    uids = discover_uids(real_root)
    vehicles = sorted({uid_to_vehicle(u) for u in uids})
    folds: list[SplitBundle] = []
    for i, held in enumerate(vehicles):
        train = sorted(u for u in uids if uid_to_vehicle(u) != held)
        test = sorted(u for u in uids if uid_to_vehicle(u) == held)
        fit_keys, val_keys, fit_uids, val_uids = build_inner_scene_val(train)
        folds.append(
            SplitBundle(
                protocol="lovo",
                meta={
                    "split_scheme": "leave_one_vehicle_out",
                    "real_root": str(Path(real_root).resolve()),
                    "claim": "Unknown vehicle — no train clip shares the test car.",
                },
                train_uids=train,
                test_uids=test,
                fit_uids=fit_uids,
                val_uids=val_uids,
                train_scene_keys=sorted({condition_key(u) for u in train}),
                test_scene_keys=sorted({condition_key(u) for u in test}),
                fit_scene_keys=sorted(fit_keys),
                val_scene_keys=sorted(val_keys),
                fold_id=i,
                held_out_vehicle=held,
            )
        )
    return folds


def build_speed_stratified_split(
    real_root: str | Path,
    *,
    train_frac: float = 0.7,
    seed: int = 42,
    n_bins: int = 5,
) -> SplitBundle:
    """Clip-level speed-bin stratified split (vehicles may overlap)."""
    from sklearn.model_selection import StratifiedShuffleSplit

    uids = discover_uids(real_root)
    speeds = np.array([uid_to_speed(u) for u in uids], dtype=np.float64)
    # Equal-frequency bins; rare bins collapsed by StratifiedShuffleSplit needs
    try:
        bins = np.quantile(speeds, np.linspace(0, 1, n_bins + 1))
        bins = np.unique(bins)
        y = np.digitize(speeds, bins[1:-1], right=True)
    except Exception:
        y = np.zeros(len(uids), dtype=int)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    tr_idx, te_idx = next(sss.split(uids, y))
    train = sorted(uids[i] for i in tr_idx)
    test = sorted(uids[i] for i in te_idx)
    fit_keys, val_keys, fit_uids, val_uids = build_inner_scene_val(train, seed=seed)
    return SplitBundle(
        protocol="speed_stratified",
        meta={
            "split_scheme": "speed_stratified_clip",
            "real_root": str(Path(real_root).resolve()),
            "seed": seed,
            "train_frac": train_frac,
            "n_bins": n_bins,
            "claim": (
                "Mixed fleet with held-out clips; same vehicle may appear in "
                "train and test. Not scene-holdout, not LOVO."
            ),
        },
        train_uids=train,
        test_uids=test,
        fit_uids=fit_uids,
        val_uids=val_uids,
        train_scene_keys=sorted({condition_key(u) for u in train}),
        test_scene_keys=sorted({condition_key(u) for u in test}),
        fit_scene_keys=sorted(fit_keys),
        val_scene_keys=sorted(val_keys),
    )


def partition_paths(
    dataset_root: str | Path,
    uids: list[str],
) -> tuple[list[str], np.ndarray]:
    """Resolve uid list → (absolute wav paths, speed labels)."""
    from .catalog import resolve_uid_path

    paths, speeds = [], []
    for u in uids:
        paths.append(str(resolve_uid_path(dataset_root, u)))
        speeds.append(float(uid_to_speed(u)))
    return paths, np.asarray(speeds, dtype=np.float64)
