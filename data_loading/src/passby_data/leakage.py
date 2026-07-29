"""Hard leakage asserts for pass-by audio splits.

Checks covered
--------------
1. Train ∩ test uid nonempty (clip leakage)
2. Train ∩ test scene-key nonempty (scene leakage; includes synth twins)
3. Fit / val intersect outer test scenes
4. Fit ∩ val nonempty
5. Mel-stats path list intersects val or test (normalization leakage)
6. LOVO: any train vehicle equals held-out vehicle
7. Eval uids whose scene key appears in train
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .catalog import condition_key, normalize_uid, uid_to_vehicle
from .splits import SplitBundle


@dataclass
class LeakageReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValueError(
                "Data leakage detected:\n  - " + "\n  - ".join(self.errors)
            )


def _norm_list(uids: Iterable[str] | None) -> list[str]:
    if not uids:
        return []
    return [normalize_uid(u) for u in uids]


def leakage_report(
    split: SplitBundle,
    *,
    stats_uids: Sequence[str] | None = None,
    eval_uids: Sequence[str] | None = None,
    seresnet_train_uids: Sequence[str] | None = None,
    dit_train_uids: Sequence[str] | None = None,
    check_adjacent_speed: bool = True,
) -> LeakageReport:
    errors: list[str] = []
    warnings: list[str] = []

    train = set(_norm_list(split.train_uids))
    test = set(_norm_list(split.test_uids))
    synth_train = set(_norm_list(split.synth_train_uids))
    synth_test = set(_norm_list(split.synth_test_uids))
    fit = set(_norm_list(split.fit_uids) or list(train))
    val = set(_norm_list(split.val_uids))

    test_keys = set(split.test_scene_keys) or {condition_key(u) for u in test}
    if synth_test:
        test_keys |= {condition_key(u) for u in synth_test}
    train_keys = set(split.train_scene_keys) or (
        {condition_key(u) for u in train} | {condition_key(u) for u in synth_train}
    )

    # --- clip overlap ---
    if train & test:
        errors.append(f"real train∩test nonempty ({len(train & test)})")
    if synth_train & synth_test:
        errors.append(f"synth train∩test nonempty ({len(synth_train & synth_test)})")
    if fit & val:
        errors.append(f"fit∩val nonempty ({len(fit & val)})")
    if fit & test:
        errors.append(f"fit∩test nonempty ({len(fit & test)}")
    if val & test:
        errors.append(f"val∩test nonempty ({len(val & test)})")

    # --- scene overlap ---
    for name, uids in (
        ("real_train", train),
        ("synth_train", synth_train),
        ("fit", fit),
        ("val", val),
    ):
        leak = [u for u in uids if condition_key(u) in test_keys]
        if leak:
            errors.append(
                f"{name} has {len(leak)} clips with test scene keys, e.g. {leak[0]}"
            )

    train_all_keys = {condition_key(u) for u in train | synth_train | fit}
    overlap = train_all_keys & test_keys
    if overlap:
        errors.append(
            f"train/test scene-key overlap ({len(overlap)}), e.g. {next(iter(overlap))}"
        )

    if split.fit_scene_keys and split.val_scene_keys:
        if set(split.fit_scene_keys) & set(split.val_scene_keys):
            errors.append("inner fit/val scene-key overlap")
        if (set(split.fit_scene_keys) | set(split.val_scene_keys)) & test_keys:
            errors.append("inner fit/val intersects outer test scenes")

    # --- LOVO ---
    if split.protocol == "lovo" and split.held_out_vehicle:
        bad = [u for u in train if uid_to_vehicle(u) == split.held_out_vehicle]
        if bad:
            errors.append(
                f"LOVO: train contains held-out vehicle {split.held_out_vehicle!r}"
            )

    # --- mel stats must not see val/test ---
    if stats_uids is not None:
        stats_set = set(_norm_list(stats_uids))
        if stats_set & test:
            errors.append(
                f"mel stats include {len(stats_set & test)} outer-test clips"
            )
        if stats_set & val:
            errors.append(
                f"mel stats include {len(stats_set & val)} inner-val clips "
                "(fit stats on fit_uids only)"
            )
        # synth test scenes
        bad_synth = [u for u in stats_set if condition_key(u) in test_keys]
        if bad_synth:
            errors.append(
                f"mel stats include {len(bad_synth)} clips with test scene keys"
            )

    # --- optional DiT / SE-ResNet train lists ---
    if dit_train_uids is not None:
        dit = set(_norm_list(dit_train_uids))
        if dit & test:
            errors.append(f"DiT train intersects Test_R ({len(dit & test)})")
        if dit - train:
            errors.append(
                f"DiT train has {len(dit - train)} uids outside Train_R"
            )
        if train - dit:
            errors.append(
                f"DiT train missing {len(train - dit)} Train_R uids"
            )

    if seresnet_train_uids is not None:
        se = set(_norm_list(seresnet_train_uids))
        if se & test:
            errors.append(f"SE-ResNet train intersects Test_R ({len(se & test)})")
        bad = [u for u in se if condition_key(u) in test_keys]
        if bad:
            errors.append(
                f"SE-ResNet train has {len(bad)} clips with test scene keys"
            )

    # --- eval set must be holdout only ---
    if eval_uids is not None:
        held = test | synth_test
        outside = set(_norm_list(eval_uids)) - held
        if not synth_test:
            outside = set(_norm_list(eval_uids)) - test
        if outside:
            still_bad = [
                u for u in outside if condition_key(u) not in test_keys
            ]
            if still_bad:
                errors.append(
                    f"Eval has {len(still_bad)} uids outside held-out partitions, "
                    f"e.g. {still_bad[0]}"
                )
        bad = [u for u in _norm_list(eval_uids) if condition_key(u) in train_keys]
        if bad:
            errors.append(
                f"Eval has {len(bad)} clips whose scene key is in train, e.g. {bad[0]}"
            )

    # --- adjacent-speed near-duplicates ---
    gap = int((split.meta or {}).get("min_speed_gap", 1) or 1)
    if check_adjacent_speed and split.protocol == "scene":
        from .splits import adjacent_twin_uids

        soft_bad = adjacent_twin_uids(
            sorted(train), sorted(test), min_speed_gap=2
        )
        if gap <= 1 and soft_bad:
            warnings.append(
                f"{len(soft_bad)} test scenes have a +/-1 km/h twin in train for "
                "the same vehicle (near-duplicate risk; not hard clip leakage). "
                "Pass --min-speed-gap 2 / scripts/make_spread_scene_split.py to ban them, "
                "or use LOVO for unknown-car claims."
            )
        if gap > 1:
            hard_bad = adjacent_twin_uids(
                sorted(train), sorted(test), min_speed_gap=gap
            )
            if hard_bad:
                errors.append(
                    f"min_speed_gap={gap} violated by {len(hard_bad)} test clips, "
                    f"e.g. {hard_bad[0]}"
                )

    if split.protocol == "scene":
        both = {uid_to_vehicle(u) for u in train} & {uid_to_vehicle(u) for u in test}
        if both:
            warnings.append(
                f"Scene protocol: {len(both)} vehicles appear in both train and "
                "test (expected). Do not report this as LOVO. "
                "Use scripts/make_lovo_split.py for unknown-vehicle folds."
            )

    return LeakageReport(ok=not errors, errors=errors, warnings=warnings)


def assert_no_leakage(
    split: SplitBundle,
    *,
    stats_uids: Sequence[str] | None = None,
    eval_uids: Sequence[str] | None = None,
    seresnet_train_uids: Sequence[str] | None = None,
    dit_train_uids: Sequence[str] | None = None,
) -> LeakageReport:
    report = leakage_report(
        split,
        stats_uids=stats_uids,
        eval_uids=eval_uids,
        seresnet_train_uids=seresnet_train_uids,
        dit_train_uids=dit_train_uids,
    )
    report.raise_if_failed()
    return report
