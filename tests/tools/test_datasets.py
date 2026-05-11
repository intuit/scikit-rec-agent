"""Tests for create_datasets tool."""

from __future__ import annotations

import pandas as pd

from scikit_rec_agent.tools.datasets import TOOL_CREATE_DATASETS


def test_create_datasets_happy_path(binary_reward_paths, session):
    result = TOOL_CREATE_DATASETS.fn(
        bundle_id="b1",
        interactions_path=binary_reward_paths["interactions"],
        users_path=binary_reward_paths["users"],
        items_path=binary_reward_paths["items"],
        session=session,
    )
    assert result["status"] == "ok"
    bundle = session.loaded_datasets["b1"]
    assert bundle.interactions is not None
    assert bundle.users is not None
    assert bundle.items is not None
    assert bundle.valid_interactions is None


def test_create_datasets_with_column_mapping(tmp_path, session):
    # Write a CSV with non-canonical column names.
    df = pd.DataFrame(
        {
            "userid": ["u0", "u1", "u2"],
            "itemid": ["i0", "i1", "i2"],
            "clicked": [1.0, 0.0, 1.0],
        }
    )
    p = tmp_path / "bad_interactions.csv"
    df.to_csv(p, index=False)

    result = TOOL_CREATE_DATASETS.fn(
        bundle_id="mapped",
        interactions_path=str(p),
        session=session,
        column_mapping={"userid": "USER_ID", "itemid": "ITEM_ID", "clicked": "OUTCOME"},
    )
    assert result["status"] == "ok"
    assert set(result["data"]["columns"]) == {"USER_ID", "ITEM_ID", "OUTCOME"}


def test_create_datasets_file_not_found(session):
    result = TOOL_CREATE_DATASETS.fn(
        bundle_id="bad",
        interactions_path="/nonexistent.csv",
        session=session,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "FileNotFoundError"


def test_create_datasets_auto_detects_multioutput(tmp_path, session):
    """Wide multi-output: USER_ID + ≥2 ITEM_* columns, no ITEM_ID/OUTCOME →
    create_datasets must construct an InteractionMultiOutputDataset, not the
    default InteractionsDataset (which would reject the schema)."""
    df = pd.DataFrame(
        {
            "USER_ID": ["u0", "u1", "u2"],
            "feat1": [1, 2, 3],
            "ITEM_payroll": [1.0, 0.0, 1.0],
            "ITEM_invoice": [0.0, 1.0, 1.0],
            "ITEM_budget": [1.0, 1.0, 0.0],
        }
    )
    p = tmp_path / "wide.csv"
    df.to_csv(p, index=False)

    result = TOOL_CREATE_DATASETS.fn(
        bundle_id="wide",
        interactions_path=str(p),
        session=session,
    )
    assert result["status"] == "ok", result
    assert result["data"]["dataset_type"] == "interaction_multioutput"
    bundle = session.loaded_datasets["wide"]
    assert bundle.dataset_type == "interaction_multioutput"
    # The bundle's interactions handle is the right scikit-rec subclass.
    from skrec.dataset.interactions_dataset import InteractionMultiOutputDataset

    assert isinstance(bundle.interactions, InteractionMultiOutputDataset)


def test_create_datasets_auto_detects_multiclass(tmp_path, session):
    """Wide multi-class: USER_ID + ITEM_ID, no OUTCOME → InteractionMultiClassDataset."""
    df = pd.DataFrame(
        {
            "USER_ID": ["u0", "u1", "u2"],
            "ITEM_ID": ["A", "B", "A"],
            "feat1": [1, 2, 3],
        }
    )
    p = tmp_path / "mc.csv"
    df.to_csv(p, index=False)

    result = TOOL_CREATE_DATASETS.fn(
        bundle_id="mc",
        interactions_path=str(p),
        session=session,
    )
    assert result["status"] == "ok", result
    assert result["data"]["dataset_type"] == "interaction_multiclass"
    from skrec.dataset.interactions_dataset import InteractionMultiClassDataset

    assert isinstance(session.loaded_datasets["mc"].interactions, InteractionMultiClassDataset)


def test_create_datasets_explicit_dataset_type_override(tmp_path, session):
    """Caller can force the dataset_type rather than relying on auto-detection."""
    df = pd.DataFrame(
        {
            "USER_ID": ["u0", "u1"],
            "ITEM_payroll": [1.0, 0.0],
            "ITEM_invoice": [0.0, 1.0],
        }
    )
    p = tmp_path / "wide.csv"
    df.to_csv(p, index=False)

    result = TOOL_CREATE_DATASETS.fn(
        bundle_id="wide",
        interactions_path=str(p),
        session=session,
        dataset_type="interaction_multioutput",
    )
    assert result["status"] == "ok"
    assert result["data"]["dataset_type"] == "interaction_multioutput"


def test_create_datasets_invalid_dataset_type_returns_error(tmp_path, session, binary_reward_paths):
    result = TOOL_CREATE_DATASETS.fn(
        bundle_id="x",
        interactions_path=binary_reward_paths["interactions"],
        session=session,
        dataset_type="bogus",
    )
    assert result["status"] == "error"
    assert result["error_type"] == "InvalidDatasetType"


def test_create_datasets_long_format_still_defaults_to_interactions(binary_reward_paths, session):
    """Auto-detection on a vanilla long-format file picks `interactions`,
    keeping the original behaviour."""
    result = TOOL_CREATE_DATASETS.fn(
        bundle_id="b",
        interactions_path=binary_reward_paths["interactions"],
        session=session,
    )
    assert result["status"] == "ok"
    assert result["data"]["dataset_type"] == "interactions"


def test_check_user_id_overlap_uses_full_intersection_not_first_10k(tmp_path):
    """previously used .unique()[:10000] which respects first-seen order.
    On >10k-user frames where the real overlap lived past the first 10k of
    either side, the guard false-positived. Full set intersection fixes it.
    """
    import pandas as pd

    from scikit_rec_agent.tools.datasets import _check_user_id_overlap

    # 15k unique IDs on each side, but the overlap is entirely past index
    # 10k. The OLD code would sample [0..9999] from each side and report
    # 'no overlap' — a false positive that wasted the user's time.
    inter_ids = [f"INT_only_{i}" for i in range(10500)] + [f"SHARED_{i}" for i in range(4500)]
    user_ids = [f"USER_only_{i}" for i in range(10500)] + [f"SHARED_{i}" for i in range(4500)]
    inter_df = pd.DataFrame({"USER_ID": inter_ids})
    u_df = pd.DataFrame({"USER_ID": user_ids})

    result = _check_user_id_overlap(inter_df, u_df, context="test")
    # 4500 shared IDs at index >=10500 — full intersection sees them; old
    # truncation-to-10k would have missed them.
    assert result is None, "overlap check should have found shared IDs past index 10k"


def test_check_user_id_overlap_still_rejects_true_disjoint_sets():
    """Sanity: the overlap check still catches actual disjoint frames
    (the original bug it was written to prevent). Without this, the
    full-intersection change might over-correct."""
    import pandas as pd

    from scikit_rec_agent.tools.datasets import _check_user_id_overlap

    inter_df = pd.DataFrame({"USER_ID": ["company_a", "company_b", "company_c"]})
    u_df = pd.DataFrame({"USER_ID": ["180", "270", "365"]})  # mistaken feature-as-id

    result = _check_user_id_overlap(inter_df, u_df, context="test")
    assert result is not None
    assert result["error_type"] == "UserIdOverlapZero"
    assert result["category"] == "user_id_overlap_zero"


def test_resolve_interactions_path_prefers_merged(binary_reward_paths, session):
    """after auto-merge fires, source_paths['interactions'] continues
    to point at the user-provided file (back-compat), and a new
    source_paths['merged_interactions'] key carries the merged path.
    Internal consumers route through _resolve_interactions_path which
    prefers the merged path when present.
    """
    import pandas as pd

    from scikit_rec_agent.tools.datasets import (
        TOOL_CREATE_DATASETS,
        _resolve_interactions_path,
    )

    # Wide multi-output frame + separate users frame — triggers auto-merge.
    inter_df = pd.DataFrame(
        {
            "USER_ID": ["u1", "u2", "u3"],
            "ITEM_a": [1, 0, 1],
            "ITEM_b": [0, 1, 1],
        }
    )
    users_df = pd.DataFrame({"USER_ID": ["u1", "u2", "u3"], "feat1": [10, 20, 30]})

    inter_p = "/tmp/_r3_inter.csv"
    users_p = "/tmp/_r3_users.csv"
    inter_df.to_csv(inter_p, index=False)
    users_df.to_csv(users_p, index=False)

    result = TOOL_CREATE_DATASETS.fn(
        bundle_id="r3",
        interactions_path=inter_p,
        users_path=users_p,
        session=session,
    )
    assert result["status"] == "ok", result

    bundle = session.loaded_datasets["r3"]
    sp = bundle.source_paths

    # source_paths["interactions"] = user-provided path (NOT the merged one).
    # Previously, auto-merge silently overwrote this key with the merged
    # path, losing the original. Pin the new contract.
    assert sp["interactions"] == inter_p

    # merged_interactions key exists and points at the merged frame.
    assert "merged_interactions" in sp
    merged_df = pd.read_csv(sp["merged_interactions"])
    assert "feat1" in merged_df.columns

    # Helper picks the merged path when present so internal consumers
    # (sweep profile, split, eval) read the data the dataset object uses.
    assert _resolve_interactions_path(sp) == sp["merged_interactions"]


def test_resolve_interactions_path_falls_back_to_original_without_merge(binary_reward_paths, session):
    """For non-wide bundles (no auto-merge), source_paths['merged_interactions']
    is absent; _resolve_interactions_path returns source_paths['interactions'].
    Pin the no-merge fallback so the helper doesn't accidentally hide a real
    missing-path bug on long bundles.
    """
    from scikit_rec_agent.tools.datasets import (
        TOOL_CREATE_DATASETS,
        _resolve_interactions_path,
    )

    TOOL_CREATE_DATASETS.fn(
        bundle_id="lb",
        interactions_path=binary_reward_paths["interactions"],
        session=session,
    )
    bundle = session.loaded_datasets["lb"]
    sp = bundle.source_paths
    assert "merged_interactions" not in sp
    assert _resolve_interactions_path(sp) == sp["interactions"]


def test_create_datasets_coerces_integer_outcome_to_float(tmp_path, session):
    # Regression: binary click logs arrive as OUTCOME in {0, 1}, which pandas
    # infers as int64. scikit-rec's required interactions schema declares
    # OUTCOME: float and rejects schemas where OUTCOME is any other type. The
    # auto-schema generator must force the required type regardless of dtype.
    import pandas as pd

    df = pd.DataFrame(
        {
            "USER_ID": ["u0", "u1", "u2"],
            "ITEM_ID": ["i0", "i1", "i2"],
            "OUTCOME": [0, 1, 1],  # int64 dtype
        }
    )
    p = tmp_path / "integer_outcome.csv"
    df.to_csv(p, index=False)

    result = TOOL_CREATE_DATASETS.fn(bundle_id="int_outcome", interactions_path=str(p), session=session)
    assert result["status"] == "ok", result
    assert "int_outcome" in session.loaded_datasets
