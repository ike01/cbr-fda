import json
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

from data_multiwoz22 import MultiWOZ22CaseBuilder, MultiWOZ22Loader
from data_onet import ONetCaseBuilder
from graph_structures import Case


def _collect_onet_raw_case_pairs(dataset_path: str) -> List[Tuple[Dict[str, Any], Case]]:
    builder = ONetCaseBuilder(lowercase=True, dedupe_lists=True)
    records = builder.load_records(dataset_path)
    pairs: List[Tuple[Dict[str, Any], Case]] = []
    for rec in records:
        built = builder.build_cases([rec], split_name="onet")
        if built:
            pairs.append((rec, built[0]))
    return pairs


def _parse_multiwoz_case_id(case_id: str) -> Tuple[str, str, int]:
    split, rest = case_id.split(":", 1)
    dlg_id, turn_idx_str = rest.rsplit(":turn", 1)
    return split, dlg_id, int(turn_idx_str)


def _find_prev_user_utt(turns: List[Dict[str, Any]], turn_idx: int) -> str:
    for i in range(turn_idx - 1, -1, -1):
        if turns[i].get("speaker") == "USER":
            return turns[i].get("utterance", "")
    return ""


def _collect_multiwoz_raw_case_pairs(root: str) -> List[Tuple[Dict[str, Any], Case]]:
    loader = MultiWOZ22Loader(root)
    builder = MultiWOZ22CaseBuilder(root=root, include_state=True)

    all_pairs: List[Tuple[Dict[str, Any], Case]] = []
    split_maps: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for split in ("train", "test", "val", "dev"):
        try:
            dialogs = loader.load_split(split)
        except Exception:
            continue
        split_maps[split] = {d.get("dialogue_id", "unknown"): d for d in dialogs}
        split_cases = builder.build_cases(dialogs, split_name=split)

        for c in split_cases:
            c_split, dlg_id, turn_idx = _parse_multiwoz_case_id(c.case_id)
            dlg = split_maps.get(c_split, {}).get(dlg_id)
            if dlg is None:
                continue
            turns = dlg.get("turns", [])
            if turn_idx < 0 or turn_idx >= len(turns):
                continue

            dlg_key = dlg_id if dlg_id in builder.dialog_acts else dlg_id.replace(".json", "") + ".json"
            raw_acts = builder.dialog_acts.get(dlg_key, {}).get(str(turn_idx), {}).get("dialog_act", {})

            raw_payload = {
                "split": c_split,
                "dialogue_id": dlg_id,
                "turn_index": turn_idx,
                "previous_user_utterance": _find_prev_user_utt(turns, turn_idx),
                "system_turn": turns[turn_idx],
                "raw_dialog_act": raw_acts,
            }
            all_pairs.append((raw_payload, c))

    return all_pairs


def display_raw_and_case(
    index: int,
    dataset_name: str,
    multiwoz_root: str = "./MultiWOZ_2.2",
    onet_dataset_path: str = "./onet_data/dataset_tasks_to_skills.json",
) -> Tuple[Dict[str, Any], Case]:
    """
    Print raw dataset item + built case for quick demonstrations.

    Args:
        index: Zero-based index into the built case list for the selected dataset.
        dataset_name: "multiwoz" or "onet" (case-insensitive).
    """
    ds = (dataset_name or "").strip().lower()
    if ds == "multiwoz":
        pairs = _collect_multiwoz_raw_case_pairs(multiwoz_root)
    elif ds == "onet":
        pairs = _collect_onet_raw_case_pairs(onet_dataset_path)
    else:
        raise ValueError("dataset_name must be either 'multiwoz' or 'onet'.")

    if not pairs:
        raise RuntimeError(f"No displayable pairs found for dataset '{ds}'.")
    if index < 0 or index >= len(pairs):
        raise IndexError(f"index={index} is out of range [0, {len(pairs) - 1}].")

    raw_item, case_item = pairs[index]
    print(f"Dataset: {ds} | Index: {index} | Total cases: {len(pairs)}")
    print("=== RAW DATA ===")
    print(json.dumps(raw_item, indent=2, ensure_ascii=False))
    print("=== CASE EQUIVALENT ===")
    print(json.dumps(asdict(case_item), indent=2, ensure_ascii=False))
    return raw_item, case_item


if __name__ == "__main__":
    # Example manual demo run:
    #   python misc.py
    display_raw_and_case(20, "multiwoz")
