# data_multiwoz22.py
import os
import json
import re
from typing import List, Dict, Any, Tuple, Optional

from graph_structures import Case


class MultiWOZ22Loader:
    """
    Loads MultiWOZ 2.2 split directories:
      root/train/dialogues_*.json
      root/val/dialogues_*.json  (optional)
      root/test/dialogues_*.json
    """
    def __init__(self, root: str):
        self.root = root

    def load_split(self, split: str) -> List[dict]:
        split_dir = os.path.join(self.root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split dir not found: {split_dir}")

        dialogs: List[dict] = []
        # Keep deterministic file traversal for reproducible experiments.
        for fn in sorted(os.listdir(split_dir)):
            if fn.endswith(".json"):
                path = os.path.join(split_dir, fn)
                with open(path, "r", encoding="utf-8") as f:
                    dialogs.extend(json.load(f))
        return dialogs


class MultiWOZ22CaseBuilder:
    """
    Turns MultiWOZ system turns into CBR cases.

    Case definition (per SYSTEM turn):
      - Problem text: service + previous user utterance + compact state
      - Needs: a decomposed list used for stable matching (service token, user text, slot=value tokens)
      - Solution: ordered list of system actions (dialogue acts)
    """
    def __init__(
        self,
        root: str = "./MultiWOZ_2.2",
        include_state: bool = True,
        state_max_slots: int = 30,
        delex_action_values: bool = True,
    ):
        self.include_state = include_state
        self.state_max_slots = state_max_slots
        self.delex_action_values = bool(delex_action_values)
        self.dialog_acts = {}
        
        # Load dialogue acts from dialog_acts.json
        dialog_acts_path = os.path.join(root, "dialog_acts.json")
        if os.path.exists(dialog_acts_path):
            with open(dialog_acts_path, "r", encoding="utf-8") as f:
                self.dialog_acts = json.load(f)

    @staticmethod
    def _norm_text(s: str) -> str:
        s = (s or "").lower().strip()
        s = re.sub(r"\s+", " ", s)
        return s

    @staticmethod
    def _action_to_label(service: str, a: Dict[str, Any]) -> Optional[str]:
        act = a.get("act")
        slot = a.get("slot")
        vals = a.get("values", [])
        if not act:
            return None
        svc = (service or "unknown").lower()
        act = str(act).lower()
        slot = str(slot).lower() if slot is not None else ""

        if vals and len(vals) > 0:
            v0 = str(vals[0]).lower()
            return f"{svc}:{act}({slot}={v0})"
        if slot:
            return f"{svc}:{act}({slot})"
        return f"{svc}:{act}"

    def _extract_dialog_acts(self, dlg_id: str, turn_id: int) -> List[str]:
        """
        Extract dialogue acts from dialog_acts.json.
        Returns ordered list of action labels like "service:act(slot=value)".
        """
        actions = []
        dlg_key = dlg_id if dlg_id in self.dialog_acts else dlg_id.replace(".json", "") + ".json"
        
        if dlg_key not in self.dialog_acts:
            return []
        
        turn_acts = self.dialog_acts[dlg_key].get(str(turn_id), {}).get("dialog_act", {})
        
        for act_type, slots_list in turn_acts.items():
            # act_type is like "Service-ActName"
            parts = act_type.split("-", 1)
            if len(parts) == 2:
                service, act_name = parts
                service = service.lower()
                act_name = act_name.lower()
                
                # slots_list is a list of [slot, value] pairs
                if slots_list and len(slots_list) > 0:
                    # Create action label for each slot-value pair
                    for slot_val in slots_list:
                        if isinstance(slot_val, list) and len(slot_val) >= 2:
                            slot, value = slot_val[0], slot_val[1]
                            slot = str(slot).lower() if slot else ""
                            value = str(value).lower() if value else ""
                            if slot and value:
                                if self.delex_action_values:
                                    actions.append(f"{service}:{act_name}({slot})")
                                else:
                                    actions.append(f"{service}:{act_name}({slot}={value})")
                            elif value:
                                if self.delex_action_values:
                                    actions.append(f"{service}:{act_name}(val)")
                                else:
                                    actions.append(f"{service}:{act_name}(val={value})")
                        else:
                            # No slot-value pairs, just the action
                            actions.append(f"{service}:{act_name}")
                else:
                    # No slots, just the action
                    actions.append(f"{service}:{act_name}")
        
        return actions

    def _state_pairs(self, frames: List[dict]) -> List[Tuple[str, str]]:
        """
        Extract slot-values from frames[*].state.slot_values (or similar keys).
        """
        pairs: List[Tuple[str, str]] = []
        for fr in frames:
            st = fr.get("state", {}) or {}
            sv = st.get("slot_values") or st.get("slots_values") or st.get("slots") or {}
            if isinstance(sv, dict):
                for k, v in sv.items():
                    if isinstance(v, list) and v:
                        pairs.append((str(k).lower(), str(v[0]).lower()))
                    elif v not in (None, [], {}):
                        pairs.append((str(k).lower(), str(v).lower()))

        # de-dupe & cap
        seen = set()
        out = []
        for s, v in pairs:
            key = (s, v)
            if key in seen:
                continue
            seen.add(key)
            out.append((s, v))
            if len(out) >= self.state_max_slots:
                break
        return out

    @staticmethod
    def build_needs(service: str, user_utt: str, state_pairs: List[Tuple[str, str]]) -> List[str]:
        """
        Needs are domain-light tokens used for stable matching (paper analogue of 'questions').
        """
        needs: List[str] = []
        svc = (service or "unknown").lower()
        needs.append(f"service={svc}")

        if user_utt:
            needs.append(f"user={user_utt}")  # keep whole utterance

        for s, v in state_pairs:
            needs.append(f"need:{s}={v}")

        # de-dupe preserve order
        seen = set()
        out = []
        for n in needs:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def build_cases(self, dialogs: List[dict], split_name: str) -> List[Case]:
        cases: List[Case] = []
        for dlg in dialogs:
            dlg_id = dlg.get("dialogue_id", "unknown")
            turns = dlg.get("turns", [])

            prev_user_utt = ""
            for ti, turn in enumerate(turns):
                speaker = turn.get("speaker", "")
                utt = self._norm_text(turn.get("utterance", ""))

                if speaker == "USER":
                    prev_user_utt = utt
                    continue
                if speaker != "SYSTEM":
                    continue

                # Extract dialogue acts from the separate dialog_acts.json
                all_actions = self._extract_dialog_acts(dlg_id, ti)

                if not all_actions:
                    continue

                # Infer service from dialogue acts (first service seen)
                service_for_actions = "unknown"
                if all_actions:
                    # Try to extract service from first action (format: "service:act...")
                    first_act = all_actions[0]
                    if ":" in first_act:
                        service_for_actions = first_act.split(":")[0]
                
                # Get state from frames (for context)
                frames = turn.get("frames", [])
                state_pairs = self._state_pairs(frames) if self.include_state else []
                needs = self.build_needs(service_for_actions, prev_user_utt, state_pairs)

                # retrieval-friendly problem text
                parts = [f"service={service_for_actions}"]
                if prev_user_utt:
                    parts.append(f"user={prev_user_utt}")
                if self.include_state and state_pairs:
                    parts.append("state=" + " ".join([f"{s}={v}" for s, v in state_pairs]))
                problem_text = " | ".join(parts)

                cases.append(Case(
                    case_id=f"{split_name}:{dlg_id}:turn{ti}",
                    problem_text=problem_text,
                    needs=needs,
                    service=service_for_actions,
                    solution_actions=all_actions
                ))
        return cases
