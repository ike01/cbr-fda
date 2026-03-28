import json
import os

train_dir = "./MultiWOZ_2.2/train"

for fn in sorted(os.listdir(train_dir))[:1]:
    path = os.path.join(train_dir, fn)
    with open(path, "r") as f:
        dialogs = json.load(f)
    
    dlg = dialogs[0]
    turns = dlg.get("turns", [])
    
    user_turns = [t for t in turns if t.get("speaker") == "USER"]
    sys_turns = [t for t in turns if t.get("speaker") == "SYSTEM"]
    
    print(f"Dialogue TURNS: {len(user_turns)} USER, {len(sys_turns)} SYSTEM")
    print("\n=== First USER turn structure ===")
    print(json.dumps(user_turns[0], indent=2)[:600])
    
    if sys_turns:
        print("\n=== First SYSTEM turn structure ===")
        print(json.dumps(sys_turns[0], indent=2)[:600])
