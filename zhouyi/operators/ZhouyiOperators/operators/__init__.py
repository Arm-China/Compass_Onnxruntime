import os

# set environment
new_path=os.path.dirname(os.path.abspath(__file__))
if "OPERATOR_PATH" in os.environ:
    old_path = os.environ["OPERATOR_PATH"]
    if new_path not in old_path:
        new_path = old_path + ":" + new_path
os.environ["OPERATOR_PATH"] = new_path