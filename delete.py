import os
import shutil

# Specify the path to delete
path_to_delete = "/workspace/my_examples/example_antenna/outputs"

# Check if the directory exists
if os.path.exists(path_to_delete):
    try:
        shutil.rmtree(path_to_delete)
        print(f"Deleted: {path_to_delete}")
    except Exception as e:
        print(f"Error deleting {path_to_delete}: {e}")
else:
    print(f"Path not found: {path_to_delete}")
