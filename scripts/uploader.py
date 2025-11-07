from huggingface_hub import HfApi

api = HfApi()
repo_id = "ishro/t5-base-extreme-summarization"

# Create the repo (if it doesn't exist yet)
api.create_repo(repo_id=repo_id, exist_ok=True)

# List of your checkpoints
checkpoints_to_upload = [
    "../runs/t5_base/checkpoint-11851",
    "../runs/t5_base/checkpoint-23702",
    "../runs/t5_base/checkpoint-35553"
]

for checkpoint_folder in checkpoints_to_upload:
    # Get the folder name (e.g., "checkpoint-35553")
    folder_name_in_repo = checkpoint_folder.split("/")[-1] 
    
    print(f"Uploading {checkpoint_folder} to {folder_name_in_repo} in {repo_id}...")
    
    # Upload the folder, placing its contents in a subfolder on the Hub
    api.upload_folder(
        folder_path=checkpoint_folder,
        repo_id=repo_id,
        path_in_repo=folder_name_in_repo # This puts it in a subfolder
    )

print("All checkpoints uploaded.")