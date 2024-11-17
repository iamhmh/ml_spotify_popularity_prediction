import subprocess

scripts = [
    "spotify_data.py",   # Data extraction from Spotify
    "prepare_data.py",   # Data preprocessing 
    "train_model.py"     # Model training
]

def run_scripts(scripts):
    for script in scripts:
        print(f"\n--- {script} in progress... ---")
        try:
            result = subprocess.run(["python", script], check=True, text=True, capture_output=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error: {script} failed.")
            print(e.stderr)
            break 

if __name__ == "__main__":
    print("Pipeline start.\n")
    run_scripts(scripts)
    print("\nPipeline finish.")
