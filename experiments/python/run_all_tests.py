import subprocess

scripts = [
    {"script": "wine_id3_experiments.py", "dir": "wine/"},
    {"script": "wine_forest_id3_experiments.py", "dir": "wine/"},
    {"script": "wine_forest_id3_bayes_experiments.py", "dir": "wine/"},
    {"script": "credit_score_naive_bayes_experiments", "dir": "credit_score/"},
    {"script": "credit_score_naive_bayes_experiments", "dir": "credit_score/"},
    {"script": "credit_score_naive_bayes_experiments", "dir": "credit_score/"},
    {"script": "credit_score_naive_bayes_experiments", "dir": "credit_score/"},
    {"script": "diabetes_naive_bayes_experiments.py", "dir": "diabetes/"},
    {"script": "diabetes_id3_experiments.py", "dir": "diabetes/"},
    {"script": "diabetes_forest_id3_experiments.py", "dir": "diabetes/"},
    {"script": "diabetes_forest_id3_bayes_experiments.py", "dir": "diabetes/"},
    {"script": "healthcare_naive_bayes_experiments.py", "dir": "healthcare/"},
    {"script": "healthcare_id3_experiments.py", "dir": "healthcare/"},
    {"script": "healthcare_forest_id3_experiments.py", "dir": "healthcare/"},
    {"script": "healthcare_forest_id3_bayes_experiments.py", "dir": "healthcare/"},
]

# Run scripts sequentially
for script_info in scripts:
    print(f"Running {script_info['script']} in {script_info['dir']}...")
    result = subprocess.run(
        ["python", script_info["script"]],  # Command to execute
        cwd=script_info["dir"],            # Directory to execute the script from
        capture_output=True,               # Capture output (optional)
        text=True                          # Output as string
    )
    print(f"Output of {script_info['script']}:\n{result.stdout}")
    if result.returncode != 0:  # Check if the script exited with an error
        print(f"Error in {script_info['script']}:\n{result.stderr}")
        break  # Stop further execution if an error occurs
