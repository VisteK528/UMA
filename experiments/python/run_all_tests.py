import subprocess

scripts = [
    {"script": "credit_score_forest_id3_experiments.py", "dir": "credit_score/"},
    {"script": "credit_score_forest_id3_bayes_experiments.py", "dir": "credit_score/"},
    {"script": "diabetes_naive_bayes_experiments.py", "dir": "diabetes/"},
    {"script": "diabetes_id3_experiments.py", "dir": "diabetes/"},
    {"script": "diabetes_forest_id3_experiments.py", "dir": "diabetes/"},
    {"script": "diabetes_forest_id3_bayes_experiments.py", "dir": "diabetes/"},
    {"script": "healthcare_naive_bayes_experiments.py", "dir": "healthcare/"},
    {"script": "healthcare_id3_experiments.py", "dir": "healthcare/"},
    {"script": "healthcare_forest_id3_experiments.py", "dir": "healthcare/"},
    {"script": "healthcare_forest_id3_bayes_experiments.py", "dir": "healthcare/"},
]

for script_info in scripts:
    print(f"Running {script_info['script']} in {script_info['dir']}...")
    result = subprocess.run(
        ["python", script_info["script"]],
        cwd=script_info["dir"],
        capture_output=True,
        text=True
    )
    print(f"Output of {script_info['script']}:\n{result.stdout}")
    if result.returncode != 0:
        print(f"Error in {script_info['script']}:\n{result.stderr}")
        break
