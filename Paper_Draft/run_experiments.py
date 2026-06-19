import subprocess
import json
import os

def run_cmd(cmd):
    print(f"\n>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
    return result.stdout

def main():
    # 1. Train Baseline
    print("Starting Baseline Training...")
    run_cmd(["python", "train_mewc_llm.py", "baseline"])
    
    # 2. Train MEWC
    print("\nStarting MEWC-LLM Training...")
    run_cmd(["python", "train_mewc_llm.py", "mewc"])
    
    # 3. Evaluate Base
    print("\nEvaluating Base Model...")
    run_cmd(["python", "evaluate_mewc.py"])
    
    # 4. Evaluate Baseline
    print("\nEvaluating Baseline Model...")
    run_cmd(["python", "evaluate_mewc.py", "./baseline_lora_weights"])
    
    # 5. Evaluate MEWC
    print("\nEvaluating MEWC-LLM Model...")
    run_cmd(["python", "evaluate_mewc.py", "./mewc_lora_weights"])
    
    # Consolidate results
    final_results = {}
    for res_file in ["results_base.json", "results_baseline_lora_weights.json", "results_mewc_lora_weights.json"]:
        if os.path.exists(res_file):
            with open(res_file, "r") as f:
                name = res_file.replace("results_", "").replace(".json", "")
                final_results[name] = json.load(f)
    
    print("\n" + "="*50)
    print("CONSOLIDATED EXPERIMENT RESULTS")
    print("="*50)
    print(f"{'Configuration':<20} | {'Factual PPL':<12} | {'Adaptation PPL':<14}")
    print("-" * 50)
    
    for config, metrics in final_results.items():
        f_ppl = metrics.get("factual_ppl", 0)
        a_ppl = metrics.get("adaptation_ppl", 0)
        print(f"{config:<20} | {f_ppl:<12.4f} | {a_ppl:<14.4f}")
    
    with open("final_experiment_summary.json", "w") as f:
        json.dump(final_results, f, indent=4)
    print("\nSummary saved to final_experiment_summary.json")

if __name__ == "__main__":
    main()
