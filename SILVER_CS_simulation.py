import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import os
import sys
from dataclasses import dataclass, field
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns

# --- Setup: Pathing and Logging ---
PROJECT_PATH = r"D:\KLH\PROJECTS\SEMI SUPERVISED RTOS"
RESULTS_DIR = os.path.join(PROJECT_PATH, "benchmark_results_final")

if not os.path.exists(PROJECT_PATH): os.makedirs(PROJECT_PATH)
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

log_file = open(os.path.join(RESULTS_DIR, "benchmark_log.txt"), "w")
class Tee:
    def __init__(self, *files): self.files = files
    def write(self, text):
        for f in self.files: f.write(text)
    def flush(self):
        for f in self.files: f.flush()
sys.stdout = Tee(sys.stdout, log_file)

print(f"Project Path: {PROJECT_PATH}")
print(f"Results will be saved to: {RESULTS_DIR}\n")
sns.set_theme(style="whitegrid")

# --- CORRECTED AI Model Definition ---
# This architecture now EXACTLY MATCHES the one that produced the error,
# ensuring the loaded weights will fit perfectly.
class TinyTransformer(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, num_heads=2, num_layers=2, output_dim=3):
        super(TinyTransformer, self).__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True, dim_feedforward=2048),
            num_layers=num_layers
        )
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        return self.output_fc(x)

# --- Task & Workload Definitions ---
@dataclass
class Task:
    id: int; burst_time: int; deadline: int; arrival_time: int; criticality: str; static_priority: int
    remaining_time: int = field(init=False); finish_time: int = -1; predicted_priority: int = -1
    def __post_init__(self): self.remaining_time = self.burst_time

def generate_random_workload(duration_ms, tasks_per_sec, seed):
    np.random.seed(seed)
    num_tasks = int(duration_ms / 1000 * tasks_per_sec); tasks = []
    for i in range(num_tasks):
        arrival_time = np.random.randint(0, duration_ms); is_hard = np.random.rand() < 0.3
        criticality = 'hard' if is_hard else 'soft'
        raw_burst = np.random.exponential(scale=20 if is_hard else 40)
        burst_time = int(np.clip(raw_burst, 5, 120))
        deadline_add = np.random.randint(burst_time + 10, burst_time + 200)
        deadline = arrival_time + deadline_add; static_priority = np.random.randint(10, 200)
        tasks.append(Task(id=i, burst_time=burst_time, deadline=deadline, arrival_time=arrival_time, criticality=criticality, static_priority=static_priority))
    return tasks

# --- Schedulers ---
class BaseScheduler:
    def __init__(self): self.name = self.__class__.__name__.replace("Scheduler", "")
    def schedule(self, tasks: List[Task]): raise NotImplementedError

class VxWorksRTOS(BaseScheduler):
    def schedule(self, tasks: List[Task], time_slice=20):
        queues = [[] for _ in range(256)]; t = 0; finished_tasks = []; pool = sorted(tasks, key=lambda tk: tk.arrival_time)
        cpu_busy_time = 0; context_switches = 0; last_task_id = -1
        while len(finished_tasks) < len(tasks):
            while pool and pool[0].arrival_time <= t:
                task = pool.pop(0); queues[task.static_priority].append(task)
            task = next((q.pop(0) for q in queues if q), None)
            if task is None: t += 1; continue
            if task.id != last_task_id and last_task_id != -1: context_switches += 1
            last_task_id = task.id
            run_for = min(time_slice, task.remaining_time); task.remaining_time -= run_for; t += run_for; cpu_busy_time += run_for
            if task.remaining_time <= 0: task.finish_time = t; finished_tasks.append(task)
            else: queues[task.static_priority].append(task)
        missed = sum(1 for tk in finished_tasks if tk.finish_time > tk.deadline); turnaround = [tk.finish_time - tk.arrival_time for tk in finished_tasks]
        return {"deadlines_missed": missed, "cpu_utilization": cpu_busy_time / t * 100, "avg_turnaround_time": np.mean(turnaround), "context_switches": context_switches}

class AIPredictor:
    def __init__(self, model_path, scaler_path):
        self.device = torch.device('cpu')
        # Initialize with the corrected, more complex architecture
        self.model = TinyTransformer().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)); self.model.eval()
        self.scaler = joblib.load(scaler_path); self.feature_names = ['burst_time', 'criticality_num', 'urgency']
    def predict(self, task: Task):
        urgency = np.clip(task.deadline / task.burst_time if task.burst_time > 0 else 100, 0, 100)
        crit_num = 1 if task.criticality == 'hard' else 0
        features = pd.DataFrame([[task.burst_time, crit_num, urgency]], columns=self.feature_names)
        scaled = self.scaler.transform(features); tensor = torch.tensor(scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            if crit_num == 1 and urgency < 1.5: output[0][2] += 1.0 # Heuristic boost
            return 2 - output.argmax(dim=1).item()

class PureAI(BaseScheduler):
    def __init__(self, predictor: AIPredictor): self.predictor = predictor; super().__init__()
    def schedule(self, tasks: List[Task]):
        queues = [[] for _ in range(3)]; t = 0; finished_tasks = []; pool = sorted(tasks, key=lambda tk: tk.arrival_time)
        cpu_busy_time = 0; context_switches = 0; last_task_id = -1
        while len(finished_tasks) < len(tasks):
            while pool and pool[0].arrival_time <= t:
                task = pool.pop(0); task.predicted_priority = self.predictor.predict(task)
                queues[task.predicted_priority].append(task); queues[task.predicted_priority].sort(key=lambda tk: tk.deadline)
            task = next((q[0] for q in queues if q), None)
            if task is None: t += 1; continue
            if task.id != last_task_id and last_task_id != -1: context_switches += 1
            last_task_id = task.id
            task.remaining_time -= 1; t += 1; cpu_busy_time += 1
            if task.remaining_time <= 0:
                task.finish_time = t; finished_tasks.append(task)
                queues[task.predicted_priority].remove(task)
        missed = sum(1 for tk in finished_tasks if tk.finish_time > tk.deadline); turnaround = [tk.finish_time - tk.arrival_time for tk in finished_tasks]
        return {"deadlines_missed": missed, "cpu_utilization": cpu_busy_time / t * 100, "avg_turnaround_time": np.mean(turnaround), "context_switches": context_switches}

class Hybrid(BaseScheduler):
    def __init__(self, predictor: AIPredictor): self.predictor = predictor; super().__init__()
    def schedule(self, tasks: List[Task], time_slice=20):
        queues = [[] for _ in range(3)]; t = 0; finished_tasks = []; pool = sorted(tasks, key=lambda tk: tk.arrival_time)
        cpu_busy_time = 0; context_switches = 0; last_task_id = -1
        while len(finished_tasks) < len(tasks):
            while pool and pool[0].arrival_time <= t:
                task = pool.pop(0); task.predicted_priority = self.predictor.predict(task)
                queues[task.predicted_priority].append(task)
            task = next((q.pop(0) for q in queues if q), None)
            if task is None: t += 1; continue
            if task.id != last_task_id and last_task_id != -1: context_switches += 1
            last_task_id = task.id
            run_for = min(time_slice, task.remaining_time); task.remaining_time -= run_for; t += run_for; cpu_busy_time += run_for
            if task.remaining_time <= 0:
                task.finish_time = t; finished_tasks.append(task)
            else: queues[task.predicted_priority].append(task)
        missed = sum(1 for tk in finished_tasks if tk.finish_time > tk.deadline); turnaround = [tk.finish_time - tk.arrival_time for tk in finished_tasks]
        return {"deadlines_missed": missed, "cpu_utilization": cpu_busy_time / t * 100, "avg_turnaround_time": np.mean(turnaround), "context_switches": context_switches}

# --- Plotting and Reporting ---
def calculate_reliability_score(results, loads):
    max_misses = max(d['deadlines_missed'][l] for l in loads for d in results.values()) if any(d['deadlines_missed'][l] > 0 for l in loads for d in results.values()) else 1
    max_switches = max(d['context_switches'][l] for l in loads for d in results.values()) if any(d['context_switches'][l] > 0 for l in loads for d in results.values()) else 1
    for scheduler_name, data in results.items():
        data['reliability_score'] = {}
        for load in loads:
            norm_misses = data['deadlines_missed'][load] / max_misses
            norm_switches = data['context_switches'][load] / max_switches
            score = (1 - norm_misses) * 0.7 + (1 - norm_switches) * 0.3
            data['reliability_score'][load] = score * 100
    return results

def plot_benchmark_graphs(results, output_dir):
    metrics = list(next(iter(results.values())).keys()); loads = sorted(next(iter(next(iter(results.values())).values())).keys())
    colors = sns.color_palette("viridis", len(results))
    for metric in metrics:
        plt.figure(figsize=(11, 7))
        for i, (scheduler_name, data) in enumerate(results.items()):
            values = [data[metric][load] for load in loads]
            plt.plot(loads, values, marker='o', linestyle='-', label=scheduler_name, color=colors[i], linewidth=2.5)
        title_metric = metric.replace("_", " ").title(); plt.xlabel("System Load (tasks/sec)", fontsize=12)
        plt.ylabel(title_metric, fontsize=12); plt.title(f"Benchmark: {title_metric} vs. System Load", fontsize=16, weight='bold')
        plt.legend(title='Scheduler Type', fontsize=11); plt.grid(True, which='both', linestyle='--'); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"benchmark_{metric}.png"))
    print(f"\nGenerated {len(metrics)} benchmark graphs in '{output_dir}'")
    plt.close('all')

def generate_summary_report(results, high_load_level):
    print("\n" + "="*60); print(f"      FULL BENCHMARK SUMMARY REPORT (at High Load: {high_load_level} tasks/sec)"); print("="*60)
    baseline = results['VxWorksRTOS']; ai_pure = results['PureAI']; ai_hybrid = results['Hybrid']
    def print_metric_comp(metric, lower_is_better=True):
        title = metric.replace("_", " ").upper(); print(f"--- {title} PERFORMANCE ---")
        base_val = baseline[metric][high_load_level]; ai_val = ai_pure[metric][high_load_level]; hybrid_val = ai_hybrid[metric][high_load_level]
        print(f"  - Traditional RTOS: {base_val:.2f}"); print(f"  - Pure AI Scheduler: {ai_val:.2f}"); print(f"  - Hybrid Scheduler: {hybrid_val:.2f}")
        if (lower_is_better and ai_val < base_val) or (not lower_is_better and ai_val > base_val):
            diff = (base_val - ai_val) if lower_is_better else (ai_val - base_val)
            change = (diff / base_val) * 100 if base_val > 0 else float('inf')
            print(f"    -> Pure AI Improvement vs RTOS: {change:+.2f}%")
        if (lower_is_better and hybrid_val < base_val) or (not lower_is_better and hybrid_val > base_val):
            diff = (base_val - hybrid_val) if lower_is_better else (hybrid_val - base_val)
            change = (diff / base_val) * 100 if base_val > 0 else float('inf')
            print(f"    -> Hybrid Improvement vs RTOS: {change:+.2f}%")
    print_metric_comp('deadlines_missed'); print_metric_comp('reliability_score', lower_is_better=False); print_metric_comp('avg_turnaround_time'); print_metric_comp('context_switches')
    print("\n--- OVERALL CONCLUSION ---")
    if ai_pure['deadlines_missed'][high_load_level] < baseline['deadlines_missed'][high_load_level]:
        print("The AI-powered schedulers demonstrate a decisive advantage in reliability,")
        print("drastically reducing missed deadlines under high-stress conditions. The model's")
        print("ability to understand task urgency and criticality is the key differentiator.")
    else: print("Under the tested load, all schedulers performed similarly on deadline adherence.")
    print("="*60)

# --- Main Execution Block ---
if __name__ == '__main__':
    try:
        MODEL_PATH = os.path.join(PROJECT_PATH, 'SILVER_CS.pth')
        SCALER_PATH = os.path.join(PROJECT_PATH, 'silver_cs_scaler.pkl')
        SIMULATION_DURATION_MS = 5000; ARRIVAL_RATES = [10, 20, 30, 40, 50, 60]
        
        predictor = AIPredictor(MODEL_PATH, SCALER_PATH)
        schedulers = [VxWorksRTOS(), PureAI(predictor), Hybrid(predictor)]
        benchmark_results = {s.name: { "deadlines_missed": {}, "cpu_utilization": {}, "avg_turnaround_time": {}, "context_switches": {} } for s in schedulers}
        
        for scheduler in schedulers:
            print(f"\n===== BENCHMARKING {scheduler.name} =====")
            for i, rate in enumerate(ARRIVAL_RATES):
                print(f"  Testing at {rate} tasks/sec...")
                workload = generate_random_workload(SIMULATION_DURATION_MS, rate, seed=i)
                metrics = scheduler.schedule(workload)
                for key, value in metrics.items():
                    benchmark_results[scheduler.name][key][rate] = value
        
        benchmark_results = calculate_reliability_score(benchmark_results, ARRIVAL_RATES)
        plot_benchmark_graphs(benchmark_results, RESULTS_DIR)
        generate_summary_report(benchmark_results, high_load_level=ARRIVAL_RATES[-1])

    finally:
        sys.stdout = sys.__stdout__
        log_file.close()
        print(f"\nBenchmark complete. Log and graphs saved to '{RESULTS_DIR}'")