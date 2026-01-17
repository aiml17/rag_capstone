"""
RGB Results Report Generator
Generates formatted reports from RGB evaluation results JSON files.

Usage:
    python rgb_report_generator.py                           # Use latest results file
    python rgb_report_generator.py --file results/rgb/rgb_results_20260113_002930.json
    python rgb_report_generator.py --html                    # Generate HTML report
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("results/rgb")

# Models list (should match evaluator)
LLM_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
]

NOISE_RATIOS = [0, 0.2, 0.4, 0.6, 0.8]
INTEGRATION_NOISE_RATIOS = [0, 0.2, 0.4]


def find_latest_results() -> Path:
    """Find the most recent results file."""
    results_files = list(RESULTS_DIR.glob("rgb_results_*.json"))
    if not results_files:
        raise FileNotFoundError(f"No results files found in {RESULTS_DIR}")
    return max(results_files, key=lambda p: p.stat().st_mtime)


def load_results(filepath: Path) -> list:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def print_console_report(results: list, filepath: Path):
    """Print formatted report to console."""

    # Group results by ability
    noise_results = [r for r in results if r["ability"] == "noise_robustness"]
    rejection_results = [r for r in results if r["ability"] == "negative_rejection"]
    integration_results = [r for r in results if r["ability"] == "information_integration"]
    counterfactual_results = [r for r in results if r["ability"] == "counterfactual_robustness"]

    print("\n" + "=" * 80)
    print("📊 RGB BENCHMARK EVALUATION REPORT")
    print("=" * 80)
    print(f"📁 Results file: {filepath}")
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 Models evaluated: {len(set(r['model'] for r in results))}")
    print(f"📈 Total evaluations: {len(results)}")

    # Table 1: Noise Robustness
    print("\n" + "=" * 80)
    print("📈 TABLE 1: Noise Robustness (Accuracy %)")
    print("=" * 80)
    print(f"{'Model':<40} | " + " | ".join([f"{r:>5}" for r in NOISE_RATIOS]) + " | Avg")
    print("-" * 85)

    for model in LLM_MODELS:
        model_results = [r for r in noise_results if r["model"] == model]
        if model_results:
            row = f"{model:<40} | "
            accuracies = []
            for nr in NOISE_RATIOS:
                match = next((r for r in model_results if r["metrics"].get("noise_ratio") == nr), None)
                if match:
                    acc = match['metrics']['accuracy']
                    accuracies.append(acc)
                    row += f"{acc:>5.1f} | "
                else:
                    row += "  N/A | "
            avg = sum(accuracies) / len(accuracies) if accuracies else 0
            row += f"{avg:>5.1f}"
            print(row)

    # Table 2: Negative Rejection
    print("\n" + "=" * 80)
    print("📈 TABLE 2: Negative Rejection (Rejection Rate %)")
    print("=" * 80)
    print(f"{'Model':<40} | {'Rejection Rate':>15}")
    print("-" * 60)
    for r in rejection_results:
        print(f"{r['model']:<40} | {r['metrics']['rejection_rate']:>14.1f}%")

    # Table 3: Information Integration
    print("\n" + "=" * 80)
    print("📈 TABLE 3: Information Integration (Accuracy %)")
    print("=" * 80)
    print(f"{'Model':<40} | " + " | ".join([f"{r:>5}" for r in INTEGRATION_NOISE_RATIOS]) + " | Avg")
    print("-" * 70)

    for model in LLM_MODELS:
        model_results = [r for r in integration_results if r["model"] == model]
        if model_results:
            row = f"{model:<40} | "
            accuracies = []
            for nr in INTEGRATION_NOISE_RATIOS:
                match = next((r for r in model_results if r["metrics"].get("noise_ratio") == nr), None)
                if match:
                    acc = match['metrics']['accuracy']
                    accuracies.append(acc)
                    row += f"{acc:>5.1f} | "
                else:
                    row += "  N/A | "
            avg = sum(accuracies) / len(accuracies) if accuracies else 0
            row += f"{avg:>5.1f}"
            print(row)

    # Table 4: Counterfactual Robustness
    print("\n" + "=" * 80)
    print("📈 TABLE 4: Counterfactual Robustness (%)")
    print("=" * 80)
    print(f"{'Model':<40} | {'Acc':>6} | {'Acc_doc':>7} | {'ED':>6} | {'CR':>6}")
    print("-" * 75)
    for r in counterfactual_results:
        m = r["metrics"]
        print(
            f"{r['model']:<40} | {m['acc']:>6.1f} | {m['acc_doc']:>7.1f} | {m['error_detection']:>6.1f} | {m['error_correction']:>6.1f}")

    # Summary & Rankings
    print("\n" + "=" * 80)
    print("🏆 MODEL RANKINGS BY ABILITY")
    print("=" * 80)

    # Noise Robustness ranking (by average)
    print("\n📊 Noise Robustness (by average accuracy):")
    noise_avgs = []
    for model in LLM_MODELS:
        model_results = [r for r in noise_results if r["model"] == model]
        if model_results:
            avg = sum(r['metrics']['accuracy'] for r in model_results) / len(model_results)
            noise_avgs.append((model, avg))
    for i, (model, avg) in enumerate(sorted(noise_avgs, key=lambda x: -x[1]), 1):
        print(f"   {i}. {model}: {avg:.1f}%")

    # Negative Rejection ranking
    print("\n📊 Negative Rejection (by rejection rate):")
    rejection_sorted = sorted(rejection_results, key=lambda x: -x['metrics']['rejection_rate'])
    for i, r in enumerate(rejection_sorted, 1):
        print(f"   {i}. {r['model']}: {r['metrics']['rejection_rate']:.1f}%")

    # Information Integration ranking
    print("\n📊 Information Integration (by average accuracy):")
    int_avgs = []
    for model in LLM_MODELS:
        model_results = [r for r in integration_results if r["model"] == model]
        if model_results:
            avg = sum(r['metrics']['accuracy'] for r in model_results) / len(model_results)
            int_avgs.append((model, avg))
    for i, (model, avg) in enumerate(sorted(int_avgs, key=lambda x: -x[1]), 1):
        print(f"   {i}. {model}: {avg:.1f}%")

    # Counterfactual ranking (by error correction)
    print("\n📊 Counterfactual Robustness (by error correction rate):")
    cf_sorted = sorted(counterfactual_results, key=lambda x: -x['metrics']['error_correction'])
    for i, r in enumerate(cf_sorted, 1):
        print(f"   {i}. {r['model']}: CR={r['metrics']['error_correction']:.1f}%")

    # Overall summary
    print("\n" + "=" * 80)
    print("📋 KEY FINDINGS")
    print("=" * 80)

    # Best performers
    best_noise = max(noise_avgs, key=lambda x: x[1])
    best_rejection = max(rejection_results, key=lambda x: x['metrics']['rejection_rate'])
    best_integration = max(int_avgs, key=lambda x: x[1])
    best_cf = max(counterfactual_results, key=lambda x: x['metrics']['error_correction'])

    print(f"\n🥇 Best Noise Robustness: {best_noise[0]} ({best_noise[1]:.1f}%)")
    print(f"🥇 Best Negative Rejection: {best_rejection['model']} ({best_rejection['metrics']['rejection_rate']:.1f}%)")
    print(f"🥇 Best Info Integration: {best_integration[0]} ({best_integration[1]:.1f}%)")
    print(f"🥇 Best Error Correction: {best_cf['model']} ({best_cf['metrics']['error_correction']:.1f}%)")

    print("\n" + "=" * 80)


def generate_html_report(results: list, filepath: Path) -> Path:
    """Generate HTML report with visualizations."""

    # Group results
    noise_results = [r for r in results if r["ability"] == "noise_robustness"]
    rejection_results = [r for r in results if r["ability"] == "negative_rejection"]
    integration_results = [r for r in results if r["ability"] == "information_integration"]
    counterfactual_results = [r for r in results if r["ability"] == "counterfactual_robustness"]

    # Calculate averages
    model_noise_avgs = {}
    model_int_avgs = {}
    for model in LLM_MODELS:
        noise_model = [r for r in noise_results if r["model"] == model]
        int_model = [r for r in integration_results if r["model"] == model]
        if noise_model:
            model_noise_avgs[model] = sum(r['metrics']['accuracy'] for r in noise_model) / len(noise_model)
        if int_model:
            model_int_avgs[model] = sum(r['metrics']['accuracy'] for r in int_model) / len(int_model)

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RGB Benchmark Evaluation Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .card h2 {{
            font-size: 1.2rem;
            margin-bottom: 15px;
            color: #667eea;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        th {{
            color: #667eea;
            font-weight: 600;
        }}
        .chart-container {{
            height: 300px;
            margin-top: 15px;
        }}
        .highlight {{
            color: #4ade80;
            font-weight: bold;
        }}
        .low {{
            color: #f87171;
        }}
        .findings {{
            background: rgba(102, 126, 234, 0.1);
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-top: 20px;
            border-radius: 0 10px 10px 0;
        }}
        .findings h3 {{
            margin-bottom: 15px;
        }}
        .findings ul {{
            list-style: none;
            padding: 0;
        }}
        .findings li {{
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .findings li:last-child {{
            border-bottom: none;
        }}
        .medal {{
            margin-right: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 RGB Benchmark Evaluation Report</h1>
        <p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Source: {filepath.name}</p>

        <div class="grid">
            <!-- Noise Robustness Table -->
            <div class="card">
                <h2>📈 Noise Robustness (Accuracy %)</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        {''.join(f'<th>{r}</th>' for r in NOISE_RATIOS)}
                        <th>Avg</th>
                    </tr>
                    {''.join(f"""<tr>
                        <td>{model.split('/')[-1]}</td>
                        {''.join(f'<td class="{"highlight" if next((x for x in noise_results if x["model"] == model and x["metrics"].get("noise_ratio") == nr), {{}}).get("metrics", {{}}).get("accuracy", 0) >= 90 else "low" if next((x for x in noise_results if x["model"] == model and x["metrics"].get("noise_ratio") == nr), {{}}).get("metrics", {{}}).get("accuracy", 0) < 50 else ""}">{next((x for x in noise_results if x["model"] == model and x["metrics"].get("noise_ratio") == nr), {{}}).get("metrics", {{}}).get("accuracy", "N/A"):.1f if isinstance(next((x for x in noise_results if x["model"]==model and x["metrics"].get("noise_ratio")==nr), { {} }).get("metrics", { {} }).get("accuracy", "N/A"), (int, float)) else "N/A"}</td>' for nr in NOISE_RATIOS)}
                        <td class="highlight">{model_noise_avgs.get(model, 0):.1f}</td>
                    </tr>""" for model in LLM_MODELS)}
                </table>
            </div>

            <!-- Negative Rejection Table -->
            <div class="card">
                <h2>🚫 Negative Rejection (Rejection Rate %)</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Rejection Rate</th>
                    </tr>
                    {''.join(f"""<tr>
                        <td>{r['model'].split('/')[-1]}</td>
                        <td class="{"highlight" if r['metrics']['rejection_rate'] >= 70 else "low" if r['metrics']['rejection_rate'] < 50 else ""}">{r['metrics']['rejection_rate']:.1f}%</td>
                    </tr>""" for r in rejection_results)}
                </table>
            </div>

            <!-- Information Integration Table -->
            <div class="card">
                <h2>🔗 Information Integration (Accuracy %)</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        {''.join(f'<th>{r}</th>' for r in INTEGRATION_NOISE_RATIOS)}
                        <th>Avg</th>
                    </tr>
                    {''.join(f"""<tr>
                        <td>{model.split('/')[-1]}</td>
                        {''.join(f'<td>{next((x for x in integration_results if x["model"] == model and x["metrics"].get("noise_ratio") == nr), {{}}).get("metrics", {{}}).get("accuracy", "N/A"):.1f if isinstance(next((x for x in integration_results if x["model"]==model and x["metrics"].get("noise_ratio")==nr), { {} }).get("metrics", { {} }).get("accuracy", "N/A"), (int, float)) else "N/A"}</td>' for nr in INTEGRATION_NOISE_RATIOS)}
                        <td class="highlight">{model_int_avgs.get(model, 0):.1f}</td>
                    </tr>""" for model in LLM_MODELS)}
                </table>
            </div>

            <!-- Counterfactual Robustness Table -->
            <div class="card">
                <h2>⚠️ Counterfactual Robustness (%)</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Acc</th>
                        <th>Acc_doc</th>
                        <th>ED</th>
                        <th>CR</th>
                    </tr>
                    {''.join(f"""<tr>
                        <td>{r['model'].split('/')[-1]}</td>
                        <td>{r['metrics']['acc']:.1f}</td>
                        <td>{r['metrics']['acc_doc']:.1f}</td>
                        <td>{r['metrics']['error_detection']:.1f}</td>
                        <td class="{"highlight" if r['metrics']['error_correction'] >= 70 else ""}">{r['metrics']['error_correction']:.1f}</td>
                    </tr>""" for r in counterfactual_results)}
                </table>
            </div>
        </div>

        <!-- Charts -->
        <div class="grid">
            <div class="card">
                <h2>📊 Noise Robustness Comparison</h2>
                <div class="chart-container">
                    <canvas id="noiseChart"></canvas>
                </div>
            </div>
            <div class="card">
                <h2>📊 Overall Performance Summary</h2>
                <div class="chart-container">
                    <canvas id="summaryChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Key Findings -->
        <div class="findings">
            <h3>🏆 Key Findings</h3>
            <ul>
                <li><span class="medal">🥇</span><strong>Best Noise Robustness:</strong> {max(model_noise_avgs.items(), key=lambda x: x[1])[0]} ({max(model_noise_avgs.values()):.1f}%)</li>
                <li><span class="medal">🥇</span><strong>Best Negative Rejection:</strong> {max(rejection_results, key=lambda x: x['metrics']['rejection_rate'])['model']} ({max(r['metrics']['rejection_rate'] for r in rejection_results):.1f}%)</li>
                <li><span class="medal">🥇</span><strong>Best Info Integration:</strong> {max(model_int_avgs.items(), key=lambda x: x[1])[0]} ({max(model_int_avgs.values()):.1f}%)</li>
                <li><span class="medal">🥇</span><strong>Best Error Correction:</strong> {max(counterfactual_results, key=lambda x: x['metrics']['error_correction'])['model']} ({max(r['metrics']['error_correction'] for r in counterfactual_results):.1f}%)</li>
            </ul>
        </div>
    </div>

    <script>
        // Noise Robustness Chart
        const noiseCtx = document.getElementById('noiseChart').getContext('2d');
        new Chart(noiseCtx, {{
            type: 'line',
            data: {{
                labels: {NOISE_RATIOS},
                datasets: [
                    {','.join(f"""{{
                        label: '{model.split('/')[-1]}',
                        data: [{','.join(str(next((r['metrics']['accuracy'] for r in noise_results if r['model'] == model and r['metrics'].get('noise_ratio') == nr), 0)) for nr in NOISE_RATIOS)}],
                        borderColor: '{"#667eea" if i == 0 else "#4ade80" if i == 1 else "#f59e0b"}',
                        backgroundColor: '{"rgba(102,126,234,0.1)" if i == 0 else "rgba(74,222,128,0.1)" if i == 1 else "rgba(245,158,11,0.1)"}',
                        tension: 0.3,
                        fill: true
                    }}""" for i, model in enumerate(LLM_MODELS))}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#888' }}
                    }},
                    x: {{
                        title: {{ display: true, text: 'Noise Ratio', color: '#888' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#888' }}
                    }}
                }},
                plugins: {{
                    legend: {{ labels: {{ color: '#eee' }} }}
                }}
            }}
        }});

        // Summary Bar Chart
        const summaryCtx = document.getElementById('summaryChart').getContext('2d');
        new Chart(summaryCtx, {{
            type: 'bar',
            data: {{
                labels: ['Noise Robust.', 'Rejection', 'Integration', 'Error Corr.'],
                datasets: [
                    {','.join(f"""{{
                        label: '{model.split('/')[-1]}',
                        data: [
                            {model_noise_avgs.get(model, 0):.1f},
                            {next((r['metrics']['rejection_rate'] for r in rejection_results if r['model'] == model), 0):.1f},
                            {model_int_avgs.get(model, 0):.1f},
                            {next((r['metrics']['error_correction'] for r in counterfactual_results if r['model'] == model), 0):.1f}
                        ],
                        backgroundColor: '{"rgba(102,126,234,0.7)" if i == 0 else "rgba(74,222,128,0.7)" if i == 1 else "rgba(245,158,11,0.7)"}'
                    }}""" for i, model in enumerate(LLM_MODELS))}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#888' }}
                    }},
                    x: {{
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#888' }}
                    }}
                }},
                plugins: {{
                    legend: {{ labels: {{ color: '#eee' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>'''

    output_path = RESULTS_DIR / f"rgb_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(output_path, 'w') as f:
        f.write(html_content)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="RGB Results Report Generator")
    parser.add_argument("--file", type=str, help="Path to results JSON file")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")

    args = parser.parse_args()

    # Find results file
    if args.file:
        filepath = Path(args.file)
    else:
        filepath = find_latest_results()

    print(f"📂 Loading results from: {filepath}")
    results = load_results(filepath)

    # Print console report
    print_console_report(results, filepath)

    # Generate HTML report if requested
    if args.html:
        html_path = generate_html_report(results, filepath)
        print(f"\n📄 HTML report saved to: {html_path}")


if __name__ == "__main__":
    main()