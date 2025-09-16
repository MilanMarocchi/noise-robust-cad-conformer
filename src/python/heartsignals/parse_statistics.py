"""
    For parsing log files to extract the main stats
"""
from collections import defaultdict
import os
import click
import logging
import json
import re
import numpy as np
from collections import defaultdict

@click.group(context_settings={'show_default': True})
@click.option('--LOG_LEVEL', type=click.Choice(['INFO', 'DEBUG', 'FINE']), default='INFO', help='Debug flag level')
@click.pass_context
def cli(ctx, **kwargs):

    logging.basicConfig(level=getattr(logging, kwargs['log_level'], None))

@cli.command()
@click.option('-I', '--input_file', required=True, help='Path to the input log file.')
@click.option('-O', '--output_file', default='statistics.json', help='Path to the output file.')
def parse_logfile(input_file, output_file):
    log_file_path = os.path.abspath(input_file)
    output_json_path = os.path.abspath(output_file)

    # Regular expressions to identify trials and the specific stats sections
    trial_pattern = re.compile(r'Training Trial (\d+)')
    fragment_stats_pattern = re.compile(r'Fragment Stats')
    patient_stats_pattern = re.compile(r'Patient Stats')

    # Initialize list to store parsed final test stats for each trial
    results = []
    current_trial = None
    capture_stats = False

    # Read and parse the log file
    with open(log_file_path, 'r') as file:
        for line in file:
            # Identify a new trial start
            trial_match = trial_pattern.match(line)
            if trial_match:
                # Save the last trial's stats if it exists
                if current_trial:
                    results.append(current_trial)

                # Start a new trial stats capture
                current_trial = {"trial": int(trial_match.group(1)), "Fragment Stats": {}, "Patient Stats": {}}
                capture_stats = False
                continue

            # Detect start of Fragment Stats
            if fragment_stats_pattern.search(line):
                capture_stats = "Fragment Stats"
                continue

            # Detect start of Patient Stats
            if patient_stats_pattern.search(line):
                capture_stats = "Patient Stats"
                continue

            # Capture final test stats values only for Fragment and Patient Stats sections
            if capture_stats and ':' in line:
                stats = current_trial[capture_stats]
                for metric in line.split(', '):
                    if ': ' in metric:
                        key, value = metric.split(': ')
                        stats[key.strip()] = float(value.strip()) if value.strip() != 'nan' else None
                capture_stats = False  # Capture only the last set of stats

        # Add the last trial's stats, if present
        if current_trial:
            results.append(current_trial)

    # Write the parsed final test stats to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print("Parsed final test stats saved to:", output_json_path)

def parse_json(input_file):
    json_file_path = os.path.abspath(input_file)

    # Load the data from JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Initialize dictionaries to hold metrics for Fragment Stats and Patient Stats
    fragment_metrics = {}
    patient_metrics = {}

    # Aggregate metrics for each trial
    for trial in data:
        # Process Fragment Stats
        for key, value in trial['Fragment Stats'].items():
            fragment_metrics.setdefault(key, []).append(value)
        
        # Process Patient Stats
        for key, value in trial['Patient Stats'].items():
            patient_metrics.setdefault(key, []).append(value)

    return fragment_metrics, patient_metrics

@cli.command()
@click.option('-I', '--input_files', required=True, help='Path to the input json file, separated by ":" [path/to/stat.json:path/to/stat2.json].')
def parse_json_stats(input_files):
    input_files = input_files.split(":")

    fragment_metrics = defaultdict(list)
    patient_metrics = defaultdict(list)

    for input_file in input_files:
        fragment_metric, patient_metric = parse_json(input_file)
        fragment_dicts = [fragment_metrics, fragment_metric]
        patient_dicts = [patient_metrics , patient_metric]

        for d in fragment_dicts:
            for key, value in d.items():
                fragment_metrics[key].extend(value)
        for d in patient_dicts:
            for key, value in d.items():
                patient_metrics[key].extend(value)

    # Function to calculate mean and standard deviation and print results
    def print_stats(metrics_dict, title):
        print(f"\n{title}")
        for metric, values in metrics_dict.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric}: Mean = {mean_val:.4f}, Std Dev = {std_val:.4f}")

    # Print the results for Fragment Stats and Patient Stats
    print_stats(fragment_metrics, "Fragment Stats")
    print_stats(patient_metrics, "Patient Stats")


@cli.command()
@click.option('-I', '--input_file', required=True, help='Path to the input log file')
def parse_vest_log(input_file):
    model_pattern = re.compile(r'Stats from (.*?)\n')
    stats_pattern = re.compile(r'best_models_test_(fragment|subject)_stats(?:_svm)?=({.*?})')
    
    trials = defaultdict(lambda: {"fragment": [], "subject": []})
    current_model = None
    file_path = os.path.abspath(input_file)
    
    with open(file_path, 'r') as file:
        for line in file:
            model_match = model_pattern.match(line)
            if model_match:
                current_model = model_match.group(1)
            
            stats_match = stats_pattern.search(line)
            if stats_match and current_model:
                stat_type = stats_match.group(1)
                stats_str = stats_match.group(2).replace("'", '"').replace('nan', 'null')  # Fix JSON format
                try:
                    stats_dict = json.loads(stats_str)  # Parse with JSON
                    trials[current_model][stat_type].append(stats_dict)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON for {current_model} {stat_type}: {e}")
                    print(f"Raw string: {stats_str}")
    
    # Compute averages
    def compute_averages(stat_list):
        stat_sums = defaultdict(float)
        stat_counts = defaultdict(int)
        for stats in stat_list:
            for key, value in stats.items():
                if isinstance(value, (int, float)) and value is not None:
                    stat_sums[key] += value
                    stat_counts[key] += 1
        return {key: stat_sums[key] / stat_counts[key] for key in stat_sums} if stat_counts else {}
    
    model_averages = {model: {"fragment": compute_averages(data["fragment"]), "subject": compute_averages(data["subject"])} 
                      for model, data in trials.items()}
    
    # Print original stats
    for model, trial in trials.items():
        print(f"Model: {model}")
        print("Fragment Stats:")
        for stats in trial["fragment"]:
            print(stats)
        print("Subject Stats:")
        for stats in trial["subject"]:
            print(stats)
        print()
    
    # Print averages
    for model, averages in model_averages.items():
        print(f"Averaged Stats for {model}:")
        print("Fragment Stats:")
        for key, value in averages["fragment"].items():
            print(f"{key}: {value:.3f}")
        print("Subject Stats:")
        for key, value in averages["subject"].items():
            print(f"{key}: {value:.3f}")
        print()

if __name__ == '__main__':
    cli()