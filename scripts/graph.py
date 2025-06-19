#* Imports
import os
import re
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import shutil
import random

#* ANSI color codes for console feedback
CYAN = "\033[96m"
RESET = "\033[0m"

#* CONFIGURATION 
METRICS_FILES = [
    "metrics/transcripts_tiny_copy_healed_metrics.xml",
    "metrics/transcripts_small_copy_healed_metrics.xml",
    "metrics/transcripts_medium_copy_healed_metrics.xml",
    "metrics/transcripts_large_copy_healed_metrics.xml",
    "metrics/transcripts_turbo_copy_healed_metrics.xml",
]
GRAPHS_DIR = "graphs"
STATS_XML = os.path.join(GRAPHS_DIR, 'stats.xml')

#* Parse metrics XML and return dict mapping block_id to lists of (wer, cer, lev).
def parse_metrics_detailed(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = {}
    for block in root.findall('block'):
        bid = block.attrib.get('id')
        wer_vals, cer_vals, lev_vals = [], [], []
        for file_elem in block.findall('file'):
            try:
                wer_vals.append(float(file_elem.findtext('wer') or 0))
                cer_vals.append(float(file_elem.findtext('cer') or 0))
                lev_vals.append(float(file_elem.findtext('levenshtein') or 0))
            except ValueError:
                continue
        data[bid] = (wer_vals, cer_vals, lev_vals)
    return data

#* Extract Whisper model name (e.g., tiny, small) from filename.
def extract_model_name(path):

    base = os.path.basename(path)
    m = re.search(r"transcripts_([a-zA-Z0-9]+)_", base)
    return m.group(1) if m else base

#* Call plotting functions for a given metric across models.
def plot_metric_funcs(metric_name, raw_values_list, model_names, colors, out_dir):
    #* mean bar
    means = [np.mean(vals) for vals in raw_values_list]
    plot_meanbar(metric_name, means, model_names, colors, out_dir)
    #* errorbars
    stds = [np.std(vals) for vals in raw_values_list]
    plot_errorbar(metric_name, means, stds, model_names, colors, out_dir)
    #* boxplot
    plot_boxplot(metric_name, raw_values_list, model_names, colors, out_dir)
    #* scatter
    plot_scatter(metric_name, raw_values_list, model_names, colors, out_dir)


def plot_meanbar(metric_name, values, names, colors, out_dir):
    items = list(zip(values, names, colors))
    items.sort(key=lambda x: x[0], reverse=True)
    vals, labs, cols = zip(*items)
    plt.figure(); x = np.arange(len(vals))
    plt.bar(x, vals, color=cols); plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(x, labs, rotation=45, ha='right'); plt.xlabel('OpenAI Whisper Model')
    plt.ylabel(metric_name); plt.title(f'{metric_name} Mean by Model'); plt.tight_layout()
    path = os.path.join(out_dir, f'{metric_name.lower()}_mean.png'); plt.savefig(path); plt.close()
    print(f"{CYAN}{metric_name} mean chart saved to {path}{RESET}")


def plot_errorbar(metric_name, means, stds, names, colors, out_dir):
    items = list(zip(means, stds, names, colors))
    items.sort(key=lambda x: x[0], reverse=True)
    m, s, labs, cols = zip(*items)
    plt.figure(); x = np.arange(len(m))
    plt.bar(x, m, yerr=s, capsize=5, color=cols); plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(x, labs, rotation=45, ha='right'); plt.xlabel('OpenAI Whisper Model')
    plt.ylabel(metric_name); plt.title(f'{metric_name} Mean±StdDev'); plt.tight_layout()
    path = os.path.join(out_dir, f'{metric_name.lower()}_errorbar.png'); plt.savefig(path); plt.close()
    print(f"{CYAN}{metric_name} errorbar chart saved to {path}{RESET}")


def plot_boxplot(metric_name, raw_lists, names, colors, out_dir):
    items = list(zip(raw_lists, names, colors))
    items.sort(key=lambda x: np.mean(x[0]), reverse=True)
    lists, labs, cols = zip(*items)
    plt.figure(); bp = plt.boxplot(lists, labels=labs, patch_artist=True)
    for patch, col in zip(bp['boxes'], cols): patch.set_facecolor(col)
    plt.xticks(rotation=45, ha='right'); plt.ylabel(metric_name)
    plt.title(f'{metric_name} Distribution'); plt.tight_layout()
    path = os.path.join(out_dir, f'{metric_name.lower()}_boxplot.png'); plt.savefig(path); plt.close()
    print(f"{CYAN}{metric_name} boxplot saved to {path}{RESET}")


def plot_scatter(metric_name, raw_lists, names, colors, out_dir):
    items = list(zip(raw_lists, names, colors))
    items.sort(key=lambda x: np.mean(x[0]), reverse=True)
    lists, labs, cols = zip(*items)
    plt.figure()
    for i, (vals, _, col) in enumerate(items):
        x = np.ones(len(vals)) * i + (np.random.rand(len(vals)) - 0.5) * 0.1
        plt.scatter(x, vals, s=10, color=col, alpha=0.7)
    plt.xticks(range(len(labs)), labs, rotation=45, ha='right'); plt.ylabel(metric_name)
    plt.title(f'{metric_name} Values'); plt.tight_layout()
    path = os.path.join(out_dir, f'{metric_name.lower()}_scatter.png'); plt.savefig(path); plt.close()
    print(f"{CYAN}{metric_name} scatter saved to {path}{RESET}")

#* Write comprehensive stats to XML: all_stats is dict model->dict block->(mean,std) and model->overall.
def write_stats_xml(all_stats, out_path):

    root = ET.Element('stats')
    for model, blocks in all_stats.items():
        m_el = ET.SubElement(root, 'model', name=model)
        for blk, stats in blocks['by_block'].items():
            b_el = ET.SubElement(m_el, 'block', id=blk)
            for metric, (mean, std) in stats.items():
                met_el = ET.SubElement(b_el, metric.lower())
                met_el.set('mean', f"{mean:.3f}")
                met_el.set('std', f"{std:.3f}")
        o_el = ET.SubElement(m_el, 'overall')
        for metric, (mean, std) in blocks['overall'].items():
            met_el = ET.SubElement(o_el, metric.lower())
            met_el.set('mean', f"{mean:.3f}")
            met_el.set('std', f"{std:.3f}")
    tree = ET.ElementTree(root)
    tree.write(out_path, encoding='utf-8', xml_declaration=True)
    print(f"{CYAN}Stats XML written to {out_path}{RESET}")


def main():
    if os.path.exists(GRAPHS_DIR): shutil.rmtree(GRAPHS_DIR)
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    model_names = [extract_model_name(p) for p in METRICS_FILES]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    all_stats = {}
    raw_data = {'WER': [], 'CER': [], 'Levenshtein': []}

    for model, path in zip(model_names, METRICS_FILES):
        data = parse_metrics_detailed(path)
        by_block = {}
        agg = {'WER': [], 'CER': [], 'Levenshtein': []}
        for blk, (w_vals, c_vals, l_vals) in data.items():
            stats = {}
            for metric, vals in [('WER', w_vals), ('CER', c_vals), ('Levenshtein', l_vals)]:
                mean = np.mean(vals) if vals else np.nan
                std = np.std(vals) if vals else np.nan
                stats[metric] = (mean, std)
                agg[metric].extend(vals)
            by_block[blk] = stats
        overall = {metric: (np.mean(agg[metric]) if agg[metric] else np.nan,
                             np.std(agg[metric]) if agg[metric] else np.nan)
                   for metric in agg}
        all_stats[model] = {'by_block': by_block, 'overall': overall}
        raw_data['WER'].append(agg['WER'])
        raw_data['CER'].append(agg['CER'])
        raw_data['Levenshtein'].append(agg['Levenshtein'])

    for metric in ['WER', 'CER', 'Levenshtein']:
        plot_metric_funcs(metric, raw_data[metric], model_names, colors, GRAPHS_DIR)

    write_stats_xml(all_stats, STATS_XML)

if __name__ == '__main__':
    main()
