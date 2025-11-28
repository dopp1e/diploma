import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mtick
from matplotlib.ticker import AutoMinorLocator
from matplotlib import patheffects as path_effects

LIKERT_SCALE = [1, 2, 3, 4, 5]
LIKERT_LABELS = {
    1: "Strongly disagree",
    2: "Disagree",
    3: "Neutral",
    4: "Agree",
    5: "Strongly agree",
}
LIKERT_COLORS = ['#b30000', '#e34a33', '#f4a259', '#5ab769', '#1a7431']

def ordered_classes(unique_classes):
    order = [2, 0, 1]
    ordered = [unique_classes[i] for i in order if i < len(unique_classes)]
    if ordered:
        return ordered
    return unique_classes.tolist() if hasattr(unique_classes, "tolist") else list(unique_classes)

def likert_distribution(series):
    counts = series.value_counts(dropna=True).reindex(LIKERT_SCALE, fill_value=0)
    total = counts.sum()
    if total == 0:
        zeros = np.zeros(len(LIKERT_SCALE))
        return zeros, zeros
    shares = (counts / total).values
    return shares, counts.values

def plot_likert_stack(ax, position, shares, counts, width, plotted_categories, label_threshold=0.08):
    bottom = 0.0
    for idx, share in enumerate(shares):
        count = counts[idx]
        if share <= 0 and count <= 0:
            continue
        category_label = LIKERT_LABELS[LIKERT_SCALE[idx]]
        label = category_label if category_label not in plotted_categories else None
        ax.bar(position, share, width=width, bottom=bottom, color=LIKERT_COLORS[idx], label=label)
        if label:
            plotted_categories.add(category_label)
        if share > 0:
            text = f"{share * 100:.0f}%\nn={int(count)}"
            if share >= label_threshold:
                text_y, va = bottom + share / 2, 'center'
            else:
                text_y, va = min(bottom + share + 0.01, 1.02), 'bottom'
            text_obj = ax.text(position, text_y, text, ha='center', va=va, fontsize=10, color='white', fontweight='bold')
            text_obj.set_path_effects([
                path_effects.Stroke(linewidth=1.2, foreground='black'),
                path_effects.Normal(),
            ])
        bottom += share

def configure_likert_axes(ax):
    ax.set_ylabel('Response share (%)')
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.minorticks_on()
    ax.grid(axis='y', alpha=0.35, which='major')
    ax.grid(axis='y', alpha=0.15, which='minor')

def add_likert_legend(ax):
    # ensure the items in the legend appear in the REVERSE order of LIKERT_SCALE
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered_handles = [label_to_handle[LIKERT_LABELS[scale]] for scale in reversed(LIKERT_SCALE) if LIKERT_LABELS[scale] in label_to_handle]
    ordered_labels = [LIKERT_LABELS[scale] for scale in reversed(LIKERT_SCALE) if LIKERT_LABELS[scale] in label_to_handle]

    # ax.legend(ordered_handles, ordered_labels, title="Response options", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    return ax.legend(
        ordered_handles, ordered_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.18),
        ncol=len(LIKERT_SCALE),
        title='Response options',
        frameon=False,
    )

def load_data(file_path):
    """Load data from a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path, sep='\t', header=0)

def base_plot(data):
    """Plot means of columns D-O vs P-AA with error bars showing standard deviation,
    all pairs plotted on a single axes with the two columns in each pair grouped closely."""
    columns1 = data.columns[3:15]  # D to O
    columns2 = data.columns[15:27]  # P to AA

    pairs = list(zip(columns1, columns2))
    n = len(pairs)
    if n == 0:
        return

    # Precompute means/stds and global y-limits
    stats = []
    max_val = -np.inf
    min_val = np.inf
    for col1, col2 in pairs:
        m1, s1 = data[col1].mean(), data[col1].std()
        m2, s2 = data[col2].mean(), data[col2].std()
        print(f"[BasePlot] Mean {col1} (Our VR Simulator): {m1:.4f}")
        print(f"[BasePlot] Mean {col2} (PC Building Simulator): {m2:.4f}")
        stats.append((col1, col2, m1, s1, m2, s2))
        max_val = max(max_val, m1 + s1, m2 + s2)
        min_val = min(min_val, m1 - s1, m2 - s2)

    if np.isfinite(max_val):
        y_max = max_val * 1.1
    else:
        y_max = None
    if np.isfinite(min_val):
        y_min = min_val * 1.1
    else:
        y_min = 0.0

    # Single axes: grouped bars (two bars per group, pairs spaced apart)
    fig_width = max(8, n * 1.2)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 6))

    group_positions = np.arange(n)  # center positions for each pair
    bar_width = 0.35  # narrow bars so the pair appears close
    offsets = [-bar_width / 2, bar_width / 2]

    for i, (col1, col2, m1, s1, m2, s2) in enumerate(stats):
        pos = group_positions[i]
        # left bar (first column in pair)
        ax.bar(pos + offsets[0], m1, width=bar_width, yerr=s1, capsize=5, color='C0',
               label='Our VR Simulator' if i == 0 else None)
        # right bar (second column in pair)
        ax.bar(pos + offsets[1], m2, width=bar_width, yerr=s2, capsize=5, color='C1',
               label='PC Building Simulator' if i == 0 else None)

    # Use stacked tick labels to show both column names for each group
    ax.set_xticks(group_positions)
    ax.set_xticks(group_positions)
    ax.set_xticklabels([f'{c1.split(" ")[1]}' for c1, _ in pairs], rotation=0, ha='center')
    ax.set_ylabel('Mean value')
    ax.set_xlabel("Question answers, grouped by application the answer pertains to")
    if y_max is not None:
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(bottom=y_min)

    ticks = ax.get_yticks()
    ticklabels = ["" if np.isclose(t, 0.0) else format(t, ".2g") for t in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)

    # add minor ticks
    ax.minorticks_on()
    # add horizontal grid lines
    ax.grid(axis = 'y', alpha = 0.6, which='major')
    ax.grid(axis = 'y', alpha = 0.3, which='minor')

    ax.legend()
    plt.tight_layout()
    fig.savefig('comparison_all_pairs_mean_grouped.png')
    plt.close(fig)

def base_plot_stacked(data):
    """Render 100% stacked bars for each question pair by application."""
    columns1 = data.columns[3:15]
    columns2 = data.columns[15:27]
    pairs = list(zip(columns1, columns2))
    if not pairs:
        return
    fig_width = max(8, len(pairs) * 1.4)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 6))
    offsets = (-0.2, 0.2)
    bar_width = 0.35
    group_positions = np.arange(len(pairs))
    xticks, xticklabels = [], []
    plotted_categories = set()
    for i, (col1, col2) in enumerate(pairs):
        for offset, col, label in (
            (offsets[0], col1, 'Our VR Simulator'),
            (offsets[1], col2, 'PC Building Simulator'),
        ):
            pos = group_positions[i] + offset
            shares, counts = likert_distribution(data[col].dropna())
            plot_likert_stack(ax, pos, shares, counts, bar_width, plotted_categories)
            short = col.split(" ")[1] if " " in col else col
            xticks.append(pos)
            xticklabels.append(f'{short}\n{label}')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    ax.set_xlabel("Question answers, grouped by which application the answer pertains to")
    configure_likert_axes(ax)
    add_likert_legend(ax)
    fig.tight_layout(rect=[0, 0, 1, 1.0])
    fig.savefig('comparison_all_pairs_likert_grouped.png')
    plt.close(fig)

def class_plot(data):
    """Plot means of columns D-O and P-AA combined, grouped by the 'Class' column."""
    columns1 = data.columns[3:15]  # D to O
    columns2 = data.columns[15:27]  # P to AA

    pairs = list(zip(columns1, columns2))
    n = len(pairs)
    if n == 0:
        return

    stats = []
    max_val = -np.inf
    min_val = np.inf
    classes = ordered_classes(data['Class'].unique())
    if not classes:
        return
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    class_colors = {cls: palette[i % len(palette)] for i, cls in enumerate(classes)}

    for col1, col2 in pairs:
        class_stats = {}
        for cls in classes:
            group = data[data['Class'] == cls]
            combined = pd.concat([group[col1], group[col2]]).dropna()
            m, s = combined.mean(), combined.std()
            print(f"[ClassPlot] Combined mean {col1}/{col2} ({cls}): {m:.4f}")
            class_stats[cls] = (m, s)
            max_val = max(max_val, m + s)
            min_val = min(min_val, m - s)
        stats.append((col1, col2, class_stats))

    if np.isfinite(max_val):
        y_max = max_val * 1.1
    else:
        y_max = None
    if np.isfinite(min_val):
        y_min = min_val * 1.1
    else:
        y_min = 0.0

    fig_width = max(8, n * 1.2)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 6))

    group_positions = np.arange(n)
    bar_width = 0.6
    offsets = np.linspace(-bar_width / 2, bar_width / 2, len(classes))
    plotted_labels = set()

    for i, (col1, col2, class_stats) in enumerate(stats):
        pos = group_positions[i]
        for j, cls in enumerate(classes):
            m, s = class_stats[cls]
            label = cls if cls not in plotted_labels else None
            ax.bar(
                pos + offsets[j],
                m,
                width=bar_width / len(classes),
                yerr=s,
                capsize=5,
                color=class_colors[cls],
                label=label,
            )
            if label:
                plotted_labels.add(cls)

    ax.set_xticks(group_positions)
    ax.set_xticklabels([f'{c1.split(" ")[1]}' for c1, _ in pairs], rotation=0, ha='center')
    ax.set_ylabel('Mean value (combined)')
    ax.set_xlabel("Question answers, grouped by application the answer pertains to")
    if y_max is not None:
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(bottom=y_min)

    ticks = ax.get_yticks()
    ticklabels = ["" if np.isclose(t, 0.0) else format(t, ".2g") for t in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)

    ax.minorticks_on()
    ax.grid(axis='y', alpha=0.6, which='major')
    ax.grid(axis='y', alpha=0.3, which='minor')

    # handles, labels = ax.get_legend_handles_labels()
    # order = [2, 0, 1]
    # ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    ax.legend()
    plt.tight_layout()
    fig.savefig('comparison_all_pairs_mean_grouped_by_class_combined.png')
    plt.close(fig)

def class_plot_stacked(data):
    """Render 100% stacked bars for each question pair combined per class."""
    if 'Class' not in data.columns:
        return
    columns1 = data.columns[3:15]
    columns2 = data.columns[15:27]
    pairs = list(zip(columns1, columns2))
    if not pairs:
        return
    classes = ordered_classes(data['Class'].unique())
    if not classes:
        return
    fig_width = max(8, len(pairs) * 1.6)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 6))
    group_positions = np.arange(len(pairs))
    bar_group_width = 0.6
    offsets = np.linspace(-bar_group_width / 2, bar_group_width / 2, len(classes))
    bar_width = bar_group_width / len(classes)
    xticks, xticklabels = [], []
    plotted_categories = set()
    for i, (col1, col2) in enumerate(pairs):
        for j, cls in enumerate(classes):
            group = data[data['Class'] == cls]
            combined = pd.concat([group[col1], group[col2]]).dropna()
            shares, counts = likert_distribution(combined)
            pos = group_positions[i] + offsets[j]
            plot_likert_stack(ax, pos, shares, counts, bar_width, plotted_categories)
            short = col1.split(" ")[1] if " " in col1 else col1
            xticks.append(pos)
            xticklabels.append(f'{short}\n{cls}')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    ax.set_xlabel("Question answers, grouped by class")
    configure_likert_axes(ax)
    add_likert_legend(ax)
    fig.tight_layout(rect=[0, 0, 1, 1.0])
    fig.savefig('comparison_all_pairs_likert_grouped_by_class.png')
    plt.close(fig)

def delta_plot(data):
    """Plot difference between means D-O vs P-AA, grouped by whether the value in the 'First' column is 'PCBVR' or 'PCBS',
    all pairs plotted on a single axes with the two columns in each pair grouped closely."""
    columns1 = data.columns[3:15]  # D to O
    columns2 = data.columns[15:27]  # P to AA

    pairs = list(zip(columns1, columns2))
    n = len(pairs)
    if n == 0:
        return

    # Precompute means/stds and global y-limits
    stats = []
    max_val = -np.inf
    min_val = np.inf
    for col1, col2 in pairs:
        group1 = data[data['First'] == 'PCBVR']
        group2 = data[data['First'] == 'PCBS']
        m1 = group1[col2].mean() - group1[col1].mean()
        s1 = np.sqrt(group1[col2].std()**2 + group1[col1].std()**2)
        m2 = group2[col1].mean() - group2[col2].mean()
        s2 = np.sqrt(group2[col1].std()**2 + group2[col2].std()**2)
        print(f"[DeltaPlot] Mean delta {col1}->{col2} (First=PCBVR): {m1:.4f}")
        print(f"[DeltaPlot] Mean delta {col2}->{col1} (First=PCBS): {m2:.4f}")
        stats.append((col1, col2, m1, s1, m2, s2))
        max_val = max(max_val, m1 + s1, m2 + s2)
        min_val = min(min_val, m1 - s1, m2 - s2)

    if np.isfinite(max_val):
        y_max = max_val * 1.1
    else:
        y_max = None
    if np.isfinite(min_val):
        y_min = min_val * 1.1
    else:
        y_min = None

    # Single axes: grouped bars (two bars per group, pairs spaced apart)
    fig_width = max(8, n * 1.2)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 6))

    group_positions = np.arange(n)  # center positions for each pair
    bar_width = 0.35  # narrow bars so the pair appears close
    offsets = [-bar_width / 2, bar_width / 2]

    for i, (col1, col2, m1, s1, m2, s2) in enumerate(stats):
        pos = group_positions[i]
        # left bar (PCBVR group)
        ax.bar(pos + offsets[0], m1, width=bar_width, yerr=s1, capsize=5, color='C0',
               label='Our VR Simulator' if i == 0 else None)
        # right bar (PCBVS group)
        ax.bar(pos + offsets[1], m2, width=bar_width, yerr
                =s2, capsize=5, color='C1',
                label='PC Building Simulator' if i == 0 else None)
    
    # Use stacked tick labels to show both column names for each group
    ax.set_xticks(group_positions)
    ax.set_xticks(group_positions)
    ax.set_xticklabels([f'{c1.split(" ")[1]}' for c1, _ in pairs], rotation=0, ha='center')
    ax.set_ylabel('Mean difference between answers')
    ax.set_xlabel("Question answers, grouped by which application was used first")
    if y_max is not None:
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(bottom=y_min)

    # add minor ticks
    ax.minorticks_on()
    # add horizontal grid lines
    ax.grid(axis = 'y', alpha = 0.6, which='major')
    ax.grid(axis = 'y', alpha = 0.3, which='minor')
    
    ticks = ax.get_yticks()
    ticklabels = ["" if np.isclose(t, 0.0) else format(t, ".2g") for t in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)

    ax.legend()
    plt.tight_layout()
    fig.savefig('comparison_all_pairs_mean_grouped_delta.png')
    plt.close(fig)

def final_plot(data):
    """Like class_final_plot but without class grouping."""
    columns = data.columns[27:30]  # AB to AD
    n = len(columns)
    if n == 0:
        return
    
    stats = []
    max_val = -np.inf
    min_val = np.inf
    for col in columns:
        m, s = data[col].mean(), data[col].std()
        print(f"[FinalPlot] Mean {col}: {m:.4f}")
        stats.append((col, m, s))
        max_val = max(max_val, m + s)
        min_val = min(min_val, m - s)

    if np.isfinite(max_val):
        y_max = max_val * 1.1
    else:
        y_max = None
    if np.isfinite(min_val):
        y_min = min_val * 1.1
    else:
        y_min = 0.0
    y_min = 0.0
    
    fig_width = max(8, n * 1.2)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 6))
    positions = np.arange(n)
    bar_width = 0.6

    for i, (col, m, s) in enumerate(stats):
        pos = positions[i]
        ax.bar(pos, m, width=bar_width, yerr=s, capsize=5, color='C0')

    ax.set_xticks(positions)
    ax.set_xticklabels([f'{col}' for col in columns], rotation=0, ha='center')
    ax.set_ylabel('Mean value')
    if y_max is not None:
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(bottom=y_min)

    ticks = ax.get_yticks()
    ticklabels = ["" if np.isclose(t, 0.0) else format(t, ".2g") for t in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)
    ax.minorticks_on()
    ax.grid(axis='y', alpha=0.6, which='major')
    ax.grid(axis='y', alpha=0.3, which='minor')
    plt.tight_layout()
    fig.savefig('final_questions_mean.png')
    plt.close(fig)

def final_plot_stacked(data):
    """Render 100% stacked bars for the final question block."""
    columns = data.columns[27:30]
    if len(columns) == 0:
        return
    fig_width = max(6, len(columns) * 1.2)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 6))
    bar_width = 0.6
    xticks, xticklabels = [], []
    plotted_categories = set()
    for idx, col in enumerate(columns):
        pos = idx
        shares, counts = likert_distribution(data[col].dropna())
        plot_likert_stack(ax, pos, shares, counts, bar_width, plotted_categories)
        xticks.append(pos)
        xticklabels.append(col)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Final questions")
    configure_likert_axes(ax)
    add_likert_legend(ax)
    fig.tight_layout(rect=[0, 0, 1, 1.0])
    fig.savefig('final_questions_likert.png')
    plt.close(fig)

def class_final_plot(data):
    """Similar to class_plot but for AB-AD columns."""
    columns = data.columns[27:30]  # AB to AD
    n = len(columns)
    if n == 0:
        return
    
    stats = []
    max_val = -np.inf
    min_val = np.inf
    classes = ordered_classes(data['Class'].unique())
    if not classes:
        return
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    class_colors = {cls: palette[i % len(palette)] for i, cls in enumerate(classes)}
    for col in columns:
        class_stats = {}
        for cls in classes:
            group = data[data['Class'] == cls]
            m, s = group[col].mean(), group[col].std()
            print(f"[ClassFinalPlot] Mean {col} ({cls}): {m:.4f}")
            class_stats[cls] = (m, s)
            max_val = max(max_val, m + s)
            min_val = min(min_val, m - s)
        stats.append((col, class_stats))

    if np.isfinite(max_val):
        y_max = max_val * 1.1
    else:
        y_max = None
    if np.isfinite(min_val):
        y_min = min_val * 1.1
    else:
        y_min = 0.0
    
    fig_width = max(8, n * 1.2)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 6))
    group_positions = np.arange(n)
    bar_width = 0.6
    offsets = np.linspace(-bar_width / 2, bar_width / 2, len(classes))

    plotted_labels = set()
    for i, (col, class_stats) in enumerate(stats):
        pos = group_positions[i]
        for j, cls in enumerate(classes):
            m, s = class_stats[cls]
            label = cls if cls not in plotted_labels else None
            ax.bar(
                pos + offsets[j],
                m,
                width=bar_width / len(classes),
                yerr=s,
                capsize=5,
                color=class_colors[cls],
                label=label,
            )
            if label:
                plotted_labels.add(cls)
    ax.set_xticks(group_positions)
    ax.set_xticklabels([f'{col}' for col in columns], rotation=0, ha='center')
    ax.set_ylabel('Mean value')
    ax.set_xlabel("Final questions grouped by users' knowledge class")
    if y_max is not None:
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(bottom=y_min)

    ticks = ax.get_yticks()
    ticklabels = ["" if np.isclose(t, 0.0) else format(t, ".2g") for t in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)
    ax.minorticks_on()
    ax.grid(axis='y', alpha=0.6, which='major')
    ax.grid(axis='y', alpha=0.3, which='minor')
    ax.legend()
    plt.tight_layout()
    fig.savefig('final_questions_mean_grouped_by_class.png')
    plt.close(fig)

def class_final_plot_stacked(data):
    """Render 100% stacked bars for final questions grouped by class."""
    if 'Class' not in data.columns:
        return
    columns = data.columns[27:30]
    if len(columns) == 0:
        return
    classes = ordered_classes(data['Class'].unique())
    if not classes:
        return
    fig_width = max(8, len(columns) * 1.6)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 6))
    group_positions = np.arange(len(columns))
    bar_group_width = 0.6
    offsets = np.linspace(-bar_group_width / 2, bar_group_width / 2, len(classes))
    bar_width = bar_group_width / len(classes)
    xticks, xticklabels = [], []
    plotted_categories = set()
    for i, col in enumerate(columns):
        for j, cls in enumerate(classes):
            group = data[data['Class'] == cls]
            shares, counts = likert_distribution(group[col].dropna())
            pos = group_positions[i] + offsets[j]
            plot_likert_stack(ax, pos, shares, counts, bar_width, plotted_categories)
            xticks.append(pos)
            xticklabels.append(f'{col}\n{cls}')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    ax.set_xlabel("Final questions grouped by class")
    configure_likert_axes(ax)
    add_likert_legend(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig('final_questions_likert_grouped_by_class.png')
    plt.close(fig)

def group_final_plot(data):
    """Like class_final_plot but grouped by whether the value in the 'First' column is 'PCBVR' or 'PCBS'."""
    columns = data.columns[27:30]  # AB to AD
    n = len(columns)
    if n == 0:
        return
    
    stats = []
    max_val = -np.inf
    min_val = np.inf
    groups = ['PCBVR', 'PCBS']
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:2]
    group_colors = {grp: default_colors[i % len(default_colors)] for i, grp in enumerate(groups)}
    for col in columns:
        group_stats = {}
        for grp in groups:
            group = data[data['First'] == grp]
            m, s = group[col].mean(), group[col].std()
            print(f"[GroupFinalPlot] Mean {col} (First={grp}): {m:.4f}")
            group_stats[grp] = (m, s)
            max_val = max(max_val, m + s)
            min_val = min(min_val, m - s)
        stats.append((col, group_stats))

    if np.isfinite(max_val):
        y_max = max_val * 1.1
    else:
        y_max = None
    if np.isfinite(min_val):
        y_min = min_val * 1.1
    else:
        y_min = 0.0
    y_min = 0.0
    
    fig_width = max(8, n * 1.0)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 6))
    group_positions = np.arange(n)
    # Use a smaller group width so bars in each group sit closer together
    bar_group_width = 0.6
    bar_width = bar_group_width
    # Place offsets within a tighter range so bars are packed more closely
    if len(groups) > 1:
        offsets = np.linspace(-bar_group_width / 4, bar_group_width / 4, len(groups))
    else:
        offsets = np.array([0.0])

    plotted_labels = set()
    for i, (col, group_stats) in enumerate(stats):
        pos = group_positions[i]
        for j, grp in enumerate(groups):
            m, s = group_stats[grp]
            label = grp if grp not in plotted_labels else None
            ax.bar(
                pos + offsets[j],
                m,
                width=bar_width / len(groups),
                yerr=s,
                capsize=5,
                color=group_colors[grp],
                label=label,
            )
            if label:
                plotted_labels.add(grp)
    ax.set_xticks(group_positions)
    ax.set_xticklabels([f'{col}' for col in columns], rotation=0, ha='center')
    ax.set_ylabel('Mean value')
    if y_max is not None:
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(bottom=y_min)
    ticks = ax.get_yticks()
    ticklabels = ["" if np.isclose(t, 0.0) else format(t, ".2g") for t in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)
    ax.minorticks_on()
    ax.grid(axis='y', alpha=0.6, which='major')
    ax.grid(axis='y', alpha=0.3, which='minor')
    ax.legend()
    plt.tight_layout()
    fig.savefig('final_questions_mean_grouped_by_first.png')
    plt.close(fig)

def group_final_plot_stacked(data):
    """Render 100% stacked bars for final questions grouped by first-experience."""
    if 'First' not in data.columns:
        return
    columns = data.columns[27:30]
    if len(columns) == 0:
        return
    groups = [grp for grp in ['PCBVR', 'PCBS'] if (data['First'] == grp).any()]
    if not groups:
        return
    fig_width = max(8, len(columns) * 1.4)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 6))
    group_positions = np.arange(len(columns))
    bar_group_width = 0.6
    offsets = np.linspace(-bar_group_width / 2, bar_group_width / 2, len(groups))
    bar_width = bar_group_width / len(groups)
    xticks, xticklabels = [], []
    plotted_categories = set()
    for i, col in enumerate(columns):
        for j, grp in enumerate(groups):
            subset = data[data['First'] == grp]
            shares, counts = likert_distribution(subset[col].dropna())
            pos = group_positions[i] + offsets[j]
            plot_likert_stack(ax, pos, shares, counts, bar_width, plotted_categories)
            xticks.append(pos)
            xticklabels.append(f'{col}\n{grp}')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    ax.set_xlabel("Final questions grouped by first experience")
    configure_likert_axes(ax)
    add_likert_legend(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig('final_questions_likert_grouped_by_first.png')
    plt.close(fig)

def main():
    data = load_data("results.tsv")
    print(data.describe())
    base_plot(data)
    base_plot_stacked(data)
    class_plot(data)
    class_plot_stacked(data)
    delta_plot(data)
    final_plot(data)
    final_plot_stacked(data)
    class_final_plot(data)
    class_final_plot_stacked(data)
    group_final_plot(data)
    group_final_plot_stacked(data)

if __name__ == "__main__":
    main()