import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to your Excel file
file_path = "/path/to/BIDS/dataset/with/tractography/results/stats_results.xlsx"

# Fixed orders
dwi_order = [
    "DTI-AP-0-600",
    "DTI-AP_2p5mm_ZOOMit",
    "DTI-AP-0-800_with_body_array"
]
strategy_order = ["none", "moco", "mppca_gibbs_eddy_topup"]

# Color map per DWI
dwi_palette = {
    "DTI-AP-0-600": "#1f77b4",               # blue
    "DTI-AP_2p5mm_ZOOMit": "#ff7f0e",        # orange
    "DTI-AP-0-800_with_body_array": "#2ca02c" # green
}

# Read all sheets
sheets = pd.read_excel(file_path, sheet_name=None)

for sheet_name, df in sheets.items():
    # apply your filter
    #filt = df[df["select_among_max_seeds"] == 10000].copy()
    filt = df[df["min_FOD_to_start"] == 0.08].copy()
    if filt.empty:
        continue

    # label for x-axis
    filt["label"] = filt["DWI"] + " + " + filt["Strategy"]

    # build ordered categories
    categories = []
    for dwi in dwi_order:
        for strat in strategy_order:
            lbl = f"{dwi} + {strat}"
            if lbl in filt["label"].values:
                categories.append(lbl)

    # build a parallel list of colors
    bar_colors = [dwi_palette[lbl.split(" + ")[0]] for lbl in categories]

    # plot
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=filt,
        x="label",
        y="Symmetry",
        order=categories,
        palette=bar_colors,
        ci="sd",          # show ±1 SD error bars; use ci=None to disable
        edgecolor="black" # optional: outline each bar
    )

    # remove legend (not needed for barplot without hue)
    # remove x-tick labels
    ax = plt.gca()
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([])

    plt.ylim(0, 1)
    #plt.title(f"Left–Right FA Symmetry ({sheet_name})")
    #plt.ylabel("Symmetry (0–1)")
    plt.tight_layout()

    # save
    out_name = f"symmetry_{sheet_name.replace(' ', '_')}.png"
    save_path = f"/path/to/BIDS/dataset/with/tractography/results/{out_name}"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved bar plot for sheet '{sheet_name}' as '{out_name}'")
