import matplotlib.pyplot as plt

# PIE CHART
# Data
total_births = 114_263
preterm_births = 6_629
extremely_preterm = 2_222
term_births = total_births - preterm_births

# Pie chart
labels = ['Term births', 'Preterm (non-extremely)', 'Extremely preterm']
sizes = [
    term_births,
    preterm_births - extremely_preterm,
    extremely_preterm
]
colors = ['#a8dadc', '#fcbf49', '#e63946']
explode = (0, 0.05, 0.1)

plt.figure(figsize=(7, 7))
wedges, texts, autotexts = plt.pie(
    sizes,
    explode=explode,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    shadow=True
)

plt.title("Births in Sweden (2021)\nHighlighting Extremely Preterm Children", fontsize=14)
plt.tight_layout()

# Show the plot
# plt.show()


# BARPLOT MOTOR
import matplotlib.pyplot as plt
import numpy as np

# Data
groups = ['Preterm', 'Extremely preterm']
criteria = ['Stricter (≤5th percentile)', 'Less strict (≤15th percentile)']

# Prevalence ranges (%)
# Format: (mean, lower, upper)
data = {
    'Preterm': [(22.5, 8, 37), (41.5, 12, 71)],
    'Extremely preterm': [(22.5, 8, 37), (41.5, 12, 71)]  # using same ranges since both mentioned together
}

x = np.arange(len(groups))  # group positions
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#e63946','#fcbf49']  # teal, coral

# Plot bars with error bars (upper-lower as asymmetric)
for i, crit in enumerate(criteria):
    means = [data[g][i][0] for g in groups]
    lower = [data[g][i][0] - data[g][i][1] for g in groups]
    upper = [data[g][i][2] - data[g][i][0] for g in groups]
    ax.bar(x + i*width - width/2, means, width, color=colors[i], label=crit,
           yerr=[lower, upper], capsize=5)

ax.set_ylabel('Prevalence of motor problems (%)')
ax.set_title('Reported prevalence of motor problems in preterm children')
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
#plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Groups and outcomes
groups = ['Extremely preterm', 'Term-born']
outcomes = ['Cerebral Palsy (CP)', 'DCD / non-CP motor impairment']

# Data: (mean, lower, upper)
# CP: 7–20% in EP; assume ~13.5% mean
# DCD/non-CP: 20–30% in EP; 0–7% in term; use midpoints for means
data = {
    'Extremely preterm': [(13.5, 7, 20), (25, 20, 30)],
    'Term-born': [(0, 0, 0), (3.5, 0, 7)]
}

# Set up positions
x = np.arange(len(groups))
width = 0.35

# Colors (colorblind-friendly)
colors = ['#0072B2', '#E69F00']  # blue and orange

fig, ax = plt.subplots(figsize=(8, 6))

for i, outcome in enumerate(outcomes):
    means = [data[g][i][0] for g in groups]
    lower = [data[g][i][0] - data[g][i][1] for g in groups]
    upper = [data[g][i][2] - data[g][i][0] for g in groups]

    ax.bar(
        x + i * width - width / 2,
        means,
        width,
        yerr=[lower, upper],
        capsize=5,
        color=colors[i],
        edgecolor='black',
        label=outcome
    )

# Labels and formatting
ax.set_ylabel('Prevalence / Incidence (%)', fontsize=12)
ax.set_title('Motor Outcomes in Extremely Preterm vs Term-born Children', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=11)
ax.legend(title='Condition')
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
#plt.show()

# PRELIMINARY RESULTS MOST IMPORTANT FEATURES
import matplotlib.pyplot as plt
import numpy as np

# Features and normalized importances (example values based on your image)
features = [
    'lh_middletemporal_sulc_depth',
    'lh_middletemporal_thickness',
    'lh_cuneus_sul_depth'
]
ridge_importance = [0.8, 1.0, 0.7]
rf_importance = [0.5, 1.0, 0.2]

x = np.arange(len(features))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

bars_ridge = ax.bar(x - width/2, ridge_importance, width, label='Ridge', color='#a8dadc')
bars_rf = ax.bar(x + width/2, rf_importance, width, label='Random Forest', color= '#fcbf49')

# Highlight the 2 most important features (visually)
for i in [0, 1]:
    bars_ridge[i].set_edgecolor('red')
    bars_ridge[i].set_linewidth(3)
    bars_rf[i].set_edgecolor('red')
    bars_rf[i].set_linewidth(3)

ax.set_ylabel('Normalized importance', fontsize=12)
ax.set_title('Top 3 Most Predictive Features', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=35, ha='right')
ax.legend()

plt.tight_layout()
plt.show()




