import numpy as np
import pandas as pd
import scipy.stats

results_pymfe = pd.read_csv("results_pymfe.csv", header=0, index_col=0)
results_tspymfe = pd.read_csv("results_tspymfe.csv", header=0, index_col=0)

print("pymfe (unsupervised meta-features) accuracy:")
print(results_pymfe)
print()
print("tspymfe (temporal meta-features) accuracy:")
print(results_tspymfe)

summary = pd.DataFrame([
    results_pymfe.mean(),
    results_pymfe.std(),
    results_tspymfe.mean(),
    results_tspymfe.std(),
    results_tspymfe.mean() - results_pymfe.mean(),
    results_pymfe.std() + results_tspymfe.std(),
])

summary.index = [
    "mean(pymfe)",
    "std(pymfe)",
    "mean(tspymfe)",
    "std(tspymfe)",
    "mean(tspymfe) - mean(pymfe)",
    "std(pymfe) + std(tspymfe)",
]

for test in results_pymfe.columns:
    acc_u = results_pymfe[test].values
    acc_t = results_tspymfe[test].values

    # Calculate the T-test for the means of two independent samples of scores.
    _, t_p_val_a = scipy.stats.ttest_ind(acc_u, acc_t, equal_var=True)
    _, t_p_val_b = scipy.stats.ttest_ind(acc_u, acc_t, equal_var=False)
    """
    We can use this test, if we observe two independent samples from the same or different population, e.g. exam scores of boys and girls or of two ethnic groups. The test measures whether the average (expected) value differs significantly across samples. If we observe a large p-value, for example larger than 0.05 or 0.1, then we cannot reject the null hypothesis of identical average scores. If the p-value is smaller than the threshold, e.g. 1%, 5% or 10%, then we reject the null hypothesis of equal averages.
    """

    # Compute the Kruskal-Wallis H-test for independent samples.
    _, h_p_val = scipy.stats.kruskal(acc_u, acc_t)
    """
    The Kruskal-Wallis H-test tests the null hypothesis that the population median of all of the groups are equal. It is a non-parametric version of ANOVA. The test works on 2 or more independent samples, which may have different sizes. Note that rejecting the null hypothesis does not indicate which of the groups differs. Post hoc comparisons between groups are required to determine which groups are different.
    """

    print(f"Experiment: {test}")
    print("   (t-test) p-value (equal variances of U and T)", t_p_val_a)
    print("   (t-test) p-value (maybe not equal variaces of U and T)", t_p_val_b)
    print("   (h-test) p-value:", h_p_val)
    print()


print("Summary:")
print(summary)
