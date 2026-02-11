import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
#
# Leider ging mir die aktuellste Version dieser Datei im Kampf mit Git verloren,
# daher musste ich sie von meinem Server wiederherstellen. Es könnte sein, dass
# Errors auftreten, da ich nicht mehr genau weiß, ob ich alle Fixes in dieser
# Version eingebaut habe.
#

def create_classifier_comparison(result_orig, result_anon, output_path='outputs/classifier_comparison.png'):
    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(2, 3, 1)
    models = ['Original Data', 'Anonymized Data']
    accuracies = [result_orig[1]['accuracy'], result_anon[1]['accuracy']]
    colors = ['#2ecc71', '#3498db']
    bars = ax1.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2%}', ha='center', va='bottom', fontweight='bold')

    ax2 = plt.subplot(2, 3, 2)
    cm_orig = confusion_matrix(result_orig[1]['y_test'], result_orig[1]['y_pred'], labels=[0, 1])
    sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Greens', ax=ax2, cbar=False)
    ax2.set_title('Confusion Matrix - Original Data', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=11)
    ax2.set_xlabel('Predicted Label', fontsize=11)

    ax3 = plt.subplot(2, 3, 3)
    cm_anon = confusion_matrix(result_anon[1]['y_test'], result_anon[1]['y_pred'], labels=[0, 1])
    sns.heatmap(cm_anon, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=False)
    ax3.set_title('Confusion Matrix - Anonymized Data', fontsize=14, fontweight='bold')
    ax3.set_ylabel('True Label', fontsize=11)
    ax3.set_xlabel('Predicted Label', fontsize=11)

    ax4 = plt.subplot(2, 3, 4)

    def safe_roc(y_true, y_prob):
        if len(np.unique(y_true)) < 2:
            return [0, 1], [0, 1], 0.5  # Return dummy diagonal if only 1 class exists
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        return fpr, tpr, auc(fpr, tpr)

    fpr_orig, tpr_orig, roc_auc_orig = safe_roc(result_orig[1]['y_test'], result_orig[1]['y_pred_proba'])
    ax4.plot(fpr_orig, tpr_orig, color='#2ecc71', lw=2,
             label=f'Original (AUC = {roc_auc_orig:.3f})')

    fpr_anon, tpr_anon, roc_auc_anon = safe_roc(result_anon[1]['y_test'], result_anon[1]['y_pred_proba'])
    ax4.plot(fpr_anon, tpr_anon, color='#3498db', lw=2,
             label=f'Anonymized (AUC = {roc_auc_anon:.3f})')

    ax4.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    ax4.set_xlabel('False Positive Rate', fontsize=11)
    ax4.set_ylabel('True Positive Rate', fontsize=11)
    ax4.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax4.legend(loc="lower right")
    ax4.grid(alpha=0.3)

    ax5 = plt.subplot(2, 3, 5)
    metrics = ['Precision', 'Recall', 'F1-Score']

    def get_w_avg(report):
        w = report.get('weighted avg', {})
        return [w.get('precision', 0), w.get('recall', 0), w.get('f1-score', 0)]

    orig_metrics = get_w_avg(result_orig[1]['report'])
    anon_metrics = get_w_avg(result_anon[1]['report'])

    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax5.bar(x - width / 2, orig_metrics, width, label='Original', color='#2ecc71', alpha=0.7, edgecolor='black')
    bars2 = ax5.bar(x + width / 2, anon_metrics, width, label='Anonymized', color='#3498db', alpha=0.7,
                    edgecolor='black')

    ax5.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax5.set_title('Metrics Comparison (Weighted Avg)', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics)
    ax5.legend()
    ax5.set_ylim([0, 1])
    ax5.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    acc_diff = (result_anon[1]['accuracy'] - result_orig[1]['accuracy']) * 100
    auc_diff = (roc_auc_anon - roc_auc_orig) * 100

    summary_text = f"""
    Performance Impact of Anonymization
    {'=' * 40}

    Original Data Accuracy:      {result_orig[1]['accuracy']:.2%}
    Anonymized Data Accuracy:    {result_anon[1]['accuracy']:.2%}
    Accuracy Change:             {acc_diff:+.2f}%

    Original AUC-ROC:            {roc_auc_orig:.3f}
    Anonymized AUC-ROC:          {roc_auc_anon:.3f}
    AUC Change:                  {auc_diff:+.2f}%

    {'=' * 40}
    Summary:
    {'Minimal' if abs(acc_diff) < 2 else 'Moderate' if abs(acc_diff) < 5 else 'Significant'} impact on model performance
    """

    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Classifier Comparison: Original vs Anonymized Data',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("\n" + "=" * 50)
    print(f"Visualization saved to: {output_path}")
    print("=" * 50)

def create_k_comparison(original_result, k_results, output_path='outputs/k_comparison.png'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Classifier Quality vs K-Anonymity Level', fontsize=16, fontweight='bold')
    k_values = sorted(k_results.keys())

    orig_metrics = original_result[1]  # original_result is (classifier, metrics_dict)

    accuracies_mean = [k_results[k]['accuracy_mean'] for k in k_values]
    accuracies_std = [k_results[k]['accuracy_std'] for k in k_values]

    axes[0, 0].errorbar(k_values, accuracies_mean, yerr=accuracies_std,
                        marker='o', linewidth=2, markersize=8, capsize=5, capthick=2)
    axes[0, 0].axhline(y=orig_metrics['accuracy'], color='r', linestyle='--',
                       label=f'Original: {orig_metrics["accuracy"]:.2%}')
    axes[0, 0].set_xlabel('k-Value', fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy', fontweight='bold')
    axes[0, 0].set_title('Accuracy vs K-Anonymity Level (with SD)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_xticks(k_values)

    precisions_mean = [k_results[k]['precision_mean'] for k in k_values]
    precisions_std = [k_results[k]['precision_std'] for k in k_values]
    recalls_mean = [k_results[k]['recall_mean'] for k in k_values]
    recalls_std = [k_results[k]['recall_std'] for k in k_values]
    f1_scores_mean = [k_results[k]['f1_score_mean'] for k in k_values]
    f1_scores_std = [k_results[k]['f1_score_std'] for k in k_values]

    axes[0, 1].errorbar(k_values, precisions_mean, yerr=precisions_std,
                        marker='o', label='Precision', linewidth=2, capsize=4)
    axes[0, 1].errorbar(k_values, recalls_mean, yerr=recalls_std,
                        marker='s', label='Recall', linewidth=2, capsize=4)
    axes[0, 1].errorbar(k_values, f1_scores_mean, yerr=f1_scores_std,
                        marker='^', label='F1-Score', linewidth=2, capsize=4)
    axes[0, 1].set_xlabel('k-Value', fontweight='bold')
    axes[0, 1].set_ylabel('Score', fontweight='bold')
    axes[0, 1].set_title('Precision, Recall, F1-Score vs K (with SD)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_xticks(k_values)

    accuracy_loss_mean = [(orig_metrics['accuracy'] - k_results[k]['accuracy_mean']) * 100
                          for k in k_values]
    accuracy_loss_std = [k_results[k]['accuracy_std'] * 100 for k in k_values]

    axes[1, 0].bar(k_values, accuracy_loss_mean, color='coral', alpha=0.7,
                   yerr=accuracy_loss_std, capsize=5, error_kw={'linewidth': 2})
    axes[1, 0].set_xlabel('k-Value', fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy Loss (%)', fontweight='bold')
    axes[1, 0].set_title('Accuracy Loss Compared to Original Data (with SD)')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_xticks(k_values)

    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')

    table_data = []
    table_data.append(['k', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

    orig_stats_1 = orig_metrics['report'].get('weighted avg', {})
    table_data.append(['Original',
                       f"{orig_metrics['accuracy']:.3f}",
                       f"{orig_stats_1.get('precision', 0):.3f}",
                       f"{orig_stats_1.get('recall', 0):.3f}",
                       f"{orig_stats_1.get('f1-score', 0):.3f}"])

    for k in k_values:
        table_data.append([
            str(k),
            f"{k_results[k]['accuracy_mean']:.3f}±{k_results[k]['accuracy_std']:.3f}",
            f"{k_results[k]['precision_mean']:.3f}±{k_results[k]['precision_std']:.3f}",
            f"{k_results[k]['recall_mean']:.3f}±{k_results[k]['recall_std']:.3f}",
            f"{k_results[k]['f1_score_mean']:.3f}±{k_results[k]['f1_score_std']:.3f}"
        ])

    table = axes[1, 1].table(cellText=table_data, cellLoc='center', loc='center',
                             colWidths=[0.15, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(1, i)].set_facecolor('#E8F5E9')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()