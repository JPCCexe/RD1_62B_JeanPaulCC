import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
import os

XCLIP_CSV = 'Results_XClip.csv'
VIDEOMAE_CSV = 'Results_VideoMAE.csv'
OUTPUT_DIR = 'results_output'

LABELS = [
    'sitting', 'drinking', 'eating', 'reading book', 'opening door',
    'lying down', 'running', 'walking', 'cooking', 'writing',
    'watching tv', 'washing hands'
]


def parse_actions(text):
    if pd.isna(text) or str(text).strip() == '':
        return []
    return [x.strip() for x in str(text).split(';') if x.strip()]


def build_binary_matrix(df, column_name):
    matrix = []
    for value in df[column_name]:
        actions = set(parse_actions(value))
        matrix.append([1 if label in actions else 0 for label in LABELS])
    return matrix


def load_model_rows(csv_path):
    df = pd.read_csv(csv_path)
    df_no_overall = df[df['video_name'].astype(str).str.upper() != 'OVERALL'].copy()
    summary = pd.DataFrame([{
        'Model': df['video_name'].iloc[-1] if str(df['video_name'].iloc[-1]).upper() == 'OVERALL' else csv_path.replace('Results_', '').replace('.csv', ''),
        'Accuracy': df['accuracy'].iloc[-1],
        'Macro Precision': df['macro_precision'].iloc[-1],
        'Macro Recall': df['macro_recall'].iloc[-1],
        'Macro F1': df['macro_F1'].iloc[-1],
        'Avg FPS': df['avg_fps'].iloc[-1]
    }])
    return df_no_overall, summary


def save_table_image(df, title, output_path):
    fig, ax = plt.subplots(figsize=(8, 1.8))
    ax.axis('off')
    table = ax.table(
        cellText=df.round(4).values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    plt.title(title, fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close()


def save_confusion_panels(df, model_name, output_path):
    y_true = build_binary_matrix(df, 'gt_actions')
    y_pred = build_binary_matrix(df, 'predicted_actions')
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    axes = axes.flatten()

    for i, label in enumerate(LABELS):
        cm = mcm[i]
        ax = axes[i]
        ax.imshow(cm, cmap='Blues')
        ax.set_title(label, fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred 0', 'Pred 1'], fontsize=7)
        ax.set_yticklabels(['True 0', 'True 1'], fontsize=7)

        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha='center', va='center', fontsize=8)

    plt.suptitle(f'{model_name} Multilabel Confusion Matrices', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    xclip_df, xclip_summary = load_model_rows(XCLIP_CSV)
    videomae_df, videomae_summary = load_model_rows(VIDEOMAE_CSV)

    xclip_summary.iloc[0, 0] = 'X-CLIP'
    videomae_summary.iloc[0, 0] = 'VideoMAE'

    xclip_summary.to_csv(f'{OUTPUT_DIR}/xclip_summary.csv', index=False)
    videomae_summary.to_csv(f'{OUTPUT_DIR}/videomae_summary.csv', index=False)

    save_table_image(xclip_summary, 'X-CLIP Results Summary', f'{OUTPUT_DIR}/xclip_summary_table.png')
    save_table_image(videomae_summary, 'VideoMAE Results Summary', f'{OUTPUT_DIR}/videomae_summary_table.png')

    save_confusion_panels(xclip_df, 'X-CLIP', f'{OUTPUT_DIR}/xclip_confusion_panels.png')
    save_confusion_panels(videomae_df, 'VideoMAE', f'{OUTPUT_DIR}/videomae_confusion_panels.png')

    print('Done')


if __name__ == '__main__':
    main()