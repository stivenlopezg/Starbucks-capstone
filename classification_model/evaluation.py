import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, \
    precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

th_props = [
    ('font-size', '11px'),
    ('text-align', 'center'),
    ('font-weight', 'bold'),
    ('color', '#6d6d6d'),
    ('background-color', '#f7f7f9')
]

# Set CSS properties for td elements in dataframe
td_props = [
    ('font-size', '11px')
]

# Set table styles
styles = [
    dict(selector="th", props=th_props),
    dict(selector="td", props=td_props)
]


def generate_report(y_true: list, y_pred: list):
    """
    Generate a DataFrame with the main classification metrics
    :param y_true: Observed value
    :param y_pred: Predicted value
    :return: pd.DataFrame
    """
    report = np.round(pd.DataFrame(classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)), 2).T
    return report


def confusion_matrix(y_true: list, y_pred: list, normalize: bool = True):
    """
    Generates a confusion matrix
    :param y_true: Observed value
    :param y_pred: Predicted value
    :param normalize: Boolean on normalize for each row
    :return: pd.DataFrame
    """
    if normalize:
        table = np.round(pd.crosstab(index=y_true, columns=y_pred,
                                     rownames=['Observed'], colnames=['Predicted'], normalize='index'), 2)
    else:
        table = np.round(pd.crosstab(index=y_true, columns=y_pred, rownames=['Observed'], colnames=['Predicted']), 2)
    table.rename(columns={0.: 'offer viewed', 1.: 'offer completed'},
                 index={0.: 'offer viewed', 1.: 'offer completed'}, inplace=True)
    table = table.style.format({'offer viewed': '{:.1%}',
                                'offer completed': '{:.1%}'}) \
                       .set_table_styles(styles) \
                       .set_caption('Confusion matrix') \
                       .background_gradient(cmap='Blues')
    return table


def calculate_metrics(y_true: list, y_pred: list):
    """
    Calculate the main classification metrics
    :param y_true: Observed value
    :param y_pred: Predicted value
    :return: dict
    """
    metrics = {
        'roc': np.round(roc_auc_score(y_true=y_true, y_score=y_pred), 2),
        'accuracy': np.round(accuracy_score(y_true=y_true, y_pred=y_pred), 2),
        'precision': np.round(precision_score(y_true=y_true, y_pred=y_pred), 2),
        'recall': np.round(recall_score(y_true=y_true, y_pred=y_pred), 2),
        'f1': np.round(f1_score(y_true=y_true, y_pred=y_pred), 2),
        'matthews_corr': np.round(matthews_corrcoef(y_true=y_true, y_pred=y_pred), 2)
    }
    return metrics


def metrics_summary(metrics: dict):
    """
    Print a text summary of the main classification metrics
    :param metrics: dictionary with main classification metrics
    :return: string
    """
    print(f'The AUC is: {metrics["roc"]}')
    print(f'The accuracy is: {metrics["accuracy"]}')
    print(f'The precision is: {metrics["precision"]}')
    print(f'The recall is: {metrics["recall"]}')
    print(f'The F1 score is: {metrics["f1"]}')
    print(f'The Matthews correlation is: {metrics["matthews_corr"]}')
