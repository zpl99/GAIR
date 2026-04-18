# coding:utf-8
"""
@file: eval_accuracy_fn.py
@author: Zeping Liu
@ide: PyCharm
@createTime: 2024.09
@contactInformation: zeping.liu@utexas.edu
@Function: calculate accuracy between input and target
"""

import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree._criterion import MAE



def regression_accuracy(y_pred, y_true):
    """
    Calculates common regression evaluation metrics: MSE, RMSE, MAE, and R-squared.

    Parameters:
    y_true (torch.Tensor): Ground truth values, shape [B, 1]
    y_pred (torch.Tensor): Predicted values, shape [B, 1]

    Returns:
    dict: A dictionary containing the computed metrics:
        - 'MSE': Mean Squared Error
        - 'RMSE': Root Mean Squared Error
        - 'MAE': Mean Absolute Error
        - 'R2': R-squared (Coefficient of Determination)
    """
    # Mean Squared Error (MSE)
    mse = torch.mean((y_true - y_pred) ** 2)

    # Root Mean Squared Error (RMSE)
    rmse = torch.sqrt(mse)

    # Mean Absolute Error (MAE)
    mae = torch.mean(torch.abs(y_true - y_pred))

    # R-squared (R^2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


# Function to calculate and print MSE, R2, RMSE, and MAE for each indicator
def calculate_and_print_metrics(predictions, true_values):
    indicators = ["popden_cbg", "bachelor_p", "health", "poc_cbg", "mhincome_c", "stationdis",
                  "walkbike_p", "over65_per", "violent", "sky"]

    metrics = {"indicator": indicators, "MSE": [], "R2": [], "RMSE": [], "MAE": []}
    eval_results = {}
    # Calculate metrics for each indicator
    for i in range(predictions.shape[1]):
        # Extract predictions and true values for the i-th indicator
        pred_i = predictions[:, i].numpy()
        true_i = true_values[:, i].numpy()

        # Calculate metrics
        mse = mean_squared_error(true_i, pred_i)
        r2 = r2_score(true_i, pred_i)
        rmse = torch.sqrt(torch.tensor(mse)).item()
        mae = mean_absolute_error(true_i, pred_i)

        # Append results to the dictionary
        metrics["MSE"].append(mse)
        metrics["R2"].append(r2)
        metrics["RMSE"].append(rmse)
        metrics["MAE"].append(mae)

    # Print metrics in the requested format
    for i in range(len(indicators)):
        print(
            f"{indicators[i]}: MSE = {metrics['MSE'][i]:.4f}, R2 = {metrics['R2'][i]:.4f}, RMSE = {metrics['RMSE'][i]:.4f}, MAE = {metrics['MAE'][i]:.4f}")
        eval_results.update({indicators[i]: {"MSE": metrics['MSE'][i], "R2": metrics['R2'][i], "RMSE": metrics['RMSE'][i], "MAE": metrics['MAE'][i]}})
    return eval_results
# Function to calculate and print MSE, R2, RMSE, and MAE for each indicator
def calculate_and_print_metrics_perception(predictions, true_values):
    indicators = ['Beautiful', 'Boring', 'Depressing', 'Lively', 'Safe', 'Wealthy']

    metrics = {"indicator": indicators, "MSE": [], "R2": [], "RMSE": [], "MAE": []}

    eval_results = {}
    # Calculate metrics for each indicator
    for i in range(predictions.shape[1]):
        # Extract predictions and true values for the i-th indicator
        pred_i = predictions[:, i].numpy()
        true_i = true_values[:, i].numpy()

        # Calculate metrics
        mse = mean_squared_error(true_i, pred_i)
        r2 = r2_score(true_i, pred_i)
        rmse = torch.sqrt(torch.tensor(mse)).item()
        mae = mean_absolute_error(true_i, pred_i)

        # Append results to the dictionary
        metrics["MSE"].append(mse)
        metrics["R2"].append(r2)
        metrics["RMSE"].append(rmse)
        metrics["MAE"].append(mae)

    # Print metrics in the requested format
    for i in range(len(indicators)):
        print(
            f"{indicators[i]}: MSE = {metrics['MSE'][i]:.4f}, R2 = {metrics['R2'][i]:.4f}, RMSE = {metrics['RMSE'][i]:.4f}, MAE = {metrics['MAE'][i]:.4f}")
        eval_results.update({indicators[i]: {"MSE": metrics['MSE'][i], "R2": metrics['R2'][i], "RMSE": metrics['RMSE'][i], "MAE": metrics['MAE'][i]}})
    return eval_results
def regression_accuracy(y_pred, y_true):
    """
    Calculates common regression evaluation metrics: MSE, RMSE, MAE, and R-squared.

    Parameters:
    y_true (torch.Tensor): Ground truth values, shape [B, 1]
    y_pred (torch.Tensor): Predicted values, shape [B, 1]

    Returns:
    dict: A dictionary containing the computed metrics:
        - 'MSE': Mean Squared Error
        - 'RMSE': Root Mean Squared Error
        - 'MAE': Mean Absolute Error
        - 'R2': R-squared (Coefficient of Determination)
    """
    # Mean Squared Error (MSE)
    mse = torch.mean((y_true - y_pred) ** 2)

    # Root Mean Squared Error (RMSE)
    rmse = torch.sqrt(mse)

    # Mean Absolute Error (MAE)
    mae = torch.mean(torch.abs(y_true - y_pred))

    # R-squared (R^2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


def calculate_classification_metrics(outputs, labels):
    """
    计算分类模型的 hit@1、hit@3 和 Mean Reciprocal Rank (MRR) 指标。

    Parameters:
    ----------
    outputs : torch.Tensor
        模型输出的 logits 或概率，形状为 [batch_size, num_classes]。其中每一行表示一个样本对每个类别的预测得分。
    labels : torch.Tensor
        真实的类别标签，形状为 [batch_size]，每个值为 [0, num_classes - 1] 之间的整数，表示每个样本的真实类别。

    Returns:
    -------
    dict
        包含三个指标的字典：
        - "hit@1": float, hit@1 的准确率，表示模型预测的最高概率的类别是否为真实类别。
        - "hit@3": float, hit@3 的准确率，表示真实类别是否在模型预测的前三个类别中。
        - "Mean Reciprocal Rank": float, MRR 平均倒数排名，表示真实类别在预测结果中的排名倒数的平均值。
    """

    # 获取每个样本的 top-1 和 top-3 预测结果
    _, top1_indices = torch.topk(outputs, 1, dim=1)
    _, top3_indices = torch.topk(outputs, 3, dim=1)

    # 计算 hit@1
    hit1 = (top1_indices == labels.view(-1, 1)).float().mean().item()

    # 计算 hit@3
    hit3 = (top3_indices == labels.view(-1, 1)).float().sum(dim=1).mean().item()

    # 计算 Reciprocal Rank
    _, sorted_indices = torch.sort(outputs, descending=True, dim=1)
    ranks = (sorted_indices == labels.view(-1, 1)).nonzero(as_tuple=True)[1] + 1  # 找到排名并从1开始计数
    reciprocal_ranks = 1.0 / ranks.float()
    mean_reciprocal_rank = reciprocal_ranks.mean().item()

    return {
        "hit@1": hit1,
        "hit@3": hit3,
        "Mean_Reciprocal_Rank": mean_reciprocal_rank
    }

import torch

def evaluate_small_classes_classification(outputs, labels, num_classes):
    """
    评估分类任务（支持二分类、多分类）的性能。

    Parameters:
    ----------
    outputs : torch.Tensor
        模型输出的 logits 或概率，形状为 [batch_size, num_classes]。
    labels : torch.Tensor
        真实的类别标签，形状为 [batch_size]，每个值为 [0, num_classes - 1]。

    Returns:
    -------
    dict
        包含分类任务的评估指标。
    """

    # 转换为预测类别
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    metrics = {}

    # 通用指标：Accuracy
    metrics["accuracy"] = accuracy_score(labels, preds)

    if num_classes == 2:
        # 二分类指标
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 正类概率
        metrics["precision"] = precision_score(labels, preds)
        metrics["recall"] = recall_score(labels, preds)
        metrics["f1_score"] = f1_score(labels, preds)
        # 检查 labels 是否包含至少两个类别
        if len(set(labels)) > 1:
            metrics["auc_roc"] = roc_auc_score(labels, probs)
        else:
            metrics["auc_roc"] = 0.0  # 或设置为某个默认值，比如 0.0
    else:
        # 多分类指标
        metrics["macro_precision"] = precision_score(labels, preds, average='macro')
        metrics["macro_recall"] = recall_score(labels, preds, average='macro')
        metrics["macro_f1_score"] = f1_score(labels, preds, average='macro')
        metrics["weighted_f1_score"] = f1_score(labels, preds, average='weighted')

        # 针对每个类别计算指标
        for i in range(num_classes):
            metrics[f"precision_class_{i}"] = precision_score(labels, preds, labels=[i], average='micro',zero_division=0)
            metrics[f"recall_class_{i}"] = recall_score(labels, preds, labels=[i], average='micro')
            metrics[f"f1_score_class_{i}"] = f1_score(labels, preds, labels=[i], average='micro')

    return metrics
