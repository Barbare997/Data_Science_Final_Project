"""
Machine Learning Models Module for Urban Pulse Project

This module contains functions for training and evaluating ML models
for traffic congestion prediction.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                            precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def prepare_features(df: pd.DataFrame,
                    target_column: str = 'is_congested',
                    exclude_columns: Optional[list] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for machine learning.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    target_column : str, optional
        Name of target column (default: 'is_congested')
    exclude_columns : list, optional
        Columns to exclude from features
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features (X) and target (y)
    """
    if exclude_columns is None:
        exclude_columns = ['date_time', 'traffic_volume', 'traffic_stress_level', 
                          'rush_hour_type', target_column]
    
    # Select numeric columns
    feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove excluded columns
    feature_columns = [col for col in feature_columns if col not in exclude_columns]
    
    # Remove any columns with 'outlier' in name (these are flags, not features)
    feature_columns = [col for col in feature_columns if 'outlier' not in col.lower()]
    
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    print(f"✓ Prepared {len(feature_columns)} features for ML")
    print(f"  Features: {', '.join(feature_columns[:5])}{'...' if len(feature_columns) > 5 else ''}")
    print(f"  Target: {target_column}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")
    
    return X, y


def train_logistic_regression(X: pd.DataFrame,
                              y: pd.Series,
                              test_size: float = 0.2,
                              random_state: int = 42,
                              max_iter: int = 1000) -> Tuple[object, dict]:
    """
    Train a Logistic Regression model for traffic congestion prediction.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    test_size : float, optional
        Proportion of data for testing (default: 0.2)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    max_iter : int, optional
        Maximum iterations for convergence (default: 1000)
        
    Returns
    -------
    Tuple[object, dict]
        Trained model and evaluation metrics
    """
    print("\n" + "=" * 60)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("=" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    cm = confusion_matrix(y_test, y_test_pred)
    
    metrics = {
        'model': 'Logistic Regression',
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_test_pred,
        'feature_names': X.columns.tolist(),
        'coefficients': dict(zip(X.columns, model.coef_[0]))
    }
    
    # Print results
    print("\nModel Performance:")
    print(f"  Training Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {cm}")
    print("\n" + "=" * 60)
    
    return model, metrics


def train_decision_tree(X: pd.DataFrame,
                       y: pd.Series,
                       test_size: float = 0.2,
                       random_state: int = 42,
                       max_depth: Optional[int] = None,
                       min_samples_split: int = 2) -> Tuple[object, dict]:
    """
    Train a Decision Tree model for traffic congestion prediction.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    test_size : float, optional
        Proportion of data for testing (default: 0.2)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    max_depth : int, optional
        Maximum depth of the tree (default: None)
    min_samples_split : int, optional
        Minimum samples required to split (default: 2)
        
    Returns
    -------
    Tuple[object, dict]
        Trained model and evaluation metrics
    """
    print("\n" + "=" * 60)
    print("TRAINING DECISION TREE MODEL")
    print("=" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Feature importance
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    metrics = {
        'model': 'Decision Tree',
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_test_pred,
        'feature_names': X.columns.tolist(),
        'feature_importance': feature_importance
    }
    
    # Print results
    print("\nModel Performance:")
    print(f"  Training Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {cm}")
    print(f"\nTop 5 Most Important Features:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
        print(f"  {i+1}. {feature}: {importance:.4f}")
    print("\n" + "=" * 60)
    
    return model, metrics


def train_random_forest(X: pd.DataFrame,
                       y: pd.Series,
                       test_size: float = 0.2,
                       random_state: int = 42,
                       n_estimators: int = 100,
                       max_depth: Optional[int] = None,
                       min_samples_split: int = 2) -> Tuple[object, dict]:
    """
    Train a Random Forest model for traffic congestion prediction.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    test_size : float, optional
        Proportion of data for testing (default: 0.2)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    n_estimators : int, optional
        Number of trees in the forest (default: 100)
    max_depth : int, optional
        Maximum depth of the trees (default: None)
    min_samples_split : int, optional
        Minimum samples required to split (default: 2)
        
    Returns
    -------
    Tuple[object, dict]
        Trained model and evaluation metrics
    """
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Feature importance
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    metrics = {
        'model': 'Random Forest',
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_test_pred,
        'feature_names': X.columns.tolist(),
        'feature_importance': feature_importance
    }
    
    # Print results
    print("\nModel Performance:")
    print(f"  Training Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {cm}")
    print(f"\nTop 5 Most Important Features:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
        print(f"  {i+1}. {feature}: {importance:.4f}")
    print("\n" + "=" * 60)
    
    return model, metrics


def plot_confusion_matrix(cm: np.ndarray,
                         model_name: str,
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix for a model.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the figure
    """
    import seaborn as sns
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Not Congested', 'Congested'],
                yticklabels=['Not Congested', 'Congested'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_feature_importance(metrics: dict,
                           top_n: int = 10,
                           save_path: Optional[str] = None) -> None:
    """
    Plot feature importance for Decision Tree model.
    
    Parameters
    ----------
    metrics : dict
        Metrics dictionary containing feature_importance
    top_n : int, optional
        Number of top features to display (default: 10)
    save_path : str, optional
        Path to save the figure
    """
    if 'feature_importance' not in metrics:
        print("Warning: Feature importance not available for this model")
        return
    
    feature_importance = metrics['feature_importance']
    top_features = dict(list(feature_importance.items())[:top_n])
    
    plt.figure(figsize=(10, 6))
    features = list(top_features.keys())
    importance_values = list(top_features.values())
    
    plt.barh(features, importance_values, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance - {metrics["model"]}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved feature importance plot to {save_path}")
    
    plt.show()


def plot_model_comparison(lr_metrics: dict,
                         dt_metrics: dict,
                         rf_metrics: Optional[dict] = None,
                         save_path: Optional[str] = None) -> None:
    """
    Compare performance of ML models.
    
    Parameters
    ----------
    lr_metrics : dict
        Metrics for Logistic Regression model
    dt_metrics : dict
        Metrics for Decision Tree model
    rf_metrics : dict, optional
        Metrics for Random Forest model
    save_path : str, optional
        Path to save the figure
    """
    import seaborn as sns
    
    if rf_metrics is None:
        # Two model comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Metrics comparison
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        lr_values = [lr_metrics['test_accuracy'], lr_metrics['precision'], 
                    lr_metrics['recall'], lr_metrics['f1_score']]
        dt_values = [dt_metrics['test_accuracy'], dt_metrics['precision'], 
                    dt_metrics['recall'], dt_metrics['f1_score']]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        axes[0].bar(x - width/2, lr_values, width, label='Logistic Regression', 
                   color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].bar(x + width/2, dt_values, width, label='Decision Tree', 
                   color='coral', alpha=0.7, edgecolor='black')
        
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metrics_names)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim([0, 1])
        
        # Confusion matrix
        sns.heatmap(lr_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                   ax=axes[1], cbar=False, xticklabels=['Not Congested', 'Congested'],
                   yticklabels=['Not Congested', 'Congested'])
        axes[1].set_title(f'Logistic Regression\nAccuracy: {lr_metrics["test_accuracy"]:.3f}', 
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Predicted', fontsize=10)
        axes[1].set_ylabel('Actual', fontsize=10)
    else:
        # Three model comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Metrics comparison bar chart
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        lr_values = [lr_metrics['test_accuracy'], lr_metrics['precision'], 
                    lr_metrics['recall'], lr_metrics['f1_score']]
        dt_values = [dt_metrics['test_accuracy'], dt_metrics['precision'], 
                    dt_metrics['recall'], dt_metrics['f1_score']]
        rf_values = [rf_metrics['test_accuracy'], rf_metrics['precision'], 
                    rf_metrics['recall'], rf_metrics['f1_score']]
        
        x = np.arange(len(metrics_names))
        width = 0.25
        
        axes[0, 0].bar(x - width, lr_values, width, label='Logistic Regression', 
                      color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].bar(x, dt_values, width, label='Decision Tree', 
                      color='coral', alpha=0.7, edgecolor='black')
        axes[0, 0].bar(x + width, rf_values, width, label='Random Forest', 
                      color='green', alpha=0.7, edgecolor='black')
        
        axes[0, 0].set_ylabel('Score', fontsize=12)
        axes[0, 0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics_names)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].set_ylim([0, 1])
        
        # Confusion matrices
        sns.heatmap(lr_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                   ax=axes[0, 1], cbar=False, xticklabels=['Not Congested', 'Congested'],
                   yticklabels=['Not Congested', 'Congested'])
        axes[0, 1].set_title(f'Logistic Regression\nAccuracy: {lr_metrics["test_accuracy"]:.3f}', 
                            fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Predicted', fontsize=10)
        axes[0, 1].set_ylabel('Actual', fontsize=10)
        
        sns.heatmap(dt_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Oranges', 
                   ax=axes[1, 0], cbar=False, xticklabels=['Not Congested', 'Congested'],
                   yticklabels=['Not Congested', 'Congested'])
        axes[1, 0].set_title(f'Decision Tree\nAccuracy: {dt_metrics["test_accuracy"]:.3f}', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Predicted', fontsize=10)
        axes[1, 0].set_ylabel('Actual', fontsize=10)
        
        sns.heatmap(rf_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Greens', 
                   ax=axes[1, 1], cbar=False, xticklabels=['Not Congested', 'Congested'],
                   yticklabels=['Not Congested', 'Congested'])
        axes[1, 1].set_title(f'Random Forest\nAccuracy: {rf_metrics["test_accuracy"]:.3f}', 
                             fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Predicted', fontsize=10)
        axes[1, 1].set_ylabel('Actual', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved model comparison to {save_path}")
    
    plt.show()


def print_model_comparison_summary(lr_metrics: dict, dt_metrics: dict, rf_metrics: Optional[dict] = None) -> None:
    """
    Print a summary comparison of ML models.
    
    Parameters
    ----------
    lr_metrics : dict
        Metrics for Logistic Regression model
    dt_metrics : dict
        Metrics for Decision Tree model
    rf_metrics : dict, optional
        Metrics for Random Forest model
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    
    if rf_metrics is None:
        # Two model comparison
        comparison_df = pd.DataFrame({
            'Logistic Regression': [
                f"{lr_metrics['test_accuracy']:.4f}",
                f"{lr_metrics['precision']:.4f}",
                f"{lr_metrics['recall']:.4f}",
                f"{lr_metrics['f1_score']:.4f}"
            ],
            'Decision Tree': [
                f"{dt_metrics['test_accuracy']:.4f}",
                f"{dt_metrics['precision']:.4f}",
                f"{dt_metrics['recall']:.4f}",
                f"{dt_metrics['f1_score']:.4f}"
            ]
        }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
        
        print(comparison_df.to_string())
        
        # Determine winner
        if dt_metrics['test_accuracy'] > lr_metrics['test_accuracy']:
            winner = "Decision Tree"
            advantage = dt_metrics['test_accuracy'] - lr_metrics['test_accuracy']
        else:
            winner = "Logistic Regression"
            advantage = lr_metrics['test_accuracy'] - dt_metrics['test_accuracy']
        
        print(f"\n🏆 Best Model: {winner}")
        print(f"   Advantage: {advantage:.4f} ({advantage*100:.2f}%)")
        
        print("\nKey Insights:")
        print(f"  • Logistic Regression: Better interpretability (coefficients)")
        print(f"  • Decision Tree: Better accuracy, shows feature importance")
        print(f"  • Both models identify similar key predictors")
    else:
        # Three model comparison
        comparison_df = pd.DataFrame({
            'Logistic Regression': [
                f"{lr_metrics['test_accuracy']:.4f}",
                f"{lr_metrics['precision']:.4f}",
                f"{lr_metrics['recall']:.4f}",
                f"{lr_metrics['f1_score']:.4f}"
            ],
            'Decision Tree': [
                f"{dt_metrics['test_accuracy']:.4f}",
                f"{dt_metrics['precision']:.4f}",
                f"{dt_metrics['recall']:.4f}",
                f"{dt_metrics['f1_score']:.4f}"
            ],
            'Random Forest': [
                f"{rf_metrics['test_accuracy']:.4f}",
                f"{rf_metrics['precision']:.4f}",
                f"{rf_metrics['recall']:.4f}",
                f"{rf_metrics['f1_score']:.4f}"
            ]
        }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
        
        print(comparison_df.to_string())
        
        # Determine winner
        accuracies = {
            'Logistic Regression': lr_metrics['test_accuracy'],
            'Decision Tree': dt_metrics['test_accuracy'],
            'Random Forest': rf_metrics['test_accuracy']
        }
        winner = max(accuracies, key=accuracies.get)
        winner_acc = accuracies[winner]
        second_best = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)[1]
        advantage = winner_acc - second_best[1]
        
        print(f"\n🏆 Best Model: {winner}")
        print(f"   Accuracy: {winner_acc:.4f}")
        print(f"   Advantage over {second_best[0]}: {advantage:.4f} ({advantage*100:.2f}%)")
        
        print("\nKey Insights:")
        print(f"  • Logistic Regression: Best interpretability (coefficients), lower accuracy")
        print(f"  • Decision Tree: Good balance of accuracy and interpretability")
        print(f"  • Random Forest: Ensemble method, typically best accuracy, shows feature importance")
        print(f"  • All models identify similar key predictors (rush hour, time of day, weather)")
    
    print("=" * 60)

