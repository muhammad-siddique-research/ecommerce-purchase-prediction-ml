"""UCI online-shopping classification benchmark (Google Colab and local Python).

Examples:
    python -m src.run_experiment
    python -m src.run_experiment --csv data/online_shoppers_intention.csv
    python -m src.run_experiment --csv data/online_shoppers_intention.csv --fast
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use("Agg")  # Safe on CI / servers; figures are saved to files.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
CATEGORICAL = [
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType", "Weekend",
]
TARGET = "Revenue"


def load_dataset(csv_path: str | None = None) -> pd.DataFrame:
    """Use a local CSV if supplied; otherwise fetch UCI 468 when online."""
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise RuntimeError("Install dependencies with `pip install -r requirements.txt`.") from exc
        try:
            dataset = fetch_ucirepo(id=468)
            df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        except Exception as exc:
            raise RuntimeError(
                "UCI download failed. Download online_shoppers_intention.csv "
                "from https://archive.ics.uci.edu/dataset/468/ and run --csv /path/to/file.csv"
            ) from exc
    required = set(CATEGORICAL + [TARGET, "PageValues"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required dataset columns: {sorted(missing)}")
    return df


def prepare_features(df: pd.DataFrame):
    """Return predictors, validated binary labels, numeric/categorical field lists."""
    data = df.copy()
    y_raw = data.pop(TARGET)
    if pd.api.types.is_bool_dtype(y_raw):
        y = y_raw.astype(int)
    elif pd.api.types.is_numeric_dtype(y_raw):
        y = pd.to_numeric(y_raw)
        if y.isna().any() or not set(y.unique()).issubset({0, 1}):
            raise ValueError("Revenue must contain only binary values 0/1.")
        y = y.astype(int)
    else:
        y = y_raw.astype('string').str.strip().str.lower().map(
            {'true': 1, 'false': 0, '1': 1, '0': 0}
        )
        if y.isna().any():
            raise ValueError("Revenue has missing or invalid class labels.")
        y = y.astype(int)

    for col in CATEGORICAL:
        # Keep NA as NA so the imputer (fit within each training fold) can handle it.
        data[col] = data[col].map(lambda v: str(v) if pd.notna(v) else np.nan)
    numerical = [c for c in data.columns if c not in CATEGORICAL]
    for col in numerical:
        data[col] = pd.to_numeric(data[col], errors='raise')
    return data, y, numerical, CATEGORICAL.copy()


def make_preprocessor(numerical: list[str], categorical: list[str]):
    numeric_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])
    categoric_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('numeric', numeric_pipe, numerical),
        ('categorical', categoric_pipe, categorical),
    ])


def model_grid(fast: bool = False):
    if fast:
        return {
            'Decision Tree': (DecisionTreeClassifier(random_state=SEED), {'model__max_depth':[5], 'model__min_samples_leaf':[10]}),
            'KNN': (KNeighborsClassifier(), {'model__n_neighbors':[11]}),
            'Gaussian Naive Bayes': (GaussianNB(), {'model__var_smoothing':[1e-9]}),
        }
    return {
        'Decision Tree': (DecisionTreeClassifier(random_state=SEED), {'model__max_depth':[5,10], 'model__min_samples_leaf':[10,30]}),
        'KNN': (KNeighborsClassifier(), {'model__n_neighbors':[11,31]}),
        'Gaussian Naive Bayes': (GaussianNB(), {'model__var_smoothing':[1e-9,1e-7]}),
    }


def calculate_metrics(name, model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return dict(
        Model=name,
        Accuracy=accuracy_score(y_test,pred),
        Balanced_accuracy=balanced_accuracy_score(y_test,pred),
        Precision=precision_score(y_test,pred,zero_division=0),
        Recall=recall_score(y_test,pred,zero_division=0),
        F1=f1_score(y_test,pred,zero_division=0),
        ROC_AUC=roc_auc_score(y_test,proba),
        Average_precision=average_precision_score(y_test,proba),
    )


def save_figures(models, X_test, y_test, figdir: Path):
    figdir.mkdir(parents=True,exist_ok=True)
    for name, model in models.items():
        fig, ax = plt.subplots(figsize=(5.5,4.5))
        ConfusionMatrixDisplay.from_estimator(
            model,X_test,y_test,ax=ax,cmap='Blues',values_format='d',
            display_labels=['No purchase','Purchase']
        )
        ax.set_title('Confusion matrix: ' + name)
        fig.tight_layout()
        fig.savefig(figdir/f'confusion_{re.sub(r"[^a-z0-9]+","_",name.lower()).strip("_")}.png',dpi=160)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for name,model in models.items():
        RocCurveDisplay.from_estimator(model,X_test,y_test,ax=ax,name=name)
    ax.plot([0,1],[0,1],color='grey',linestyle='--',label='Chance')
    ax.set_title('ROC curves (held-out test data)')
    ax.legend(loc='lower right')
    fig.tight_layout();fig.savefig(figdir/'roc_curves.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for name,model in models.items():
        PrecisionRecallDisplay.from_estimator(model,X_test,y_test,ax=ax,name=name)
    ax.axhline(float(y_test.mean()),color='grey',linestyle='--',label='Class prevalence')
    ax.set_title('Precision–recall curves (held-out test data)')
    ax.legend(loc='best')
    fig.tight_layout();fig.savefig(figdir/'precision_recall_curves.png',dpi=160);plt.close(fig)


def run_experiment(df: pd.DataFrame, output_dir: Path, *, fast=False, n_splits=3):
    """Reproducible split, CV tuning, final evaluation, and PageValues ablation."""
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    X,y,numerical,categorical=prepare_features(df)
    if y.nunique()!=2:
        raise ValueError('Revenue must contain both positive and negative classes.')
    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=.20,random_state=SEED,stratify=y
    )
    print(f'Rows: {len(X)} | Features: {X.shape[1]} | Purchase fraction: {y.mean():.3f}')
    print('Missing predictor values:',int(X.isna().sum().sum()))
    print('Exact duplicate feature rows (not necessarily duplicate events):',int(X.duplicated().sum()))

    cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=SEED)
    fitted={}; cv_records=[]
    for name,(estimator,grid) in model_grid(fast=fast).items():
        print('Training and tuning:',name)
        pipeline=Pipeline([
            ('prep',make_preprocessor(numerical,categorical)),
            ('model',estimator),
        ])
        search=GridSearchCV(
            pipeline,param_grid=grid,cv=cv,scoring='f1',n_jobs=1,refit=True
        )
        search.fit(X_train,y_train)
        fitted[name]=search.best_estimator_
        cv_records.append({
            'Model':name,'CV_F1':search.best_score_,
            'Hyperparameters':str(search.best_params_),
        })
    cv_result=pd.DataFrame(cv_records).sort_values('CV_F1',ascending=False)
    cv_result.to_csv(output_dir/'cross_validation_summary.csv',index=False)

    dummy=DummyClassifier(strategy='most_frequent').fit(X_train,y_train)
    results=pd.DataFrame([
        calculate_metrics(name,model,X_test,y_test)
        for name,model in {'Majority baseline':dummy,**fitted}.items()
    ]).sort_values('F1',ascending=False)
    results.to_csv(output_dir/'test_model_metrics.csv',index=False)
    save_figures(fitted,X_test,y_test,output_dir/'figures')

    # The selection is based only on training-fold cross validation, not test metrics.
    selected=cv_result.iloc[0]['Model']
    full=fitted[selected]
    keep_numeric=[name for name in numerical if name!='PageValues']
    if len(keep_numeric)==len(numerical):
        raise ValueError('PageValues field absent; cannot perform ablation.')
    ablated=Pipeline([
        ('prep',make_preprocessor(keep_numeric,categorical)),
        ('model',clone(full.named_steps['model'])),
    ])
    ablated.fit(X_train.drop(columns='PageValues'),y_train)
    ablation=pd.DataFrame([{
        'Model_selected_by_train_CV':selected,
        'Full_test_F1':f1_score(y_test,full.predict(X_test),zero_division=0),
        'Without_PageValues_test_F1':f1_score(
            y_test,ablated.predict(X_test.drop(columns='PageValues')),zero_division=0
        ),
    }])
    ablation.to_csv(output_dir/'pagevalues_ablation.csv',index=False)
    print('\nHeld-out test scores:\n',results.round(3).to_string(index=False))
    print('\nPageValues ablation:\n',ablation.round(3).to_string(index=False))
    print('\nOutputs:',output_dir.resolve())
    return results, cv_result, ablation


def main():
    parser=argparse.ArgumentParser(description='Evaluate three ML classifiers on UCI shopping sessions.')
    parser.add_argument('--csv',help='Path to downloaded online_shoppers_intention.csv')
    parser.add_argument('--output',default='results',help='Output directory')
    parser.add_argument('--fast',action='store_true',help='Minimal grids for smoke runs; not final analysis')
    args=parser.parse_args()
    df=load_dataset(args.csv)
    run_experiment(df,Path(args.output),fast=args.fast)


if __name__=='__main__':
    main()
