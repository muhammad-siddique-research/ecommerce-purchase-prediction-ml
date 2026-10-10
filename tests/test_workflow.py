"""Tests use *synthetic fixture data only*; never treat these scores as study findings."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.run_experiment import prepare_features, make_preprocessor, run_experiment


def fake_data(n=120, seed=12):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        'Administrative':rng.integers(0,8,n),
        'Administrative_Duration':rng.exponential(100,n),
        'Informational':rng.integers(0,5,n),
        'Informational_Duration':rng.exponential(20,n),
        'ProductRelated':rng.integers(1,60,n),
        'ProductRelated_Duration':rng.exponential(280,n),
        'BounceRates':rng.uniform(0,.2,n),
        'ExitRates':rng.uniform(0,.2,n),
        'PageValues':rng.exponential(8,n),
        'SpecialDay':rng.choice([0,.2,.6],n),
        'Month':rng.choice(['Feb','May','Nov'],n),
        'OperatingSystems':rng.integers(1,5,n),
        'Browser':rng.integers(1,6,n),
        'Region':rng.integers(1,4,n),
        'TrafficType':rng.integers(1,4,n),
        'VisitorType':rng.choice(['New_Visitor','Returning_Visitor'],n),
        'Weekend':rng.choice([True,False],n),
    })
    p=1/(1+np.exp(-(df['PageValues']/10 + df['ProductRelated']/18 - 3)))
    df['Revenue']=rng.binomial(1,p).astype(bool)
    return df


def test_preprocess_shape_and_target_validation():
    df=fake_data()
    X,y,num,cat=prepare_features(df)
    assert X.shape==(120,17)
    assert set(y.unique())=={0,1}
    assert 'Revenue' not in X.columns
    assert 'PageValues' in num
    assert 'Month' in cat
    assert len(make_preprocessor(num,cat).transformers)==2


def test_bad_target_rejected():
    df=fake_data()
    df['Revenue']=df['Revenue'].astype(object)
    df.loc[0,'Revenue']='unknown'
    with pytest.raises(ValueError):
        prepare_features(df)


def test_end_to_end_smoke(tmp_path:Path):
    df=fake_data(n=160)
    metrics,cv,ablation=run_experiment(df,tmp_path,fast=True,n_splits=2)
    assert set(metrics['Model'])=={'Majority baseline','Decision Tree','KNN','Gaussian Naive Bayes'}
    assert metrics['Accuracy'].between(0,1).all()
    assert len(cv)==3 and len(ablation)==1
    assert (tmp_path/'test_model_metrics.csv').exists()
    assert (tmp_path/'cross_validation_summary.csv').exists()
    assert (tmp_path/'pagevalues_ablation.csv').exists()
    assert len(list((tmp_path/'figures').glob('*.png')))==5
