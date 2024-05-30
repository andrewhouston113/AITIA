# AITIA: A Model Interrogation Library for Improving the Robustness and Transparency of AI models
AITIA offers user-friendly functions for determining the competencies of classification algorithms from a complexity perspective, 
assessing concept drift robustness, understanding what causes a model to make errors, developing models with abstention capabilities based on uncertainty, 
and providing layperson-friendly explanations.

The functions of AITIA can be accessed from interrogation_analysis.py. Examples of AITIA's functionality is presented below.

## Competency Analysis
This method evaluates the competency of a given machine learning model across the generated synthetic datasets using cross-validation.

```
analysis = CompetencyAnalysis(n_datasets=100, max_distance=0.4, pop_size=40, n_gen=10)
analysis.prepare_analysis(X_train, y_train)
analysis.evaluate_competency(model, cv=5, scoring='roc_auc')
analysis.visualise_results()
```

## Concept Drift Analysis
This method evaluates a given machine learning model's performance on $X$ and $y$, using $k$-fold cross validation, incrementally introducing more synthetic data to emulate concept drift.

```
analysis = ConceptDriftAnalysis(X_train, y_train, f1_target=0.8, n1_target=0.9, pop_size=50, n_gen=15)
results = analysis.evaluate_concept_drift(model, n_splits=5, scoring='roc_auc')
```

## Uncertainty-driven Abstention
This method provides a means of determining the optimal abstention threshold to meet the performance requirements of end users, whilst maintaining the applicability of the model to the maximum number of instances.

```
uncertainty_estimator = UncertaintyEstimator(model)
abstention_threshold = derive_abstention_threshold(X_train, y_train, uncertainty_estimator)
```

## Uncertainty Explanations
This method provides a means of explaining model uncertainty using SHAP values at the instance and dataset level, in terms of actual features and meta-features.

```
explainer = UncertaintyExplainer(uncertainty_system=my_uncertainty_model, M=100)
explainer.explain(X_train, x=X_train[:10,:], feature_names=['Feature1', 'Feature2'], level='meta')
```

## Misclassification Analysis
This method explains misclassifications using heuristic scores and logistic regression. 
It makes predictions using the provided model, identifies misclassifications, calculates heuristic scores, standardises them using z-score normalisation, 
and fits a logistic regression model to understand the associations between each heuristic and the misclassifications events. 
It also computes AUROC and Area Under the AUPRC scores for each heuristic.

```
explainer = MisclassificationExplainer(n_neighbors=10, max_depth=4, balanced=False)
explainer.fit(X_train, y_train, categorical_idx=[2, 5, 7])
explainer.explain(model, X_test, y_test)
explainer.plot_results()
```

## Requirements
```
deap==1.4.1
gower==0.1.2
matplotlib==3.9.0
numpy==1.26.4
pandas==2.2.2
pymop==0.2.4
scikit_fuzzy==0.4.2
scikit_learn==1.2.2
scikit_optimize==0.10.1
scipy==1.13.1
seaborn==0.13.2
statsmodels==0.14.2
tqdm==4.66.4
zepid==0.9.1
```
