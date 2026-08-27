# Horse Racing Bets & Evals

This is a side project exploring the use of machine learning for horse racing prediction, betting strategy evaluation, and comparison against market-implied probabilities.

## Overview

This repository explores whether machine learning models can predict horse racing outcomes and identify potential betting opportunities beyond the information already reflected in the betting market.

The project focuses on three questions:

1. **Prediction:** How accurately can we predict which horse will win a race?
2. **Market comparison:** How do model predictions compare with market-implied probabilities?
3. **Betting strategy:** Do model predictions translate into useful betting strategies after accounting for the market?

The project combines **feature engineering, probabilistic prediction, model evaluation, calibration, and betting strategy analysis**.

## Data

The data used in this repo is from [Kaggle](https://www.kaggle.com/datasets/gdaley/hkracing) and contains historical horse racing results and race-level information of the Hong Kong racing scene, including:

- Race date and race ID
- Horse, jockey, and trainer identifiers
- Race distance
- Going/ track conditions
- Draw
- Horse weight
- Field size
- Finishing position
- Market odds and implied probabilities
- Win rates

Historical features are constructed using only information available **before the race**, with the aim of avoiding data leakage.

## Approach

### 1. Feature engineering

Race-level and historical features are generated from the raw racing data.

Examples include:

- Prior number of races for a horse
- Prior number of races for a jockey
- Prior number of races for a trainer
- Prior win rates
- Debut indicators, distinguishing horses with no prior racing history from horses with an established record
- Race-level normalisation
- Encoded categorical variables such as going and horse gear

A key consideration is avoiding **data leakage**, i.e. information from a horse's current or future races should not be available when generating its prediction. For example, historical win rates and race counts must be calculated using only races that occurred before the prediction race.

Categorical encoding also needs to be handled carefully. Encoders should be fitted on the training data and then applied consistently to validation and test data. If a category appears in the test set that was not present during training, the model will not have learned how to represent that category, so the encoding strategy needs to define how previously unseen categories are handled.

### 2. Predictive modelling

The repository experiments with several modelling approaches, including:

- Logistic regression
- XGBoost
- Ranking-oriented approaches
- Market-implied probability baselines

The primary prediction target is whether a horse wins its race. Because exactly one horse wins each race, model outputs can also be normalized within each race so that predicted probabilities sum to one.


### 3. Evaluation and market comparison

A central part of the project is comparing model predictions against the betting market.

Decimal odds are converted into implied probabilities:

```
implied probability = 1 / decimal odds
```

Because bookmaker odds generally include a margin (the overround), the implied probabilities can be normalized within each race to produce a market-implied probability distribution.

The difference between the model's estimated probability and the market probability can then be used to measure the model's potential edge:

```
edge = model probability - market probability
```

A positive edge indicates that the model assigns a higher probability of winning than the market does, while a negative edge indicates that the model is less optimistic than the market.



## Evaluation

The models are evaluated using both standard predictive metrics and betting-oriented metrics.

### 1. Predictive Performance

The main metrics include:

- **Log loss (or cross-entropy)** - evaluates the quality of probabilistic predictions by penalising incorrect predictions, with a particularly strong penalty for predictions that are confidently wrong. This makes it useful for assessing whether the model is assigning high probabilities to horses that actually win, rather than simply identifying the most likely winner.
- **Brier score** - measures the mean squared difference between predicted probabilities and the actual outcomes. Unlike top-1 accuracy, it evaluates the full probability estimate, rewarding predictions that are both accurate and appropriately confident.
- **Top-1 accuracy** - measures the proportion of races in which the horse assigned the highest predicted probability by the model actually wins. This provides a simple measure of how often the model correctly identifies the winner, but does not assess the quality or calibration of the underlying probabilities.
- **Calibration** - evaluates whether predicted probabilities correspond to observed win frequencies. For example, among horses predicted to have a 20% chance of winning, approximately 20% should actually win if the model is well calibrated. This is particularly important for the betting component, since identifying a horse as the most likely winner is different from accurately estimating its probability of winning.


### 2. Betting Strategy

Prediction accuracy and betting profitability are treated as separate questions.

A model can outperform another model in terms of predictive metrics without generating profitable bets if the market already incorporates the same information.

The betting analysis therefore evaluates strategies based on different levels of model-vs-market edge, including:

- Which horses are selected for betting
- Number of bets placed
- Win rate
- Average odds
- Total return
- Profit/ loss
- Return on Investment (ROI)
- Performance across different edge thresholds

Where possible, betting strategies are evaluated on held-out data rather than the data used to train the model.

### 3. Calibration

A key diagnostic is a calibration curve comparing:

- mean predicted probability
- vs. observed win rate

Predictions are grouped into probability bins and the average predicted probability is compared with the empirical frequency of winning.

A well-calibrated model should approximately satisfy:

```
predicted probability ≈ observed win frequency
```

This is particularly important because the betting analysis relies on the probabilities themselves, rather than simply ranking horses.

### 4. Validation

Horse racing is a temporal prediction problem, so validation should reflect how the model would actually be used.

Historical features are constructed using only information available before each race. Care is also taken when splitting the data into training and test sets to avoid information from future races influencing predictions for earlier races.

Future iterations of the project will focus on time-based splits and walk-forward validation, which more closely simulate making predictions on future races.

## Results

Results will be added as the modelling and evaluation pipeline develops. The notebooks contain the exploratory analysis and modelling workflows used to generate these results.

## Project structure

```
horse-racing-bets-and-evals/
│
├── data/
│   └── ...
├── notebooks/
│   └── ...
├── src/
│   └── ...
├── models/
│   ├── tests/
|   └── ...
├── results/
│   └── ...
├── requirements.txt
└── README.md
```

## Reproducing the Analysis

The analysis is implemented in Python. The notebooks contain the exploratory data analysis, feature engineering, model training, and evaluation workflows.

To set up the environment:

```bash
pip install -r requirements.txt
```

The main modelling pipeline can be run from the repository root with:

```bash
python -m scripts.run_pipeline
```


## Current models

| Model | Purpose |
|---|---|
| Logistic Regression | Simple probabilistic baseline |
| XGBoost | Non-linear tree-based model |
| Market Probabilities | Benchmark against the existing betting market |

## Key questions

Some of the questions I am interested in exploring through this project are:

- How much predictive signal can be extracted from historical horse, jockey, and trainer performance?
- How much of this information is already reflected in market odds?
- Can ML models produce better-calibrated probabilities than market-implied probabilities?
- Does a positive model-vs-market edge translate into improved betting returns?
- Does model performance generalise to genuinely unseen races?

## Limitations

There are several important limitations to this analysis:

- Horse racing outcomes are inherently noisy and difficult to predict.
- As with any time-seris forecasting taks, historical performance does not guarantee future performance.
- The betting market provides a strong baseline and may already incorporate much of the available information.
- Apparent betting profits can result from statistical noise or overfitting.
- Results can be sensitive to how the train/test split is constructed. A safer approach is to further split the training set into training and validation (e.g. in a 80:20 ratio, making sure data doesn't leak from the validation set).
- Real-world betting involves factors such as changing odds, bookmaker limits, liquidity, and transaction costs.
- Repeated experimentation and model selection can lead to overfitting to the evaluation dataset.

For these reasons, any apparent profitability should be interpreted cautiously and evaluated on genuinely out-of-sample data.

## Future work

Potential extensions include:

- [ ] Hierarchical horse / jockey / trainer effects, and investigating whether reducing the number of feature classes improves generalisation
- [ ] Investigate whether reducing the number of feature classes improves generalisation
- [ ] Walk-forward validation
- [ ] Additional gradient-boosting models (e.g. LightGBM, CatBoost)
- [ ] Bootstrap confidence intervals for betting returns
- [ ] Test whether model edge persists after accounting for the bookmaker's overround
- [ ] Evaluate whether model performance is robust across different time periods

## Long-term work

A few more exploratory, higher-effort directions beyond the near-term items above include:

- Using an LLM to clean and standardize feature categories. Several categorical fields in this dataset are inconsistent in ways that are hard to fix with fixed lookup rules:
  - Geographic fields mix granularity — some entries are countries, others are specific cities or local venues, which currently need to be manually aggregated to be usable as a single consistent feature.
  - Fields like horse_gear and horse_type have inconsistent or overloaded codes (e.g. coat colours appearing where a sex value is expected — see the horse_sex handling in features.py). An LLM could help propose a canonical taxonomy for these fields, flag ambiguous or contradictory entries, and suggest a consistent mapping, rather than relying on a hand-maintained rule set that has to be manually extended every time a new inconsistent value shows up.

- Using VLMs to extract additional features from race footage. Beyond the tabular data, race video could be a source of features that aren't otherwise captured - early pace position, visual signs of a horse traveling well or struggling, track/ weather conditions at race time, gate behavior, etc. This is a more speculative, data-hungry direction than the tabular work above, but could meaningfully expand the feature set beyond what's available in structured form.


## Disclaimer

This is an independent project for educational purposes.

The code is written by me, with GenAI used for syntax assistance, troubleshooting, and some semantic normalisation tasks.

Historical model performance and simulated betting returns are not indicative of future results. Nothing in this repository constitutes financial or betting advice.
