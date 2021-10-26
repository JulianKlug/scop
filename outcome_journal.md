Quick and dirty keras implementation to have a functioning baselines.
Scores obtained on first try on test data: 
{'loss': 0.0, 'acc': 0.800000011920929, 'auc': 0.8366667032241821}
with basic densenet and 
with config:
-    channels = [0, 1, 2, 3]
-    outcome = "combined_mRS_0-2_90_days"
-    desired_shape = (46, 46, 46)
-    split_ratio = 0.3
-   batch_size = 2
-   epochs = 200
-  initial_learning_rate = 0.0001
