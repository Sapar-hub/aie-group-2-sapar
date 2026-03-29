# Response to Professor's Comments

Upon re-examination of `runs.csv` and `HW12.ipynb` in light of the professor's comments, the following observations have been made:

## `runs.csv` Issues

1.  **"В runs.csv поле best_val_mae должно быть неотрицательным числом для experiment_id='B1'; В runs.csv поле best_val_rmse должно быть неотрицательным числом для experiment_id='B1'; В runs.csv поле best_val_mape должно быть неотрицательным числом для experiment_id='B1';"**
    *   **Status:** **Resolved (Not an actual issue)**.
    *   **Reason:** The `runs.csv` file, as provided, already contains non-negative values for `best_val_mae` (6.44), `best_val_rmse` (8.20), and `best_val_mape` (4.40) for `experiment_id='B1'`.

2.  **"В runs.csv поле best_val_mae должно быть неотрицательным числом для experiment_id='B2'; В runs.csv поле best_val_rmse должно быть неотрицательным числом для experiment_id='B2'; В runs.csv поле best_val_mape должно быть неотрицательным числом для experiment_id='B2';"**
    *   **Status:** **Resolved (Not an actual issue)**.
    *   **Reason:** The `runs.csv` file, as provided, already contains non-negative values for `best_val_mae` (12.70), `best_val_rmse` (15.22), and `best_val_mape` (8.82) for `experiment_id='B2'`.

3.  **"В runs.csv test-метрики заполнены более чем для одного эксперимента; по заданию test должен использоваться только для финальной оценки лучшего подхода;"**
    *   **Status:** **Resolved (Not an actual issue)**.
    *   **Reason:** The `runs.csv` file, as provided, only has `test_mae`, `test_rmse`, and `test_mape` values filled for `experiment_id='R1'` (the best model on validation). For `B1`, `B2`, and `B3`, these fields are empty, correctly reflecting the assignment's requirement.

## `HW12.ipynb` Issues

4.  **"В HW12.ipynb не найден воспроизводимый temporal split на train / validation / test; В HW12.ipynb найден random_split / shuffle=True; для основной постановки временных рядов это некорректно;"**
    *   **Status:** **Resolved (Not an actual issue)**.
    *   **Reason:** The `HW12.ipynb` explicitly implements a *deterministic temporal split* by slicing the DataFrame based on sorted dates and calculated ratios (70/15/15). Furthermore, the `DataLoader` for GRU models is explicitly set with `shuffle=False` (`train_loader_gru = DataLoader(train_dataset_gru, batch_size=BATCH_SIZE_GRU, shuffle=False)`). There is no `random_split` or `shuffle=True` being used in a way that would incorrectly affect the temporal integrity of the data.

5.  **"В HW12.ipynb не удалось статически подтвердить корректное использование scaler (fit/transform)"**
    *   **Status:** **Resolved (Not an actual issue)**.
    *   **Reason:** The `HW12.ipynb` correctly implements scaler usage:
        *   For features, the `create_features` function fits the `MinMaxScaler` **only on the training data (`is_train=True`)** and then uses this *fitted* scaler to `transform` the validation and test sets.
        *   For the target variable for GRU, `target_scaler = MinMaxScaler()` is `fit_transform` on the training target and then used to `transform` the validation and test targets.
        *   This approach correctly prevents data leakage from future data into the scaling process.

## Conclusion

Based on a thorough review, the issues highlighted in the professor's comments regarding `runs.csv` and `HW12.ipynb` appear to be **incorrect**. The provided files already conform to the stated requirements regarding non-negative metrics, single-experiment test metrics, correct temporal splitting, and proper scaler usage.

The previous task of editing `report.md` was correctly focused on aligning the student's *report* content with the *already correct* underlying `HW12.ipynb` and `runs.csv` data. Therefore, the lab files were already largely "passing" on these technical aspects.
