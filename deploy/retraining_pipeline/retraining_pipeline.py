class Pipeline:
    def __init__(self):
        self.data_loader = NumpyDataLoader()
        self.feature_calculator = MFCCs()
        self.pre_processor = TrainTestSplit()
        self.trainer = SklearnTrainer()

    def run(self):
        data, labels = self.data_loader.run()
        features = self.feature_calculator.run(data)
        X_train, X_test, y_train, y_test = self.pre_processor.run(features, labels)
        new_model = self.trainer.run(X_train, X_test, y_train, y_test)
        new_predictions = self.model_runner(X_test, new_model)
        old_predictions = self.model_runner(X_test, old_model)

        is_better = self.model_comparison(y_test, old_predictions, new_predictions)

        if is_better:
            self.save_model(new_model)

        # Compare old_new
        # Save model if better


class NumpyDataLoader:
    def __init__(self, data_path='/app/data'):
        self.path = path

    def run(self):
        ambient_path = os.path.join(self.path, 'ambient_data.npy')
        bell_path = os.path.join(self.path, 'bell_data.npy')
        ambient_data = np.load(open(ambient_path, 'rb'))
        bell_data = np.load(open(bell_path, 'rb'))

        ambient_labeled = 

        features = np.vstack([bell_data, ambient_data])
        labels = np.vstack([1]*len(bell_data) + [0]*len(ambient_data))

        return features, labels


class MFCCs:
    def __init__(self, sr=22050):
        self.sr = sr

    def run(self, data):
        mfccs = np.array([np.mean(librosa.feature.mfcc(y=librosa.util.normalize(entry), sr=sr, n_mfcc=13).T, axis=0)
                          for entry in data])
        return mfccs


class TrainTestSplit:
    def __init__(self, test_split=0.2, random_state=42):
        self.train_split = train_split
        self.random_state = random_state

    def run(self, features, labels):
        X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                            test_size=self.train_split,
                                                            random_state=self.random_state)
        return X_train, X_test, y_train, y_test


class SklearnTrainer:
    def __init__(self):
        self.param_grid = [
            {'kernel': ['rbf'],
            'gamma': [1e-3, 1e-4],
            'C': [1, 10, 100, 1000]},
            {'kernel': ['linear'],
            'C': [1, 10, 100, 1000]}
        ]

    def _algorithm_pipeline(self, X_train_data, X_test_data, y_train_data, y_test_data,
                            model, param_grid, cv=5, scoring_fit='accuracy'):
        gs = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1,
            scoring=scoring_fit,
            verbose=3
        )
        fitted_model = gs.fit(X_train_data, y_train_data)

        return fitted_model

    def run(self, X_train, X_test, y_train, y_test):
        model = SVC()
        model = _algorithm_pipeline(X_train, X_test, y_train, y_test, model,
                                    self.param_grid, cv=5, scoring_fit='accuracy')

        print(model.best_score_)
        print(model.best_params_)
        return model, new_predictions

class SVCRunner:
    def __init__(self):
        pass

    def run(self, test_data, model, do_probabilities=False):
        if do_probabilities:
            pred = model.predict_proba(test_data)
        else:
            pred = model.predict(test_data)

        return pred


class BasicComparison:
    def __init__(self):
        pass

    def run(y_true, old_predictions, new_predictions):
        old_auc = roc_auc_score(y_true, old_predictions)
        new_auc = roc_auc_score(y_true, new_predictions)

        old_cm = confusion_matrix(y_true, old_predictions)
        new_cm = confusion_matrix(y_true, new_predictions)

