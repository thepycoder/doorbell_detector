from datetime import datetime
import pickle
import os

import librosa
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

from config import ORIGINAL_DATA_PATH, LABELED_DATA_PATH, MODEL_PATH


class Pipeline:
    def __init__(self):
        self.data_loader = NumpyDataLoader()
        self.feature_calculator = MFCCs()
        self.pre_processor = TrainTestSplit()
        self.trainer = SklearnTrainer()
        self.model_runner = SVCRunner()
        self.model_loader = ModelLoader()
        self.model_comparison = BasicComparison()
        self.model_saver = ModelSaver()

    def run(self):
        print('Loading data')
        data, labels = self.data_loader.run()
        print('Calculating features')
        features = self.feature_calculator.run(data)
        print('Preprocessing')
        X_train, X_test, y_train, y_test = self.pre_processor.run(features, labels)
        print('Training new model')
        new_model = self.trainer.run(X_train, X_test, y_train, y_test)
        print('Loading old model')
        old_model = self.model_loader.run()
        
        if not old_model:
            is_better = True
        else:
            print('Benchmarking both models')
            new_predictions = self.model_runner.run(X_test, new_model)
            old_predictions = self.model_runner.run(X_test, old_model)

            is_better = self.model_comparison.run(y_test, old_predictions, new_predictions)

        print('Saving new model')
        model_name = self.model_saver.run(new_model)

        print(f'New model is better: {is_better}')
        if is_better:
            print('Symlinking new model')
            old_cwd = os.getcwd()
            os.chdir(MODEL_PATH)
            latest_path = 'latest.p'
            if os.path.lexists(latest_path):
                os.unlink(latest_path)
            os.symlink(model_name, latest_path)
            os.chdir(old_cwd)

        print('Done!')


class NumpyDataLoader:
    def __init__(self, original_data_path=ORIGINAL_DATA_PATH, labeled_data_path=LABELED_DATA_PATH):
        self.original_data_path = original_data_path
        self.labeled_data_path = labeled_data_path

    def run(self):
        ambient_path = os.path.join(self.original_data_path, 'ambient_data.npy')
        bell_path = os.path.join(self.original_data_path, 'bell_data.npy')
        ambient_data = np.load(open(ambient_path, 'rb'))
        bell_data = np.load(open(bell_path, 'rb'))

        ambient_labeled = self.load_wav(os.path.join(self.labeled_data_path, 'ambient'))
        bell_labeled = self.load_wav(os.path.join(self.labeled_data_path, 'bell'))

        features = np.vstack([bell_data, bell_labeled, ambient_data, ambient_labeled])
        labels = np.ravel(np.vstack([1]*(len(bell_data) + len(bell_labeled)) + [0]*(len(ambient_data) + len(ambient_labeled))))

        return features, labels
    
    def load_wav(self, folder):
        clips = []
        for clip_name in os.listdir(folder):
            # TODO: implement check on SR
            x , _ = librosa.load(os.path.join(folder, clip_name))
            clips.append(x)
        return np.array(clips)


class MFCCs:
    def __init__(self, sr=22050):
        self.sr = sr

    def run(self, data):
        mfccs = np.array([np.mean(librosa.feature.mfcc(y=librosa.util.normalize(entry), sr=self.sr, n_mfcc=13).T, axis=0)
                          for entry in tqdm(data)])
        return mfccs


class TrainTestSplit:
    def __init__(self, test_split=0.2, random_state=42):
        self.test_split = test_split
        self.random_state = random_state

    def run(self, features, labels):
        X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                            test_size=self.test_split,
                                                            random_state=self.random_state)
        return X_train, X_test, y_train, y_test


class SklearnTrainer:
    def __init__(self):
        self.param_grid = [
            {'kernel': ['rbf'],
            'gamma': [1e-3, 1e-4],
            'C': [1, 10, 100, 1000]},
            # {'kernel': ['linear'],
            # 'C': [1, 10, 100, 1000]}
        ]

    def _algorithm_pipeline(self, X_train_data, X_test_data, y_train_data, y_test_data,
                            model, param_grid, cv=5, scoring_fit='accuracy'):
        gs = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            n_jobs=1,
            scoring=scoring_fit,
            verbose=3
        )
        fitted_model = gs.fit(X_train_data, y_train_data)

        return fitted_model

    def run(self, X_train, X_test, y_train, y_test):
        model = SVC()
        model = self._algorithm_pipeline(X_train, X_test, y_train, y_test, model,
                                    self.param_grid, cv=5, scoring_fit='accuracy')

        print(model.best_score_)
        print(model.best_params_)
        return model

class ModelLoader:
    def __init__(self, path=MODEL_PATH):
        self.path = path

    def run(self):
        try:
            model = pickle.load(open(os.path.join(self.path, 'latest.p'), 'rb'))
        except FileNotFoundError:
            model = False
        
        return model

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

    def run(self, y_true, old_predictions, new_predictions):
        old_auc = roc_auc_score(y_true, old_predictions)
        new_auc = roc_auc_score(y_true, new_predictions)

        old_cm = confusion_matrix(y_true, old_predictions)
        new_cm = confusion_matrix(y_true, new_predictions)

        print(f'Old AUC: {old_auc}')
        print('Old CM')
        print(old_cm)
        print(f'New AUC: {new_auc}')
        print('New CM')
        print(new_cm)

        return new_auc > old_auc


class ModelSaver:
    def __init__(self, dt_format="%Y-%m-%d_%H:%M:%S", model_path=MODEL_PATH):
        self.dt_format = dt_format
        self.model_path = model_path
    
    def run(self, fitted_model):
        datetimestr = datetime.now().strftime(self.dt_format)
        model_name = f'{datetimestr}.p'
        pickle.dump(fitted_model, open(os.path.join(self.model_path, model_name), 'wb'))

        return model_name


def run_pipeline():
    pipeline = Pipeline()
    pipeline.run()


if __name__ == '__main__':
    run_pipeline()
