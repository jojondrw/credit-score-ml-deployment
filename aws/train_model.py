from abc import ABC, abstractmethod
from sklearn.ensemble import (RandomForestClassifier,
                              HistGradientBoostingClassifier,
                              ExtraTreesClassifier)


# Parent
class BaseModel(ABC):

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


# Child 1: Random Forest (model terbaik di notebook)
class RandomForestModel(BaseModel):
    def __init__(self):
        self.name = "Random_Forest"
        self.model = RandomForestClassifier(class_weight='balanced', random_state=42)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# Child 2: Hist Gradient Boosting
class HistGradientBoostingModel(BaseModel):
    def __init__(self):
        self.name = "Hist_Gradient_Boosting"
        self.model = HistGradientBoostingClassifier(random_state=42)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# Child 3: Extra Trees
class ExtraTreesModel(BaseModel):
    def __init__(self):
        self.name = "Extra_Trees"
        self.model = ExtraTreesClassifier(class_weight='balanced', random_state=42)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
