def calculate_f1(y_gold, y_model):
    """
    TODO: MODIFY 
    
    Computes the F1 of the model predictions using the
    gold labels. Each of y_gold and y_model are lists with
    labels 1 or -1. The function should return the F1
    score as a number between 0 and 1.
    """
    import numpy as np
#     diff = y_gold - y_model
    diff = np.zeros(len(y_gold))
    for i in range(len(y_gold)):
        diff[i] = y_gold[i] - y_model[i]
    TP = 0
    FP = 0
    FN = 0
    
    for i in range(len(diff)):
        if diff[i] == 0 and y_gold[i] == 1:
            TP += 1
        if diff[i] > 0:
            FN += 1
        if diff[i] < 0:
            FP += 1
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    
    return F1
    

def highest_and_lowest_f1_score():
    """
    TODO: MODIFY
    
    Return the highest and lowest possible F1 score
    (ie one line solution returning the theoretical max and min)
    """
    
    maxscore = 1
    minscore = 0
    
    return maxscore, minscore



class Classifier(object):
    """
    DO NOT MODIFY

    The Classifier class is the base class for all of the Perceptron-based
    algorithms. Your class should override the "process_example" and
    "predict_single" functions. Further, the averaged models should
    override the "finalize" method, where the final parameter values
    should be calculated. 
    
    You should not need to edit this class any further.
    """
    
    ITERATIONS = 10
    
    def train(self, X, y):
        for iteration in range(self.ITERATIONS):
            for x_i, y_i in zip(X, y):
                self.process_example(x_i, y_i)
        self.finalize()

    def process_example(self, x, y):
        """
        Makes a prediction using the current parameter values for
        the features x and potentially updates the parameters based
        on the gradient. Note "x" is a dictionary which maps from the feature
        name to the feature value and y is either 1 or -1.
        """
        raise NotImplementedError

    def finalize(self):
        """Calculates the final parameter values for the averaged models."""
        pass

    def predict(self, X):
        """
        Predicts labels for all of the input examples. You should not need
        to override this method.
        """
        return [self.predict_single(x) for x in X]

    def predict_single(self, x):
        """
        Predicts a label, 1 or -1, for the input example. "x" is a dictionary
        which maps from the feature name to the feature value.
        """
        raise NotImplementedError



class Perceptron(Classifier):
    """
    DO NOT MODIFY THIS CELL

    The Perceptron model. Note how we are subclassing `Classifier`.
    """
    
    def __init__(self, features):
        """
        Initializes the parameters for the Perceptron model. "features"
        is a list of all of the features of the model where each is
        represented by a string.
        """
        
        # NOTE: Do not change the names of these 3 data members because
        # they are used in the unit tests
        self.eta = 1
        self.theta = 0
        self.w = {feature: 0.0 for feature in features}

    def process_example(self, x, y):
        y_pred = self.predict_single(x)
        if y != y_pred:
            for feature, value in x.items():
                self.w[feature] += self.eta * y * value
            self.theta += self.eta * y

    def predict_single(self, x):
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        if score <= 0:
            return -1
        return 1


class Winnow(Classifier):
    def __init__(self, alpha, features):
        # DO NOT change the names of these 3 data members because
        # they are used in the unit tests
        self.alpha = alpha
        self.w = {feature: 1.0 for feature in features}
        self.theta = -len(features)
        
    def process_example(self, x, y):
        """ TODO: IMPLEMENT"""
        
        y_pred = self.predict_single(x)
        if y != y_pred:
            for feature, value in x.items():
                self.w[feature] = self.w[feature] * pow(self.alpha,y*value)
        

    def predict_single(self, x):
        """ TODO: IMPLEMENT"""
        
        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        
        if score <= 0:
            return -1
        return 1





class AdaGrad(Classifier):
    def __init__(self, eta, features):
        # DO NOT change the names of these 3 data members because
        # they are used in the unit tests
        self.eta = eta
        self.w = {feature: 0.0 for feature in features}
        self.theta = 0
        self.G = {feature: 1e-5 for feature in features}  # 1e-5 prevents divide by 0 problems
        self.H = 0
        
    def process_example(self, x, y):
        """ TODO: IMPLEMENT"""
        import numpy as np
        
        y_pred = self.predict_single(x)
        dotpro = 0
        
        if y != y_pred:
            
#             calculate dot product
            for feature, value in x.items():
                dotpro += self.w[feature] * value
                
#             update w
            for feature, value in x.items():
                if(y * (dotpro + self.theta) > 1):
                    dldw = 0                   
                else:
                    dldw = -y * value
                self.G[feature] += dldw ** 2
                self.w[feature] += self.eta * y * value / np.sqrt(self.G[feature])
                
#             update theta        
            if(y * (dotpro + self.theta) > 1):
                dldtheta = 0
            else:
                dldtheta = -y
            self.H += dldtheta ** 2
            self.theta += self.eta * y / np.sqrt(self.H)  
        

    def predict_single(self, x):
        """ TODO: IMPLEMENT"""

        score = 0
        for feature, value in x.items():
            score += self.w[feature] * value
        score += self.theta
        
        if score <= 0:
            return -1
        return 1
        
        
        




