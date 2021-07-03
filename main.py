# from sklearn import datasets
# from sklearn import svm

import pickle

from predictor_precios import PredictorPrecios

if __name__ == "__main__":
    # Load training data
    # iris = datasets.load_iris()
    # X, y = iris.data, iris.target

    # Model Training
    # clf = svm.SVC(gamma='scale')
    # clf.fit(X, y)

    # Create a iris classifier service instance
    predictor_service = PredictorPrecios()

    # Pack the newly trained model artifact
    loaded_model = pickle.load(open('modelo_pricing_2.pkl', 'rb'))
    predictor_service.pack('model', loaded_model)

    # Save the prediction service to disk for model serving
    saved_path = predictor_service.save()


