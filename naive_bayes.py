# Para compilar
# spark-submit --master local[4] ../examples/src/main/python/mllib/naive_bayes_example.py
# data/mllib/sample_libsvm_data.txt

from __future__ import print_function
import shutil
from pyspark import SparkContext
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils

if __name__ == "__main__":

    sc = SparkContext(appName="PythonNaiveBayesExample")
    # Cargar dataset
    data = MLUtils.loadLibSVMFile(sc, "data/mllib/dataset.txt")
    # Separar data
    training, test = data.randomSplit([0.6, 0.4])

    # Entrenar a naive Bayes.
    model = NaiveBayes.train(training, 1.0)

    # Hacer la prediccion and test accuracy.
    predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
    print('model accuracy {}'.format(accuracy))

    # Guardar y cargar el modelo
    output_dir = 'target/tmp/myNaiveBayesModel'
    shutil.rmtree(output_dir, ignore_errors=True)
    model.save(sc, output_dir)
    sameModel = NaiveBayesModel.load(sc, output_dir)
    predictionAndLabel = test.map(lambda p: (sameModel.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
    print('sameModel accuracy {}'.format(accuracy))
