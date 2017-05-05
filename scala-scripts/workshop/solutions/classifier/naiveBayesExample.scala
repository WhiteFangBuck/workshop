import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val rawText = spark.read.format("csv")
  .option("delimiter", "\t")
  .load("data/SMSSpamCollection.txt")
  .toDF("id", "text")

val tokenized = new RegexTokenizer()
  .setInputCol("text")
  .setOutputCol("words")
  .setPattern("\\W")
  .transform(rawText)

val indexed = new StringIndexer()
  .setInputCol("id")
  .setOutputCol("label")
  .fit(tokenized)
  .transform(tokenized)

val hashed = new HashingTF()
  .setInputCol("words")
  .setOutputCol("rawFeatures")
  .setNumFeatures(1000)
  .transform(indexed)

val tfidf = new IDF()
  .setInputCol("rawFeatures")
  .setOutputCol("features")
  .fit(hashed)
  .transform(hashed)

val Array(train, test) = tfidf.randomSplit(Array(.8, .2), 102059L)

val model = new NaiveBayes()
  .fit(train)

val metrics = new BinaryClassificationEvaluator()
  .setLabelCol("label")
  .setRawPredictionCol("prediction")

metrics.evaluate(predictions)
