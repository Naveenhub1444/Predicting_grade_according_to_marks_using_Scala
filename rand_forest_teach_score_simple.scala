import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DecimalType

import java.util.Calendar

object rand_forest_teach_score_simple {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val dataspath = "D:\\data_files\\scores.csv"
    var start_Time = Calendar.getInstance()
    var start_date_time = start_Time.getTime()
    println("Starting timeand date =" + start_date_time)


    val marksDF = spark.read.option("header", "true")
      .option("inferSchema", true)
      .option("mode", "DROPMALFORMED")
      .csv(dataspath)

    marksDF.printSchema()

  /*
root
 |-- score1: double (nullable = true)
 |-- score2: double (nullable = true)
 |-- result: integer (nullable = true)

   */
    marksDF.show(5,false)
/*
+-----------------+-----------------+------+
|score1           |score2           |result|
+-----------------+-----------------+------+
|34.62365962451697|78.0246928153624 |0     |
|30.28671076822607|43.89499752400101|0     |
|35.84740876993872|72.90219802708364|0     |
|60.18259938620976|86.30855209546826|1     |
|79.0327360507101 |75.3443764369103 |1     |
+-----------------+-----------------+------+
 */

    val marksDf_new= marksDF
      .withColumn("score1", col ( "score1").cast(DecimalType(6,2)))
      .withColumn("score2", col ( "score2").cast(DecimalType(6,2)))

    marksDf_new.show(25,false)

/*
+------+------+------+
|score1|score2|result|
+------+------+------+
|34.62 |78.02 |0     |
|30.29 |43.89 |0     |
|35.85 |72.90 |0     |
|60.18 |86.31 |1     |
|79.03 |75.34 |1     |
+------+------+------+
 */

    val labelIndexer = new StringIndexer()
      .setInputCol("result")
      .setOutputCol("label")
      .fit(marksDf_new)


    val assembler = new VectorAssembler()
      .setInputCols(Array("score1","score2"))
      .setOutputCol("features")
      .setHandleInvalid("skip")

    val featureced_Df = assembler.transform(marksDf_new)

    featureced_Df.printSchema()

/*
root
 |-- score1: decimal(6,2) (nullable = true)
 |-- score2: decimal(6,2) (nullable = true)
 |-- result: integer (nullable = true)
 |-- features: vector (nullable = true)
 */

    val labeled_DF = labelIndexer.transform(featureced_Df)

    labeled_DF.printSchema()

/*
    root
 |-- score1: decimal(6,2) (nullable = true)
 |-- score2: decimal(6,2) (nullable = true)
 |-- result: integer (nullable = true)
 |-- features: vector (nullable = true)
 |-- label: double (nullable = false)
 */

    labeled_DF.show(5,false)

/*
    +------+------+------+-------------+-----+
|score1|score2|result|features     |label|
+------+------+------+-------------+-----+
|34.62 |78.02 |0     |[34.62,78.02]|1.0  |
|30.29 |43.89 |0     |[30.29,43.89]|1.0  |
|35.85 |72.90 |0     |[35.85,72.9] |1.0  |
|60.18 |86.31 |1     |[60.18,86.31]|0.0  |
|79.03 |75.34 |1     |[79.03,75.34]|0.0  |
+------+------+------+-------------+-----+
 */

    val seed = 5043
    val Array(trainData,testData)=labeled_DF.randomSplit(Array(0.7,0.3),seed)

    val classifier = new RandomForestClassifier()

    /*
    Random Forests allow us to look at feature importances,
     which is the how much the Gini Index for a feature decreases at each split.
      The more the Gini Index decreases for a feature, the more important it is.
       The figure below rates the features from 0–100, with 100 being the most important.
     */
      .setImpurity("gini")
      .setMaxDepth(10)
      .setNumTrees(20)
    .setFeatureSubsetStrategy("auto")
      .setSeed(5043)

/*
    val classifier = new RandomForestClassifier()
      .setImpurity("gini")
      .setMaxDepth(3)
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(20)

 */
      val model = classifier.fit(trainData)

     val prediction_df = model.transform(testData)

    prediction_df.show(10,false)

/*
  +------+------+------+-------------+-----+----------------------------------------+-----------------------------------------+----------+
|score1|score2|result|features     |label|rawPrediction                           |probability                              |prediction|
+------+------+------+-------------+-----+----------------------------------------+-----------------------------------------+----------+
|30.29 |43.89 |0     |[30.29,43.89]|1.0  |[0.42411884411884415,19.575881155881156]|[0.021205942205942206,0.9787940577940578]|1.0       |
|33.92 |98.87 |0     |[33.92,98.87]|1.0  |[8.0,12.0]                              |[0.4,0.6]                                |1.0       |
|34.52 |60.40 |0     |[34.52,60.4] |1.0  |[0.42411884411884415,19.575881155881156]|[0.021205942205942206,0.9787940577940578]|1.0       |
|35.85 |72.90 |0     |[35.85,72.9] |1.0  |[0.42411884411884415,19.575881155881156]|[0.021205942205942206,0.9787940577940578]|1.0       |
|40.46 |97.54 |1     |[40.46,97.54]|0.0  |[9.0,11.0]                              |[0.45,0.55]                              |1.0       |
|42.26 |87.10 |1     |[42.26,87.1] |0.0  |[13.796296296296296,6.203703703703704]  |[0.6898148148148148,0.31018518518518523] |0.0       |
|44.67 |66.45 |0     |[44.67,66.45]|1.0  |[4.22041514041514,15.779584859584858]   |[0.211020757020757,0.7889792429792429]   |1.0       |
|45.08 |56.32 |0     |[45.08,56.32]|1.0  |[3.2204151404151404,16.77958485958486]  |[0.16102075702075702,0.8389792429792429] |1.0       |
|49.07 |51.88 |0     |[49.07,51.88]|1.0  |[1.387081807081807,18.612918192918194]  |[0.06935409035409035,0.9306459096459097] |1.0       |
|50.29 |49.80 |0     |[50.29,49.8] |1.0  |[1.387081807081807,18.612918192918194]  |[0.06935409035409035,0.9306459096459097] |1.0       |
+------+------+------+-------------+-----+----------------------------------------+-----------------------------------------+----------+
 */

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")


    val accuracy = evaluator.evaluate(prediction_df)
    println("accuracy % = " + accuracy * 100)


    import spark.implicits._
    val df1 = Seq(
      (35,35),(35,45),(76,76),(30,40),(46,30),(98,39),(45,18),(56,84),(39,39),(38,39),(39,60)
    ).toDF("score1","score2")

    df1.show()

    val df2 = assembler.transform(df1)
    df2.show()

    val df3 = model.transform(df2)
    df3.show(false)





}
}
