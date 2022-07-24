import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DecimalType

import java.util.Calendar

object teach_scores_lin_reger {

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

    marksDF.show(5,false)

    val marksDf_new= marksDF
      .withColumn("score1", col ( "score1").cast(DecimalType(6,2)))
      .withColumn("score2", col ( "score2").cast(DecimalType(6,2)))

    marksDf_new.show(5,false)

    marksDf_new.describe("score1").show()

    marksDf_new.describe("score2").show()

    marksDf_new.describe("result").show()

    marksDf_new.describe().show()

    marksDf_new.createOrReplaceTempView("my_View")
    spark.sql("select * from my_View limit 10") .show()

    spark.sql("select * from my_view  where result =1 limit 10") .show()

    spark.sql("select * from my_view where result =1 order by score1 asc  limit 10 ") .show()

    spark.sql("select * from my_view where result =1 order by score2 asc  limit 5 ") .show()

    spark.sql("select * from my_view where score1 < 40   limit 5 ") .show()

    val cols = Array("score1","score2")
    val my_Assembler = new VectorAssembler()
      .setInputCols(cols)
      .setOutputCol("features")
    val marks_Feature = my_Assembler.transform(marksDf_new)

    marks_Feature.printSchema()

    marks_Feature.show(5,false)

    val indexer = new StringIndexer()
      .setInputCol("result")
      .setOutputCol("label")
    // sethandleinvaild is used to incorporate unseen labels or to eliminate label not found error
    val marks_Label = indexer.setHandleInvalid("keep")
      .fit(marks_Feature)
      .transform(marks_Feature)
    marks_Label.printSchema()

    marks_Label.show(10,false)

    val seed = 5043
    val Array(trainData,testData)=marks_Label.randomSplit(Array(0.65,0.35),seed)

    //train above splited data frames used logistic regression

    val linearRegression=new LinearRegression()
      .setMaxIter(20)
      .setRegParam(0.02)
      .setElasticNetParam(0.8)

    // create model using training data

    val linearRegression_model=linearRegression.fit(trainData)

    //run the above model using test data to get prediction

    val predictionDf=linearRegression_model.transform(testData)
    predictionDf.show(5,false)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")
      .setMetricName("areaUnderROC")
    val accuracy = evaluator.evaluate(predictionDf)
    println("accuracy % = " + accuracy * 100)

    import spark.implicits._
    val df1 = Seq(
      (35,35),(35,45),(76,76),(30,40),(46,30),(98,39),(45,18),(56,84),(39,39),(38,39),(39,60)
    ).toDF("score1","score2")

    df1.show()

    val df2 = my_Assembler.transform(df1)
    df2.show()

    val df3 = linearRegression_model.transform(df2)
    df3.show(false)

/*
+------+------+-----------+-------------------+
|score1|score2|features   |prediction         |
+------+------+-----------+-------------------+
|35    |35    |[35.0,35.0]|1.2206756675399908 |
|35    |45    |[35.0,45.0]|1.1102256936642327 |
|76    |76    |[76.0,76.0]|0.12430360513493977|
|30    |40    |[30.0,40.0]|1.2439296037136292 |
|46    |30    |[46.0,30.0]|1.1032470236325316 |
|98    |39    |[98.0,39.0]|0.18766124678456864|
|45    |18    |[45.0,18.0]|1.2514827769057448 |
|56    |84    |[56.0,84.0]|0.34985931848040286|
|39    |39    |[39.0,39.0]|1.1137125395004737 |
|38    |39    |[38.0,39.0]|1.1294083241227773 |
|39    |60    |[39.0,60.0]|0.8817675943613814 |
+------+------+-----------+-------------------+
 */

    println("Starting time and date ="+start_date_time)
    var end_Time= Calendar.getInstance()
    var end_date_time= end_Time.getTime()
    println("End time and date ="+end_date_time)




  }
}
