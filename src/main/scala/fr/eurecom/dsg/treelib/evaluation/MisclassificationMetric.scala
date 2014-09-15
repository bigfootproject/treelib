package fr.eurecom.dsg.treelib.evaluation

import org.apache.spark.rdd.RDD
import fr.eurecom.dsg.treelib.core._

/**
 * This class use standard deviation and square error to determine the accuracy of a prediction
 */
class MisclassificationMetric extends BaseEvaluation {
    /**
     * Evaluate the accuracy of a prediction
     * 
     * @param predictedResult the predicted values
     * @param actualResult the actual result
     */
	override def Evaluate(predictedResult: RDD[String], actualResult: RDD[String]) : Any = {
	    
	    var newRDD = predictedResult zip actualResult
	    newRDD.collect.foreach(println)
	    
	    
        var validRDD = newRDD.filter(v => (!v._1.equals("???"))) // filter invalid record, v._1 is predicted value
        
        
        var invalidRDD = newRDD.filter(v => (v._1.equals("???"))) // filter invalid record, v._1 is predicted value

        //val numTest = validRDD.count

        val diff = validRDD.map(x => {
            (if (x._1 == x._2) 1
            else 0
            , 1)
        })
        
        val sums = diff.reduce((x, y) => (x._1 + y._1, x._2 + y._2))

        val numTest = sums._2
        val numTruePrediction = sums._1
        
        val numInvalidRecords = invalidRDD.count

        println("Number of true prediction:%d Rate:%d/%d".format(numTruePrediction, numTruePrediction, numTest))
	    println("Total records: %d".format(numTest) )
	    println("Number of invalid records: %d".format(numInvalidRecords))
	    
	    (numTruePrediction, numTest)

	}
}