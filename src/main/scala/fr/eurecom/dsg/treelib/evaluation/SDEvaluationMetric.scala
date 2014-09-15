package fr.eurecom.dsg.treelib.evaluation

import org.apache.spark.rdd.RDD
import fr.eurecom.dsg.treelib.core._

/**
 * This class use standard deviation and square error to determine the accuracy of a prediction
 */
class SDEvaluationMetric extends BaseEvaluation {
    /**
     * Evaluate the accuracy of a prediction
     * 
     * @param predictedResult the predicted values
     * @param actualResult the actual result
     */
	override def Evaluate(predictedResult: RDD[String], actualResult: RDD[String]) : Any = {
	    
	    var newRDD = predictedResult zip actualResult
	    
        var validRDD = newRDD.filter(v => (!v._1.equals("???") 
            && (Utility.parseDouble(v._2) match {
                case Some(d :Double) => true 
                case None => false 
        }))) // filter invalid record, v._1 is predicted value
        
        
        var invalidRDD = newRDD.filter(v => (v._1.equals("???") 
            || !(Utility.parseDouble(v._2) match {
                case Some(d :Double) => true 
                case None => false 
        }))) // filter invalid record, v._1 is predicted value

        //val numTest = validRDD.count

        val diff = validRDD.map(x => (x._1.toDouble, x._2.toDouble, 1)).map(x => (x._2 - x._1, (x._2 - x._1) * (x._2 - x._1) , x._3))

        val sums = diff.reduce((x, y) => (x._1 + y._1, x._2 + y._2, x._3 + y._3))

        val numTest = sums._3
        val meanDiff = sums._1 / numTest
        val meanDiffPower2 = sums._2 / numTest
        val deviation = math.sqrt(meanDiffPower2 - meanDiff * meanDiff)
        val standardError = deviation / math.sqrt(numTest)
        val numInvalidRecords = invalidRDD.count

        println("Mean of error:%f\nDeviation of error:%f\nStandard error of error:%f\nMean square error of error:%f".format(meanDiff, deviation, standardError, meanDiffPower2))
	    println("Total records: %d".format(numTest) )
	    println("Number of invalid records: %d".format(numInvalidRecords))
	    
	    new StatisticalInformation(sums._1, sums._2, numTest)	// different, different^2,numberInput
	    //(sums._1, sums._2, numTest)	// different, different^2,numberInput
	}
}