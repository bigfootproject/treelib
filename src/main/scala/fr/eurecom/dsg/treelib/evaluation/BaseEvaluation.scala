package fr.eurecom.dsg.treelib.evaluation
import org.apache.spark.rdd._

abstract class BaseEvaluation {
    
    /**
     * Evaluate the accuracy of a prediction
     * 
     * @param predictedResult the predicted values
     * @param actualResult the actual result
     */
	def Evaluate(predictedResult: RDD[String], actualResult: RDD[String]) : Any
}