package fr.eurecom.dsg.treelib.evaluation

import org.apache.spark.rdd._

/**
 * Use to evaluate a prediction
 */
object Evaluation {
    
    /**
     * The metric which we want to use
     */
    var metric : BaseEvaluation = new SDEvaluationMetric()
    
    /**
     * Choose the strategy to determine the accuracy metric
     */
    def apply(strategy : String = "standard-deviation") = {
        println("Evaluation Method:" + strategy)
    	strategy match {
            case "misclassification" => { metric = new MisclassificationMetric() }
    	    case whatever => { metric = new SDEvaluationMetric() }
    	}
        this
    }
    
    /**
     * Evaluate the accuracy of regression tree
     *
     * @param 	predictedResult predicted values
     * @param 	actualResult actual values, and may be some more related values
     */
    def evaluate(predictedResult: RDD[String], actualResult: RDD[String]) = {
        if (metric != null){
            metric.Evaluate(predictedResult, actualResult)
        }else{
            println("metric is null")
        }
    }
    
    /**
     * Evaluate the accuracy of regression tree with the custom function
     *
     * @param 	predictedResult predicted values
     * @param 	actualResult actual values, and may be some more related values
     * @f		the custom function
     */
    def evaluateWith(predictedResult: RDD[String], actualResult: RDD[String])(f : (RDD[String], RDD[String]) => Any) = {
        try{
        	f(predictedResult, actualResult)
        }catch{
            case e: Exception =>
            	println("ERROR:Something wrong:" + e.getMessage())
        }
    }
    
}