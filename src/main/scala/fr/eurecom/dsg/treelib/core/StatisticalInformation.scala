package fr.eurecom.dsg.treelib.core

class StatisticalInformation(
    
    /**
     *     with RegressionTree: is sum of YValue
     *     with ClassificationTree: is the distribution of yValue
     */    
    var YValue: Any = 0,
    
    /**
     * Is sum of power 2 of yValue
     * Only used in Regression Tree
     */
    var sumOfYPower2: Double = 0,
    
    /**
     * The number of observations
     */
    var numberOfInstances: Int = 0) extends Serializable {
    
    override def toString(): String = { "(%s,%f,%d)".format(
            if (YValue.isInstanceOf[Array[_]]) YValue.asInstanceOf[Array[_]].mkString(";") else YValue, sumOfYPower2, numberOfInstances) }
}
