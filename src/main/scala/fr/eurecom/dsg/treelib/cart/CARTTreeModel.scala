package fr.eurecom.dsg.treelib.cart

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import java.io._
import fr.eurecom.dsg.treelib.core._
import scala.math.BigInt.int2bigInt

/***
 * This class is representative forimport treelib.cart.CARTNode
 a tree model. 
 * It contains all related information: index of target feature, indexes of the other features which were used to build tree
 */
class CARTTreeModel extends TreeModel {
    
	/**
     * Predict Y base on input features
     * 
     * @param record			an array, which its each element is a value of each input feature
     * @param ignoreBranchSet	a set of branch ID, which won't be used to predict (only use it's root node)
     * @return a predicted value
     * @throw Exception if the tree is empty
     */
    override def predict(record: Array[String], ignoreBranchSet: Set[BigInt] = Set[BigInt]()): String = {

        def predictIter(currentNode: CARTNode, currentID: BigInt): String = {
            if (currentNode.isLeaf || ignoreBranchSet.contains(currentID)) {
                currentNode.value.toString
            } else {
                currentNode.feature.Type match {
                    case FeatureType.Categorical => {
                        if (currentNode.splitpoint.point.asInstanceOf[Set[String]].contains(record(currentNode.feature.index)))
                            predictIter(currentNode.left, currentID << 1)
                        else predictIter(currentNode.right, (currentID << 1) + 1)
                    }
                    case FeatureType.Numerical => {
                        if (record(currentNode.feature.index).toDouble < currentNode.splitpoint.point.asInstanceOf[Double])
                            predictIter(currentNode.left, currentID << 1)
                        else predictIter(currentNode.right, (currentID << 1) + 1)
                    }
                }
            }
        }

        if (isEmpty) throw new Exception("ERROR: The tree is empty.Please build tree first")
        else predictIter(tree.asInstanceOf[CARTNode] , 1)
    }
	
	/**
     * Evaluate the accuracy of regression tree
     * @param input	an input record (uses the same delimiter with trained data set)
     * @param delimiter the delimiter of training data
     */
    override def evaluate(input: RDD[String], delimiter : Char = ',') =  {
        if (!this.isEmpty){
            val numTest = input.count
            val inputdata = input.map(x => x.split(delimiter))
            val diff = inputdata.map(x => (predict(x).toDouble, x(yIndex).toDouble)).map(x => (x._2 - x._1, (x._2 - x._1)*(x._2-x._1)))

            val sums = diff.reduce((x,y) => (x._1 + y._1, x._2 + y._2))

            val meanDiff = sums._1/numTest
            val meanDiffPower2 = sums._2/numTest
            val deviation = math.sqrt(meanDiffPower2 - meanDiff*meanDiff)
            val SE = deviation/numTest
            
            println("Mean of different:%f\nDeviation of different:%f\nSE of different:%f".format(meanDiff, deviation, SE) )
        }else {
            throw new Exception("Please build tree first")
        }
    }
    
    
    /**
     * convert instance of this class into string
     */
    override def toString() = {
      (
    		  "* FeatureSet:\n" + fullFeatureSet.toString + "\n"
    		  + "* xIndexes:" + xIndexes  + "\n"
    		  + "* yIndex (target index):" + yIndex + "\n"
    		  + "* minSplit:%d\n".format(this.minsplit)
    		  + "* threshold:%f\n".format(this.threshold)
    		  + "* maximumComplexity:%f\n".format(this.maximumComplexity)
    		  + "* Is complete:" + isComplete + "\n"
    		  + "* Tree:\n"
    		  + tree
      )
    }
}