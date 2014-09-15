package fr.eurecom.dsg.treelib.id3

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import java.io._
import fr.eurecom.dsg.treelib.cart.CARTNode
import fr.eurecom.dsg.treelib.core._
import scala.math.BigInt.int2bigInt

/**
 * *
 * This class is representative for a tree model.
 * It contains all related information: index of target feature, indexes of the other features which were used to build tree
 */
class ID3TreeModel extends TreeModel {

    def createNode(sp : SplitPoint = null) : ID3Node = {
        var newNode = new ID3Node(sp)
        newNode.id = getNextID
        newNode
    }
    
    //object ID3Node extends Serializable {
        private var MAXIMUM_ID = 1
        def getNextID: Int = {
            this.MAXIMUM_ID = this.MAXIMUM_ID + 1
            MAXIMUM_ID
        }
    //}

    
        
    /**
     * Return a node and its index in parent's children array
     */
    def findNodeByID(id : BigInt) : (ID3Node, Int) =  {
        def findNodeByIDIter(currentNode : ID3Node, indexInParent : Int) : (ID3Node, Int) = {
            if (currentNode != null) {
                //println("consider node " + currentNode.id + " with range" + currentNode.rangeOfChildren)
                if (currentNode.id == id)
                    return (currentNode, indexInParent)
                else if (currentNode.containNode(id)) {
                    var i = 0
                    while (i < currentNode.numberOfChildren) {
                        if (currentNode.children(i).asInstanceOf[ID3Node].id == id){
                            val result =  (currentNode.children(i).asInstanceOf[ID3Node], i)
                            if (result._1 != null)
                                return result
                        }
                        if (currentNode.children(i).asInstanceOf[ID3Node].containNode(id)){
                            val result =  findNodeByIDIter(currentNode.children(i).asInstanceOf[ID3Node], i)
                            if (result._1 != null)
                                return result
                        }
                        i = i + 1
                    }
                }
            }
            (null,-1)
        }
        //println("find id=" + id)
        findNodeByIDIter(tree.asInstanceOf[ID3Node], 1)
    }
    
    private implicit object OrderingAny extends Ordering[Any] {
        def compare(x : Any, y : Any) : Int = {
            x.toString.compareTo(y.toString)
        }
    }
    
    def find[T <: Any](array: Array[T], element: T)(implicit odering : Ordering[T]) : Int = {
        def findIter(left: Int, right:Int) : Int = {
            if (right < left)
                return -1
            val mid = (left + right)/2
            if (array(mid) == element)
                return mid
            else {
            
                if (array(mid).toString.compareTo(element.toString) > 0)
                //if (ordering.compare(array(mid),element ) > 0)
                {
                    findIter(left, mid-1)
                } else{
                    findIter(mid + 1, right)
                }
            }
        }
        findIter(0, array.length - 1)
    }
    
	/**
     * Predict Y base on input features
     * 
     * @param record			an array, which its each element is a value of each input feature
     * @param ignoreBranchSet	a set of branch ID, which won't be used to predict (only use it's root node)
     * @return a predicted value
     * @throw Exception if the tree is empty
     */
    override def predict(record: Array[String], ignoreBranchSet: Set[BigInt] = Set[BigInt]()): String = {

        def predictIter(currentNode: Node): String = {
            if (currentNode.isLeaf || ignoreBranchSet.contains(currentNode.asInstanceOf[ID3Node].id)) {
                currentNode.value.toString
            } else {
                val fIndex = currentNode.feature.index
                currentNode.feature.Type match {
                    case FeatureType.Categorical => {
                        val splitpoint = currentNode.splitpoint.point.asInstanceOf[Array[Any]]
                        val ind = find(splitpoint, record(fIndex))
                        if (ind >= 0) {
                            predictIter(currentNode.children(ind))
                        }else {
                            currentNode.value.toString
                        }
                    }
                    case FeatureType.Numerical => {
                        val splitpoint = currentNode.splitpoint.point.asInstanceOf[Double]
                        if (record(fIndex).toDouble <= splitpoint) {
                            predictIter(currentNode.children(0))
                        }else{
                            predictIter(currentNode.children(1))
                        }
                    }
                }
                /*
                currentNode.feature.Type match {
                    case FeatureType.Categorical => {
                        if (currentNode.splitpoint.point.asInstanceOf[Set[String]].contains(record(currentNode.feature.index)))
                            predictIter(currentNode.asInstanceOf[CARTNode].left, currentID << 1)
                        else predictIter(currentNode.asInstanceOf[CARTNode].right, (currentID << 1) + 1)
                    }
                    case FeatureType.Numerical => {
                        if (record(currentNode.feature.index).toDouble < currentNode.splitpoint.point.asInstanceOf[Double])
                            predictIter(currentNode.asInstanceOf[CARTNode].left, currentID << 1)
                        else predictIter(currentNode.asInstanceOf[CARTNode].right, (currentID << 1) + 1)
                    }
                }
                * 
                */
                // TODO: implement prediction here
            }
        }

        if (isEmpty) throw new Exception("ERROR: The tree is empty.Please build tree first")
        else predictIter(tree)
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
    		  + "* xIndexes:" + xIndexes//.map(index => fullFeatureSet.getIndex(usefulFeatureSet.data(index).Name))  
    		  + "\n"
    		  + "* yIndex (target index):" + yIndex + "\n"//+ fullFeatureSet.getIndex(usefulFeatureSet.data(yIndex).Name) + "\n"
    		  + "* minSplit:%d\n".format(this.minsplit)
    		  + "* threshold:%f\n".format(this.threshold)
    		  + "* maximumComplexity:%f\n".format(this.maximumComplexity)
    		  + "* Is complete:" + isComplete + "\n"
    		  + "* Tree:\n"
    		  + tree
      )
    }
}
