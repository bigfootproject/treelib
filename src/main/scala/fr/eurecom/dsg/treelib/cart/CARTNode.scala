package fr.eurecom.dsg.treelib.cart

import fr.eurecom.dsg.treelib.core._




/**
 * An interface of node in tree
 */
trait CARTNode extends Node {
    
    /**
     * The split point which this node contains
     */
    //def splitpoint: SplitPoint
    
    /**
     * The associated feature of this node
     */
    //var feature: Feature
    
    /**
     * The predicted value of this node
     */
    //var value : Any
    
//    /**
//     * The number of records/lines/instances which were used to build this node
//     */
//    var numberOfInstances : Int
//    
//    /**
//     * The metric to consider the error of prediction of this node
//     */
//    var errorMetric : Double
    
    //var statisticalInformation : StatisticalInformation
    
    /**
     * The left child
     */
    def left: CARTNode
    
    /**
     * The right child
     */
    def right: CARTNode
    
    /**
     * Is this node empty?
     */
    //def isLeaf: Boolean
    
    /**
     * Set the left child
     * @param node the desired left node 
     */
    def setLeft(node : CARTNode): Unit
    
    /**
     * Set the right child
     * @param node the desired right node
     */
    def setRight(node: CARTNode) : Unit
    
    //def toLeafNode() : CARTNode
    
    /**
     * A function support to convert an object of this class to string
     * @param level The level of this node. Root node has level 1
     */
    //def toStringWithLevel(level: Int): String
    
    //override def toString: String = "\n" + toStringWithLevel(1)
}

/**
 * A leaf node
 */
class CARTLeafNode(xValue: String = "Empty") extends CARTNode {
    /**
     * Constructor of node. The default value of this node is 'Empty'
     */
    def this() = this("Empty")
    
    /**
     * Is this node empty ?
     */
    def isLeaf = true
    
    /**
     * Value of this leaf node
     */
    var value : Any = xValue
    
//    /**
//     * Number of instances which were used to build this node
//     */
//    var numberOfInstances : Int = 0
//    
//    
//    /**
//     * Error rate if we use this node to make a prediction
//     */
//    var errorMetric : Double = 0
    
    var statisticalInformation : StatisticalInformation = new StatisticalInformation()
    
    
    
    /**
     * The split point of this node
     */
    def splitpoint: Nothing = throw new NoSuchElementException("empty.splitpoint")
    
    /**
     * Get the left child of this node
     */
    def left: Nothing = throw new NoSuchElementException("empty.left")
    
    /**
     * Get the right child of this node
     */
    def right: Nothing = throw new NoSuchElementException("empty.right")
    

    /**
     * Set the left child
     * @param node The desired left child
     */
    def setLeft(node : CARTNode)= {}
    
    /**
     * Set the right child
     * @param node The desired right child
     */
    def setRight(node: CARTNode)= {}
    
    def toLeafNode() : CARTNode = this
    
    /**
     * The feature which is associated to this node
     */
    var feature: Feature = _ //FeatureInfo("Empty", "d", 0)
    
    def toStringWithLevel(level: Int) = "\"" + value + "\"  info:" + statisticalInformation
}

/**
 * An internal node (the node contains children)
 * 
 * @param xFeature		the associated feature
 * @param xSplitpoint	the split point of feature which associated to this node
 * @param xLeft			the left child
 * @param xRight		the right child
 */
class CARTNonLeafNode(
        xFeature: Feature, 
        xSplitpoint: SplitPoint, 
        var xLeft: CARTNode = new CARTLeafNode("empty.left"), 
        var xRight: CARTNode = new CARTLeafNode("empty.right")
) extends CARTNode {
    
    /**
     * Predicted value if we use this node to make prediction
     */
    var value : Any = xSplitpoint.point
    
//    /**
//     * The error rate if we use this node to make a prediction
//     */
//    var errorMetric : Double = 0
//    
//     /**
//     * Number of instances which were used to build this node
//     */
//    var numberOfInstances : Int = 0
    
    
    var statisticalInformation : StatisticalInformation = new StatisticalInformation()
    
    /**
     * Is this node empty ?
     */
    def isLeaf = false
    
    /**
     * The split point of feature which associated to this node
     */
    def splitpoint = xSplitpoint
    
    /**
     * Get the left child
     */
    def left = xLeft
    
    /**
     * Get the right child
     */
    def right = xRight
    
    /**
     * Set the left child
     * @param node The desired left node
     */
    def setLeft(node: CARTNode) = {xLeft = node;}
    
    /**
     * Set the right child
     * @param node The desired right node
     */
    def setRight(node: CARTNode) = {xRight = node;}
    
    /**
     * The feature which is associated to this node
     */
    var feature: Feature = xFeature
    
    def toLeafNode() : CARTNode = {
        var newleaf = new CARTLeafNode(this.value.toString)
        newleaf.statisticalInformation = this.statisticalInformation
        newleaf
    }
    
    /**
     * Get the conditions to go to the left and right child
     */
    /*
    val (conditionLeft, conditionRight) = feature.Type match {
        case FeatureType.Categorical => (this.splitpoint.point , "Not in %s".format(this.splitpoint.point.toString))
        case FeatureType.Numerical => ("%s < %f".format(feature.Name, splitpoint.point), "%s >= %f".format(feature.Name, splitpoint.point))
    }*/

    def toStringWithLevel(level: Int) =
        "%s(%s)  %s\n%s-(yes)%s%s\n%s-(no)-%s%s".format(
            feature.Name,
            (feature.Type match {
                case FeatureType.Categorical => Utility.setToString(splitpoint.point.asInstanceOf[Set[String]])
                case FeatureType.Numerical => " < %f".format(splitpoint.point.asInstanceOf[Double])
            }),
            "predict:%s  info:%s".format(value, statisticalInformation) ,
            ("".padTo(level, "|")).mkString("    "),
            ("".padTo(2, "-")).mkString(""),
            left.toStringWithLevel(level + 1),
            ("".padTo(level, "|")).mkString("    "),
            ("".padTo(2, "-")).mkString(""),
            right.toStringWithLevel(level + 1)
            )
}