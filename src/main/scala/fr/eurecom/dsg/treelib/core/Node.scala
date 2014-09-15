package fr.eurecom.dsg.treelib.core

trait Node extends Serializable {
    /**
     * The split point which this node contains
     */
    def splitpoint: SplitPoint
    
    /**
     * The associated feature of this node
     */
    var feature: Feature
    
    /**
     * The predicted value of this node
     */
    var value : Any
    
//    /**
//     * The number of records/lines/instances which were used to build this node
//     */
//    var numberOfInstances : Int
//    
//    /**
//     * The metric to consider the error of prediction of this node
//     */
//    var errorMetric : Double
    
    var statisticalInformation : StatisticalInformation
    
    /**
     * Is this node empty?
     */
    def isLeaf: Boolean
    
    /**
     * Children
     */
    var children : Array[Node] = _
    
    /**
     * Number of children
     */
    def numberOfChildren = if (children !=  null) children.length else 0
    
    /**
     * Update the children of position "index"
     */
    def setChildren(index : Int, node : Node) = {
        if (index >=0 && index < numberOfChildren){
            children.update(index, node)
        }
    }
    
    /**
     * Convert the current node to a leaf node
     */
    def toLeafNode() : Node
    
    /**
     * A function support to convert an object of this class to string
     * @param level The level of this node. Root node has level 1
     */
    def toStringWithLevel(level: Int): String
    
    override def toString: String = "\n" + toStringWithLevel(1)
}
