package fr.eurecom.dsg.treelib.id3

import fr.eurecom.dsg.treelib.core._



/**
 * An interface of node in tree
 */
class ID3Node(xsplitPoint : SplitPoint = null) extends Node {
   
    
    var id : BigInt = 0
    
    /**
     * The split point which this node contains
     */
    def splitpoint: SplitPoint = xsplitPoint
    
    /**
     * The associated feature of this node
     */
    var feature: Feature = null
    
    /**
     * The predicted value of this node
     */
    var value : Any = null
    
    /**
     * The range of indices of children
     */
    var rangeOfChildren : (BigInt, BigInt) = (0,0)	// (minIndex, maxIndex)
    
    /**
     * The number of records/lines/instances which were used to build this node
     */
    var numberOfInstances : Int = 0

    /**
     * The statistical information of the target feature
     */
    var statisticalInformation : StatisticalInformation = _
    
    /**
     * Parent
     */
    var parent : ID3Node = null
    
    /**
     * Is this node empty?
     */
    var isLeaf: Boolean = true
    
    override def toLeafNode() : Node = {
        this.isLeaf = true
        this.children = null
        this.rangeOfChildren = (0,0)
        this
    }
    
    def containNode(id : BigInt) : Boolean = {
        id >= rangeOfChildren._1 && id <= rangeOfChildren._2
    }
    
    /**
     * Find a child by its id
     * @output	(node, indexInChildrenArray)
     */
    def findChildrenByID(id : BigInt) : (Node, Int) = {
        def findChildrenIter(left: Int, right:Int) : (Node,Int) = {
            if (right < left)
                return (null,-1)
            val mid = (left + right)/2
            if (children(mid).asInstanceOf[ID3Node].id == id)
                return (children(mid),mid)
            else {
                if (children(mid).asInstanceOf[ID3Node].id > id) {
                    findChildrenIter(left, mid-1)
                } else{
                    findChildrenIter(mid + 1, right)
                }
            }
        }
        findChildrenIter(0, numberOfChildren - 1)
    }
    
    
    /**
     * A function support to convert an object of this class to string
     * @param level The level of this node. Root node has level 1
     */
    def toStringWithLevel(level: Int): String = {
        if (isLeaf) {
            "[%s] \"".format(id) + value + "\"  info:" + statisticalInformation
        }
        else {
            // featureName predict:x infor:y
            //   |
            var str = "[%s] %s predict:%s info%s".format(id,feature.Name, value, statisticalInformation.YValue.asInstanceOf[Array[_]].mkString(","))
            for (i <- 0 until numberOfChildren) {
                str = "%s\n%s-(%s)%s%s".format(str, 
                        ("".padTo(level, "|")).mkString("    "),
                        if (feature.Type == FeatureType.Categorical){
                        	splitpoint.point.asInstanceOf[Array[_]](i)
                        } else {
                            splitpoint.point
                        },
                        ("".padTo(2, "-")).mkString(""),
                        children(i).toStringWithLevel(level + 1)
                        )
            }
            str
            /*
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
            * 
            */
        }
    }
        
    /**
     *     Add a node to the children collection
     */    
    def addChild(node : ID3Node) : BigInt = {

        //println("current node:id = " + id + " insert node id=" + node.id)
        node.parent = this
        if (children == null){
            children = Array[Node]()
        }
        
        if (children.length == 0){
                rangeOfChildren = (node.id, rangeOfChildren._2)
        }
        
        //println("update range of node " + id + " to " + rangeOfChildren)
        
        var p : ID3Node = this
        while (p != null) {
            if (p.rangeOfChildren._2 < node.id){
            	p.rangeOfChildren = (p.rangeOfChildren._1, node.id)
            	//println("update range of node " + p.id + " to " + p.rangeOfChildren)
            }
            
            p = p.parent
        }
        
        
        children = children.:+(node)
        node.id
    }
    
    override def toString: String = "\n" + toStringWithLevel(1)
}