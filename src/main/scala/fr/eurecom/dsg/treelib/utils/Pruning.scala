package fr.eurecom.dsg.treelib.utils
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import fr.eurecom.dsg.treelib.core._
import fr.eurecom.dsg.treelib.evaluation.Evaluation
import fr.eurecom.dsg.treelib.cart.CARTNode
import fr.eurecom.dsg.treelib.cart.RegressionTree
import fr.eurecom.dsg.treelib.cart.ClassificationTree
import fr.eurecom.dsg.treelib.id3.ID3Node
import fr.eurecom.dsg.treelib.id3.ID3TreeBuilder

/*
object Pruning {
    private val DEBUG : Boolean = false
    
    var maxcp = 0.0
    
    private def risk(node : Node) : Double = {
        // in case of Regression Tree
        if  ( node.isInstanceOf[CARTNode] && node.statisticalInformation.YValue.isInstanceOf[Double]){
	        val EX = node.statisticalInformation.YValue.asInstanceOf[Double] / node.statisticalInformation.numberOfInstances
	        val EX2 = node.statisticalInformation.sumOfYPower2 / node.statisticalInformation.numberOfInstances 
	        //math.sqrt(EX2 - EX*EX)*math.sqrt(node.statisticalInformation.numberOfInstances)///node.statisticalInformation.numberOfInstances 	// MSE
	        (EX2 - EX*EX)*(node.statisticalInformation.numberOfInstances)
        }
        else{	// case of Classification tree, we calculate the misclasification rate
            //println("yvalue:" + node.statisticalInformation.YValue + " id3=" + node.isInstanceOf[ID3Node] + " node:" + node)
            //println("test:" + node.statisticalInformation.YValue.asInstanceOf[Array[(Any, Int)]].mkString(","))
            var temp = node.statisticalInformation.YValue.asInstanceOf[Array[(Any, Int)]]
            if (temp.length > 0) {
                var sum : Int = 0
                for ( i <- 1 until temp.length){
                    sum = sum + temp(i)._2
                }
                sum
                //return 1 - temp.head._2*1.0 /node.statisticalInformation.numberOfInstances
            }
            0
        }
    }
    
    /**
     * Prune the full tree to get tree T1 which has the same error rate but smaller size
     */
    private def firstPruneTree(root: Node): Set[BigInt] = {
        
        var leafNodes = Set[BigInt]()
        def isLeaf(node : Node, id : BigInt) = {
            node.isLeaf || leafNodes.contains(id)
        }
        
        def firstPruneTreeIterBinary(node: CARTNode, currentId : BigInt) : Unit = {
            
            if (!node.isLeaf){
                firstPruneTreeIterBinary(node.left, currentId << 1)
                firstPruneTreeIterBinary(node.right, (currentId << 1) + 1)
                if (isLeaf(node.left, currentId << 1) 
                        && isLeaf(node.right, (currentId << 1) + 1)) {
                    val Rl = risk(node.left)
	                val Rr = risk(node.right)
	                val R = risk(node)
	                if (R == (Rl + Rr))
	                    leafNodes = leafNodes + currentId
                }
            }
        }
        
        def firstPruneTreeIterMultiway(node: ID3Node) : Unit = {
            
            if (!node.isLeaf){
                var everyChildIsLeaf = true
                for (i <- 0 until node.numberOfChildren){
                	firstPruneTreeIterMultiway(node.children(i).asInstanceOf[ID3Node])
                	everyChildIsLeaf = everyChildIsLeaf & isLeaf(node.children(i), node.children(i).asInstanceOf[ID3Node].id)
                }
                
                var riskOfChildren : Double = 0.0
                
                for (i <- 0 until node.numberOfChildren){
                    riskOfChildren = riskOfChildren + risk(node.children(i))
                }
                
                val R = risk(node)
	                if (R == riskOfChildren)
	                    leafNodes = leafNodes + node.asInstanceOf[ID3Node].id
                
            }
        }
         
        if (root.isInstanceOf[CARTNode]) {
            firstPruneTreeIterBinary(root.asInstanceOf[CARTNode], 1)
        } else if (root.isInstanceOf[ID3Node]) {
            firstPruneTreeIterMultiway(root.asInstanceOf[ID3Node])
        }
        
        leafNodes
    }
    
   /**
     * Calculate g(t) of every node
     *
     * @return (id_of_node_will_be_pruned, alpha)
     */
    def selectNodesToPrune(root : Node, already_pruned_node : Set[BigInt]) : (Set[BigInt], Double) = {	// (prunedIDs, alpha)
        var result = Set[BigInt]()
        var currentMin : Double = Double.MaxValue
        var currentID : BigInt = 0

        def selectNodeToPrunIterBinary(node: CARTNode, id: BigInt): (Double, Int) = { // (riskOfBranch, numberChildren)
            println("currentID:" + id + " info:" + node.statisticalInformation)
            // if this is an internal node, and the branch of this node hasn't pruned
            if (!node.isLeaf && !already_pruned_node.contains(id)) {

                var (riskBranchLeft, numLeft) = selectNodeToPrunIterBinary(node.left, (id << 1) )
                var (riskBranchRight, numRight) = selectNodeToPrunIterBinary(node.right, (id << 1) + 1)
                var gT = (risk(node) - (riskBranchLeft + riskBranchRight)) / (numLeft + numRight - 1)
                println("Node " + id + " r(t)=" + risk(node) +  "  g(t)=" + gT +
                " (numLeft:%d,gLeft=%f) (numRight=%d, gRight=%f)".format(numLeft, riskBranchLeft, numRight, riskBranchRight))

                if (gT == currentMin) {
                    result = result + id
                }
                else if (gT < currentMin) {
                    result = Set[BigInt]()
                    result = result + id
                    currentMin = gT
                }

								//println("minGT = " + currentMin)
                (riskBranchLeft + riskBranchRight, numLeft + numRight)
            } else {
                (risk(node), 1) // one leaf node
            }
        }
        
        def selectNodeToPrunIterMultiway(node: ID3Node): (Double, Int) = { // (riskOfBranch, numberChildren)
            val id = node.id
            
            println("currentID:" + id + " info:" + node.statisticalInformation)
            // if this is an internal node, and the branch of this node hasn't pruned
            if (!node.isLeaf && !already_pruned_node.contains(id)) {
                
                var riskOfChildren : Double = 0
                var numLeafNodeInBranch : Int = 0
                for (i <- 0 until node.numberOfChildren) {
                    val (risk,numLeafNode) = selectNodeToPrunIterMultiway(node.children(i).asInstanceOf[ID3Node])
                    riskOfChildren = riskOfChildren + risk
                    numLeafNodeInBranch = numLeafNodeInBranch + numLeafNode
                }

                
                var gT = (risk(node) - riskOfChildren) / (numLeafNodeInBranch - 1)

                if (gT == currentMin) {
                    result = result + id
                }
                else if (gT < currentMin) {
                    result = Set[BigInt]()
                    result = result + id
                    currentMin = gT
                }

								//println("minGT = " + currentMin)
                (riskOfChildren, numLeafNodeInBranch)
            } else {
                (risk(node), 1) // one leaf node
            }
        }
        
        if (!already_pruned_node.contains(1)){
            println("root node:" + root)
            if (root.isInstanceOf[CARTNode]){
                println("CARTTTTTTTTTTT")
            	selectNodeToPrunIterBinary(root.asInstanceOf[CARTNode], 1)
            }
            else{
                println("ID3333333")
            	selectNodeToPrunIterMultiway(root.asInstanceOf[ID3Node])    
            }
        }
        
        //println("result:" + result)
        // union with the previous set of pruned nodes
        result = result ++ already_pruned_node;
        var pruned_node_list = result.toList.sortWith((a,b) => a < b)
        
        //filter useless pruned nodes
        // for example, if this set contain 3 and 7;
        // 3 is the parent of 7, we don't need to keep 7 in this set
        pruned_node_list.foreach(id => {
            var tmp = id
            while (tmp > 0){
                if (result.contains(tmp >> 1)){
                    result = result - id
                    tmp = 0
                }
                tmp = tmp >> 1
            }
        })
        
        println("final result" + result)
        
        
        (result, currentMin)
    }
    
    /**
     *  filter useless pruned nodes
     *  for example, if this set contain 3 and 7;
     *  3 is the parent of 7, we don't need to keep 7 in this set
     */
    protected def standardizePrunedNodeSet(nodes : Set[BigInt]) : Set[BigInt] = {
        var result = nodes
        nodes.toArray.sortWith((a,b) => a < b).foreach (
        		id => {
            var tmp = id
            while (tmp > 0){
                if (result.contains(tmp >> 1)){
                    result = result - id
                    tmp = 0
                }
                tmp = tmp >> 1
            }
        }
        )
        result
    }

    private def pruneBranchs(root: Node, prunedNodeIDs: Set[BigInt]): Node = {

        def pruneBranchsIterBinary(node: CARTNode, id: BigInt): CARTNode = {
            if (!node.isLeaf) {
                if (prunedNodeIDs.contains(id)) {
                    node.toLeafNode.asInstanceOf[CARTNode]
                } else {
                    node.setLeft(pruneBranchsIterBinary(node.left, id << 1))
                    node.setRight(pruneBranchsIterBinary(node.right, (id << 1) + 1))
                    node
                }
            } else {
                node
            }
        }

        def pruneBranchsIterMultiway(node: ID3Node): Node = {
            if (!node.isLeaf) {
                if (prunedNodeIDs.contains(node.id)) {
                    node.toLeafNode
                } else {
                    for (i <- 0 until node.numberOfChildren) {
                        node.setChildren(i, pruneBranchsIterMultiway(node.children(i).asInstanceOf[ID3Node]))    
                    }
                    node
                }
            } else {
                node
            }
        }
        
        if (root.isInstanceOf[CARTNode])
            pruneBranchsIterBinary(root.asInstanceOf[CARTNode], 1)
        else if (root.isInstanceOf[ID3Node]) {
            pruneBranchsIterMultiway(root.asInstanceOf[ID3Node])
        } else null

    }
    
    /*
    private def pruneBranch(root: Node, prunedNodeID: BigInt): Node = {
        var currentid: BigInt = 0

        val level = (Math.log(prunedNodeID.toDouble) / Math.log(2)).toInt
        var i: Int = level - 1
        if (root.isInstanceOf[CARTNode]) {
            var parent = root.asInstanceOf[CARTNode]; // start adding from root node

            // try to find the parent of pruned node
            while (i > 0) {

                if ((prunedNodeID / (2 << i - 1)) % 2 == 0) {
                    // go to the left
                    parent = parent.left
                } else {
                    // go go the right
                    parent = parent.right
                }
                i -= 1
            } // end while

            if (prunedNodeID % 2 == 0) {
                parent.setLeft(parent.left.toLeafNode.asInstanceOf[CARTNode])
            } else {
                parent.setRight(parent.right.toLeafNode.asInstanceOf[CARTNode])
            }
        }
        else if (root.isInstanceOf[ID3Node]) {
            // TODO: Implement here
            throw new Exception( "Not implement pruneBranch in Prunning")
            //root.asInstanceOf[ID3Node].
        }
        root

    }
    */
    
    def getSubTreeSequence(tree : Node) : List[(Set[BigInt], Double)] = {
        println("root:" + tree + " is ID3:" + tree.isInstanceOf[ID3Node])
        var leafNodesOfT1 = firstPruneTree(tree)	// get T1
        if (DEBUG) println("Leaf nodes of after the first prune:" + leafNodesOfT1)
	    
	    var sequence_alpha_tree = List[(Set[BigInt], Double)]((leafNodesOfT1, 0))
        //var sequence_alpha_tree = List[(Set[BigInt], Double)]()
	    
	    var finish = false
	    
	    var prunedNodeSet = standardizePrunedNodeSet(leafNodesOfT1)
	    if (DEBUG) println("Leaf nodes of after the standardization:" + prunedNodeSet)

        // SELECT SEQUENCE OF BEST SUB-TREE AND INTERVALS OF ALPHA
        do {
            var (nodesNeedToBePruned, alpha) = selectNodesToPrune(tree, prunedNodeSet)
            if (DEBUG) println("select node to prune:" + nodesNeedToBePruned + " alpha:" + alpha)

            sequence_alpha_tree = sequence_alpha_tree.:+((nodesNeedToBePruned, alpha))
            if (DEBUG) println("sequence alpha-tree:" + sequence_alpha_tree)

            prunedNodeSet = nodesNeedToBePruned

            finish = (nodesNeedToBePruned.contains(1)) // until we prune branch of the root node
        } while (!finish)

        sequence_alpha_tree    
    }
    
    def getTreeIndexByAlpha(givenAlpha : Double, sequence_tree_alpha : List[(Set[BigInt], Double)]) : Int = {
        if (givenAlpha == 0.0)
            0
        else {
            var i = 0
            val sequence_tree_alpha_with_infinity_alpha = sequence_tree_alpha :+ (Set[BigInt](), Double.MaxValue)
            var result: Int =
                sequence_tree_alpha_with_infinity_alpha.indexWhere {
                    case (leafNodes, alpha) =>
                        {
                            alpha >= givenAlpha
                        }
                } - 1

            result
        }
    }
    /**
     * @param treeModel
     * @param complexityParamter
     * @param dataset
     * @return
     */
    def Prune(treeModel : TreeModel, complexityParamter : Double, dataset: RDD[String], foldTimes : Int = 5) : TreeModel = {
	    
        if (treeModel.isEmpty)
            return treeModel
        
        // First, from the full tree, get all possible pruned sub-trees and the corresponding intervals of alpha (complexity parameter)
        // but we don't know which alpha is the best value 
        // (if we can determine the best value for alpha, we can select the corresponding tree)
        this.maxcp = complexityParamter
	    
	    var sequence_alpha_tree = getSubTreeSequence(treeModel.tree) 
	    if (DEBUG) println("sequence_alpha_tree:" + sequence_alpha_tree.mkString(" || "))
	        
	        
	    // CROSS-VALIDATION
	    // for each interval of alpha, we select the typical value: the geometric midpoint. 
	    // We get a list of typical value for alpha. We call it: list of betas (1)
	    // to determine best value for alpha, we can do cross-validation:
	    // we divide the original training set D into N partitions: D1,D2,...,DN
	    // In each fold:
	    //     for instance, fold k, we use dataset D\Dk to build the tree model and use the rest for testing
	    //     after building the full tree, we use the same technique to get all possible pruned subtree (2) 
	    //     for each beta in list (1), we select the corresponding pruned tree in (2) and calculate the error square
	    //     so, in each fold, we get a list of (beta, error) or for simple we can use list of (index_of_beta, error)
	    // After cross-validation, we have a matrix MxN, M is the number values of beta; N is the number of fold
	    // value of (i,j) is the square error of tree corresponding to beta_j in fold i.
	    // for each beta, we calculate the average of error and chose the beta which has minimum error.
	    // But, it's not the best solution, we can use 1-SE rule to select the better tree:
	    //  after we have the minimum error, select the smallest tree, which has the error <= minimum_error + SE(error)
	    
	    val N = foldTimes
        var newdata = dataset.mapPartitions(partition => {
            var i = -1
            partition.map(x => {
            	i = (i + 1) % N
            	(i, x)
            })
        })
        
        var yIndex = treeModel.fullFeatureSet.getIndex(treeModel.usefulFeatureSet.data(treeModel.yIndex).Name)
        var mapTreeIndexToListErrorMetric : Map[Int, List[Double]] = Map[Int, List[Double]]()
        
        println("Start CROSS-VALIDATION")
        
        for (fold <- (0 until N)){
            println("\n ==================== Begin round %d ====================\n".format(fold))
            // split dataset into training data and testing data
            var datasetOfThisFold = newdata.filter(x => x._1 != fold).map(x => x._2)
            var testingData = newdata.filter(x => x._1 == fold).map(x => x._2)
            
            // build the full tree with the new training data
            val tree = if (treeModel.usefulFeatureSet.data(treeModel.yIndex).Type == FeatureType.Numerical){
                println("Regression Tree")
                //if (treeModel.tree.isInstanceOf[CARTNode])
                	new RegressionTree()
                                
            }
            else{
                if (treeModel.tree.isInstanceOf[CARTNode])
                    new ClassificationTree()
                else
                    new ID3TreeBuilder()
            }
            tree.setDataset(datasetOfThisFold)
            if (DEBUG) println("feature names:" + treeModel.fullFeatureSet.data.map(x => x.Name).toArray.mkString(","))
            tree.setFeatureNames(treeModel.fullFeatureSet.data.map(x => x.Name).toArray)
            tree.setMinSplit(treeModel.minsplit)
            tree.setMaxDepth(treeModel.maxDepth)
            tree.setThreshold(treeModel.threshold)
            tree.setMaximumComplexity(treeModel.maximumComplexity)
            
            val treeModelOfThisFold = tree.buildTree(treeModel.yFeature, treeModel.xFeatures)
            
            //println("new model:\n" + treeModelOfThisFold)
            
            // broadcast the tre for using later (in prediction)
            var tree_broadcast = dataset.context.broadcast(tree)
            
            // get 'best' pruned tree candidates of the current tree 
            val sequence_alpha_tree_this_fold = getSubTreeSequence(treeModelOfThisFold.tree)
            
            if (DEBUG) println("sequence_alpha_tree_this_fold\n%s".format(sequence_alpha_tree_this_fold.mkString(",")))
            
            // init the list of (alpha,sub-tree) pairs
            var list_subtree_correspoding_to_beta = List[(Int,Set[BigInt])]()
            
            // beta is the typical value of each interval of alpha
            // this is the geometric midpoint of the interval: [alpha1, alpha2] => beta1 = sqrt(alpha1*alpha2)
            // get the list of sub-tree depends on value of beta
            for (i <- (0 to sequence_alpha_tree.length -2)){
                val beta = math.sqrt(sequence_alpha_tree(i)._2 * sequence_alpha_tree(i + 1)._2)
                val index = getTreeIndexByAlpha(beta, sequence_alpha_tree_this_fold)
                if (index < 0 && DEBUG) println("current beta:" + beta + "alpha_i:" + sequence_alpha_tree(i)._2  + " alpha_i+1:" + sequence_alpha_tree(i+1)._2 )
                list_subtree_correspoding_to_beta = list_subtree_correspoding_to_beta.:+( i , sequence_alpha_tree_this_fold(index)._1)
            }
            
            // get the prediction of all sub-tree in this fold
            // we don't want to send many tree model through the network
            // so, the solution is: send the 'full' tree model (or the tree has been pre-processed already)
            // and the lists of leaf nodes of each pruned-tree candidates
            var predicted_value_by_subtrees = testingData.map(line => {
		        var record = line.split(",")
		        (
		        		list_subtree_correspoding_to_beta.map(sequence => {
				            (sequence._1, tree_broadcast.value.predictOneInstance(record, sequence._2))
				        }),
		        	record(yIndex)
		        )
            })
            
            var diff_predicted_value_and_true_value = (
                    predicted_value_by_subtrees.flatMap{
                        case (list_of_index_predictedValue, truevalue) =>{
                            list_of_index_predictedValue.map{case (index, predictedValue) => (index, predictedValue, truevalue)}
                        }
                    }
            )
            // RDD[(index, predictedValue, trueValue)]

            if (treeModel.usefulFeatureSet.data(treeModel.yIndex).Type == FeatureType.Numerical) {	// Regression Tree
                var valid_diff_predicted_value_and_true_value = diff_predicted_value_and_true_value.filter(v => (!v._2.equals("???")
                    && (Utility.parseDouble(v._3) match {
                        case Some(d: Double) => true
                        case None => false
                    }))) // filter invalid record, v._2 is predicted value

                val diff = (valid_diff_predicted_value_and_true_value
                    .map { case (index, predictedStringValue, trueStringValue) => (index, predictedStringValue.toDouble, trueStringValue.toDouble, 1) }
                    .map { case (index, pValue, tValue, counter) => (index, (pValue - tValue, (pValue - tValue) * (pValue - tValue), counter)) })

                val sums = diff.reduceByKey((x, y) => {
                    (x._1 + y._1, x._2 + y._2, x._3 + y._3)
                })

                val squareErrors = sums.map {
                    case (index, (sumDiff, sumDiffPower2, numInstances)) => (index, sumDiffPower2 / numInstances)
                }.collect

                squareErrors.foreach {
                    case (idx, mse) => {
                        var errors = mapTreeIndexToListErrorMetric.getOrElse(idx, List[Double]())
                        errors = errors.+:(mse)
                        mapTreeIndexToListErrorMetric = mapTreeIndexToListErrorMetric.updated(idx, errors)
                    }
                }

            } else { // Classification tree
                
                var valid_diff_predicted_value_and_true_value = 
                    diff_predicted_value_and_true_value.filter(v => (!v._2.equals("???"))) // filter invalid record, v._2 is predicted value

                val diff = (valid_diff_predicted_value_and_true_value
                    .map { case (index, predictedStringValue, trueStringValue) => 
                        (index, (if (predictedStringValue == trueStringValue) 1 else 0, 1)) })
                
                val sums = diff.reduceByKey((x, y) => {
                    (x._1 + y._1, x._2 + y._2)
                })

                val squareErrors = sums.map {
                    case (index, (truePrediction, numInstances)) => (index, truePrediction*1.0 / numInstances)
                }.collect

                squareErrors.foreach {
                    case (idx, err) => {
                        var errors = mapTreeIndexToListErrorMetric.getOrElse(idx, List[Double]())
                        errors = errors.+:(err)
                        mapTreeIndexToListErrorMetric = mapTreeIndexToListErrorMetric.updated(idx, errors)
                    }
                }
            }
            
            var valid_diff_predicted_value_and_true_value = diff_predicted_value_and_true_value.filter(v => (!v._2.equals("???") 
            && (Utility.parseDouble(v._3) match {
                case Some(d :Double) => true 
                case None => false 
            }))) // filter invalid record, v._2 is predicted value
            
            val diff = (valid_diff_predicted_value_and_true_value
                    .map{ case (index, predictedStringValue, trueStringValue) => (index, predictedStringValue.toDouble, trueStringValue.toDouble, 1)}
                    .map{ case (index, pValue, tValue, counter) => (index, (pValue - tValue, (pValue-tValue)*(pValue-tValue), counter))}
            	)

            val sums = diff.reduceByKey((x,y) => {
                (x._1 + y._1, x._2 + y._2, x._3 + y._3)
            })
            
            val squareErrors = sums.map { 
                case (index, (sumDiff, sumDiffPower2, numInstances)) => (index, sumDiffPower2/numInstances) 
            }.collect
            
            squareErrors.foreach {
                case (idx, mse) => {
                    var errors = mapTreeIndexToListErrorMetric.getOrElse(idx, List[Double]())
                    errors = errors.+:(mse)
                    mapTreeIndexToListErrorMetric = mapTreeIndexToListErrorMetric.updated(idx, errors)
                }
            }
            println("\n ==================== End round %d ====================\n".format(fold))
            
        }	// END CROSS-VALIDATION
        
        var indexOfTreeHasMinAverageError = 0
        var minAverageError = Double.MaxValue
        
        // mapTreeIndexToListErrorMetric : Map< key: index-of-beta; value : list of errors>
        if (DEBUG)
            println("mapTreeIndexToListErrorMetric:")
            mapTreeIndexToListErrorMetric.foreach {
            case (key, value) => {
                println("index:" + key + " List of errors:" + value.mkString(","))
            }
        }
        
        val averageErrors = new Array[(Double, Double)](sequence_alpha_tree.length - 1)
        
        mapTreeIndexToListErrorMetric.foreach{
            case (key, value) => {
                var sumError : Double = 0.0
                var sumErrorPower2 : Double = 0.0
                var numElement = 0
                value.foreach(error => { 
                    numElement = numElement + 1; 
                    sumError = sumError + error
                    sumErrorPower2 = sumErrorPower2 + error*error
                    })
                val averageError = sumError/numElement
                if (averageError < minAverageError){
                    minAverageError = averageError
                    indexOfTreeHasMinAverageError = key
                }
                
                val EX = sumError/numElement;
                val EX2 = sumErrorPower2/numElement
                averageErrors(key) = (EX, math.sqrt((EX2 - EX*EX )/numElement))	// (averageError, SE)
                
            }
        }
        
        var indexOfFinalTree : Int = 0
        
        // Select best alpha by 1-SE Rule
        for (i <- (0 to averageErrors.length - 1)){
        	val (averageError, standardError) = averageErrors(i) 
                if (averageError <= averageErrors(indexOfTreeHasMinAverageError)._1 +  averageErrors(indexOfTreeHasMinAverageError)._2) {
                	indexOfFinalTree = i
                }
        }

        println("index of tree having min average error:" + indexOfTreeHasMinAverageError)
        println("min average: " + minAverageError)
        println("index of the final tree:" + indexOfFinalTree)
        if (averageErrors.length > 0)
        println("Error of final tree: (averageError,SE)=" + averageErrors(indexOfFinalTree))
        
        
        
        var leafNodesOfTheBestTree = sequence_alpha_tree(indexOfFinalTree)._1
        println("the final leaf nodes:" + leafNodesOfTheBestTree)
        
        //println("tree after pruning:" + 
        //pruneBranchs(treeModel.tree, leafNodesOfTheBestTree)
        //)
        
        //pruneBranch(treeModel., prunedNodeID)
        
        treeModel.tree = pruneBranchs(treeModel.tree, leafNodesOfTheBestTree)
        
        treeModel
	}
}

*/
abstract class AbstractPruning(treeModel : TreeModel) extends Serializable {
    
    var DEBUG = true
    
    var maxcp = 0.0
    
    private def getTreeBuilder() : TreeBuilder = treeModel.treeBuilder.createNewInstance
    
    protected def getSubTreeSequence(tree : Node) : List[(Set[BigInt], Double)] = {
        var leafNodesOfT1 = standardizePrunedNodeSet(firstPruneTree(tree))	// get T1
        if (DEBUG) println("Leaf nodes of after the first prune:" + leafNodesOfT1)
	    
	    var sequence_alpha_tree = List[(Set[BigInt], Double)]((leafNodesOfT1, 0))
        //var sequence_alpha_tree = List[(Set[BigInt], Double)]()
	    
	    var finish = false
	    
	    var prunedNodeSet = leafNodesOfT1
	    if (DEBUG) println("Leaf nodes of after the standardization:" + prunedNodeSet)

        // SELECT SEQUENCE OF BEST SUB-TREE AND INTERVALS OF ALPHA
        if (!prunedNodeSet.contains(1)) {
            do {
                println("Select node to be pruned")
                var (nodesNeedToBePruned, alpha) = selectNodesToPrune(tree, prunedNodeSet)
                //if (DEBUG) println("select node to prune:" + nodesNeedToBePruned + " alpha:" + alpha)

                sequence_alpha_tree = sequence_alpha_tree.:+((nodesNeedToBePruned, alpha))
                //if (DEBUG) println("sequence alpha-tree:" + sequence_alpha_tree)

                prunedNodeSet = nodesNeedToBePruned

                finish = (nodesNeedToBePruned.contains(1)) // until we prune branch of the root node
            } while (!finish)
        }

        sequence_alpha_tree    
    }
    
    /**
     * Prune the full tree to get tree T1 which has the same error rate but smaller size
     */
    protected def firstPruneTree(root: Node): Set[BigInt] 
    
    /**
     * Calculate g(t) of every node
     *
     * @return (id_of_node_will_be_pruned, alpha)
     */
    def selectNodesToPrune(root : Node, already_pruned_node : Set[BigInt]) : (Set[BigInt], Double)
    
    /**
     *  filter useless pruned nodes
     *  for example, if this set contain 3 and 7;
     *  3 is the parent of 7, we don't need to keep 7 in this set
     */
    protected def standardizePrunedNodeSet(nodes : Set[BigInt]) : Set[BigInt]
    
    private def getTreeIndexByAlpha(givenAlpha : Double, sequence_tree_alpha : List[(Set[BigInt], Double)]) : Int = {
        if (givenAlpha == 0.0)
            0
        else {
            var i = 0
            val sequence_tree_alpha_with_infinity_alpha = sequence_tree_alpha :+ (Set[BigInt](), Double.MaxValue)
            var result: Int =
                sequence_tree_alpha_with_infinity_alpha.indexWhere {
                    case (leafNodes, alpha) =>
                        {
                            alpha >= givenAlpha
                        }
                } - 1

            result
        }
    }
    
    /**
     * @param treeModel
     * @param complexityParamter
     * @param dataset
     * @return
     */
    def Prune(complexityParamter : Double, dataset: RDD[String], foldTimes : Int = 5) : TreeModel = {
        if (treeModel.isEmpty)
            return treeModel
        
        // First, from the full tree, get all possible pruned sub-trees and the corresponding intervals of alpha (complexity parameter)
        // but we don't know which alpha is the best value 
        // (if we can determine the best value for alpha, we can select the corresponding tree)
        this.maxcp = complexityParamter
	    
        println("ORIGIN:" + treeModel.tree + "\n\n\n")
        
	    var sequence_alpha_tree = getSubTreeSequence(treeModel.tree) 
	    sequence_alpha_tree = sequence_alpha_tree.:+((null, Double.MaxValue))
	    if (DEBUG) println("sequence_alpha_tree:" + sequence_alpha_tree.mkString(" || "))
	    
	           
	    // CROSS-VALIDATION
	    // for each interval of alpha, we select the typical value: the geometric midpoint. 
	    // We get a list of typical value for alpha. We call it: list of betas (1)
	    // to determine best value for alpha, we can do cross-validation:
	    // we divide the original training set D into N partitions: D1,D2,...,DN
	    // In each fold:
	    //     for instance, fold k, we use dataset D\Dk to build the tree model and use the rest for testing
	    //     after building the full tree, we use the same technique to get all possible pruned subtree (2) 
	    //     for each beta in list (1), we select the corresponding pruned tree in (2) and calculate the error square
	    //     so, in each fold, we get a list of (beta, error) or for simple we can use list of (index_of_beta, error)
	    // After cross-validation, we have a matrix MxN, M is the number values of beta; N is the number of fold
	    // value of (i,j) is the square error of tree corresponding to beta_j in fold i.
	    // for each beta, we calculate the average of error and chose the beta which has minimum error.
	    // But, it's not the best solution, we can use 1-SE rule to select the better tree:
	    //  after we have the minimum error, select the smallest tree, which has the error <= minimum_error + SE(error)
	    
	    val N = foldTimes
        var newdata = dataset.mapPartitions(partition => {
            var i = -1
            partition.map(x => {
            	i = (i + 1) % N
            	(i, x)
            })
        }).cache
        
        var yIndex = treeModel.yIndex//treeModel.fullFeatureSet.getIndex(treeModel.usefulFeatureSet.data(treeModel.yIndex).Name)
        var mapTreeIndexToListErrorMetric : Map[Int, List[Double]] = Map[Int, List[Double]]()
        
        println("Start CROSS-VALIDATION")
        
        for (fold <- (0 until N)){
            println("\n ==================== Begin round %d ====================\n".format(fold))
            // split dataset into training data and testing data
            var datasetOfThisFold = newdata.filter(x => x._1 != fold).map(x => x._2)
            var testingData = newdata.filter(x => x._1 == fold).map(x => x._2)
            
            println("full tree before:\n" + treeModel.tree)
            
            // build the full tree with the new training data
            val tree = getTreeBuilder            
            tree.setDataset(datasetOfThisFold)
            if (DEBUG) println("feature names:" + treeModel.fullFeatureSet.data.map(x => x.Name).toArray.mkString(","))
            tree.setFeatureNames(treeModel.fullFeatureSet.data.map(x => x.Name).toArray)
            tree.setMinSplit(treeModel.minsplit)
            tree.setMaxDepth(treeModel.maxDepth)
            tree.setThreshold(treeModel.threshold)
            tree.setMaximumComplexity(treeModel.maximumComplexity)
            
            val treeModelOfThisFold = tree.buildTree(treeModel.yFeature, treeModel.xFeatures)
            
            println("new model:\n" + treeModelOfThisFold.tree)
            println("full tree:\n" + treeModel.tree)
            
            // broadcast the tre for using later (in prediction)
            var tree_broadcast = dataset.context.broadcast(tree)
            
            // get 'best' pruned tree candidates of the current tree 
            val sequence_alpha_tree_this_fold = getSubTreeSequence(treeModelOfThisFold.tree)
            
            if (DEBUG) println("sequence_alpha_tree_this_fold\n%s".format(sequence_alpha_tree_this_fold.mkString(",")))
            
            // init the list of (alpha,sub-tree) pairs
            var list_subtree_correspoding_to_beta = List[(Int,Set[BigInt])]()
            
            // beta is the typical value of each interval of alpha
            // this is the geometric midpoint of the interval: [alpha1, alpha2] => beta1 = sqrt(alpha1*alpha2)
            // get the list of sub-tree depends on value of beta
            for (i <- (0 to sequence_alpha_tree.length -2)){
                var beta = math.sqrt(sequence_alpha_tree(i)._2 * sequence_alpha_tree(i + 1)._2)
                if (beta == Double.PositiveInfinity)
                    beta = sequence_alpha_tree(i)._2*1.5
                val index = getTreeIndexByAlpha(beta, sequence_alpha_tree_this_fold)
                if (index < 0 && DEBUG) println("current beta:" + beta + "alpha_i:" + sequence_alpha_tree(i)._2  + " alpha_i+1:" + sequence_alpha_tree(i+1)._2 )
                list_subtree_correspoding_to_beta = list_subtree_correspoding_to_beta.:+( i , sequence_alpha_tree_this_fold(index)._1)
            }
            
            // get the prediction of all sub-tree in this fold
            // we don't want to send many tree model through the network
            // so, the solution is: send the 'full' tree model (or the tree has been pre-processed already)
            // and the lists of leaf nodes of each pruned-tree candidates
            var predicted_value_by_subtrees = testingData.map(line => {
		        var record = line.split(",")
		        (
		        		list_subtree_correspoding_to_beta.map(sequence => {
				            (sequence._1, tree_broadcast.value.predictOneInstance(record, sequence._2))
				        }),
		        	record(yIndex)
		        )
            })
            
            var diff_predicted_value_and_true_value = (
                    predicted_value_by_subtrees.flatMap{
                        case (list_of_index_predictedValue, truevalue) =>{
                            list_of_index_predictedValue.map{case (index, predictedValue) => (index, predictedValue, truevalue)}
                        }
                    }
            )
            // RDD[(index, predictedValue, trueValue)]

            if (treeModel.fullFeatureSet.data(treeModel.yIndex).Type == FeatureType.Numerical) {	// Regression Tree
                var valid_diff_predicted_value_and_true_value = diff_predicted_value_and_true_value.filter(v => (!v._2.equals("???")
                    && (Utility.parseDouble(v._3) match {
                        case Some(d: Double) => true
                        case None => false
                    }))) // filter invalid record, v._2 is predicted value

                val diff = (valid_diff_predicted_value_and_true_value
                    .map { case (index, predictedStringValue, trueStringValue) => (index, predictedStringValue.toDouble, trueStringValue.toDouble, 1) }
                    .map { case (index, pValue, tValue, counter) => (index, (pValue - tValue, (pValue - tValue) * (pValue - tValue), counter)) })

                val sums = diff.reduceByKey((x, y) => {
                    (x._1 + y._1, x._2 + y._2, x._3 + y._3)
                })

                val squareErrors = sums.map {
                    case (index, (sumDiff, sumDiffPower2, numInstances)) => (index, sumDiffPower2 / numInstances)
                }.collect

                squareErrors.foreach {
                    case (idx, mse) => {
                        var errors = mapTreeIndexToListErrorMetric.getOrElse(idx, List[Double]())
                        errors = errors.+:(mse)
                        mapTreeIndexToListErrorMetric = mapTreeIndexToListErrorMetric.updated(idx, errors)
                    }
                }

            } else { // Classification tree
                
                var valid_diff_predicted_value_and_true_value = 
                    diff_predicted_value_and_true_value.filter(v => (!v._2.equals("???"))) // filter invalid record, v._2 is predicted value

                val diff = (valid_diff_predicted_value_and_true_value
                    .map { case (index, predictedStringValue, trueStringValue) => 
                        (index, (if (predictedStringValue == trueStringValue) 1 else 0, 1)) })
                
                val sums = diff.reduceByKey((x, y) => {
                    (x._1 + y._1, x._2 + y._2)
                })

                val squareErrors = sums.map {
                    case (index, (truePrediction, numInstances)) => (index, truePrediction*1.0 / numInstances)
                }.collect

                squareErrors.foreach {
                    case (idx, err) => {
                        var errors = mapTreeIndexToListErrorMetric.getOrElse(idx, List[Double]())
                        errors = errors.+:(err)
                        mapTreeIndexToListErrorMetric = mapTreeIndexToListErrorMetric.updated(idx, errors)
                    }
                }
            }
            
            var valid_diff_predicted_value_and_true_value = diff_predicted_value_and_true_value.filter(v => (!v._2.equals("???") 
            && (Utility.parseDouble(v._3) match {
                case Some(d :Double) => true 
                case None => false 
            }))) // filter invalid record, v._2 is predicted value
            
            val diff = (valid_diff_predicted_value_and_true_value
                    .map{ case (index, predictedStringValue, trueStringValue) => (index, predictedStringValue.toDouble, trueStringValue.toDouble, 1)}
                    .map{ case (index, pValue, tValue, counter) => (index, (pValue - tValue, (pValue-tValue)*(pValue-tValue), counter))}
            	)

            val sums = diff.reduceByKey((x,y) => {
                (x._1 + y._1, x._2 + y._2, x._3 + y._3)
            })
            
            val squareErrors = sums.map { 
                case (index, (sumDiff, sumDiffPower2, numInstances)) => (index, sumDiffPower2/numInstances) 
            }.collect
            
            squareErrors.foreach {
                case (idx, mse) => {
                    var errors = mapTreeIndexToListErrorMetric.getOrElse(idx, List[Double]())
                    errors = errors.+:(mse)
                    mapTreeIndexToListErrorMetric = mapTreeIndexToListErrorMetric.updated(idx, errors)
                }
            }
            println("\n ==================== End round %d ====================\n".format(fold))
            
            datasetOfThisFold.unpersist(false)
        }	// END CROSS-VALIDATION
        
        var indexOfTreeHasMinAverageError = 0
        var minAverageError = Double.MaxValue
        
        // mapTreeIndexToListErrorMetric : Map< key: index-of-beta; value : list of errors>
        if (DEBUG)
            println("mapTreeIndexToListErrorMetric:")
            mapTreeIndexToListErrorMetric.foreach {
            case (key, value) => {
                println("index:" + key + " List of errors:" + value.mkString(","))
            }
        }
        
        val averageErrors = new Array[(Double, Double)](sequence_alpha_tree.length - 1)
        
        mapTreeIndexToListErrorMetric.foreach{
            case (key, value) => {
                var sumError : Double = 0.0
                var sumErrorPower2 : Double = 0.0
                var numElement = 0
                value.foreach(error => { 
                    numElement = numElement + 1; 
                    sumError = sumError + error
                    sumErrorPower2 = sumErrorPower2 + error*error
                    })
                val averageError = sumError/numElement
                if (averageError < minAverageError){
                    minAverageError = averageError
                    indexOfTreeHasMinAverageError = key
                }
                
                val EX = sumError/numElement;
                val EX2 = sumErrorPower2/numElement
                averageErrors(key) = (EX, math.sqrt((EX2 - EX*EX )/numElement))	// (averageError, SE)
                
            }
        }
        
        var indexOfFinalTree : Int = 0
        
        // Select best alpha by 1-SE Rule
        for (i <- (0 to averageErrors.length - 1)){
        	val (averageError, standardError) = averageErrors(i) 
                if (averageError <= averageErrors(indexOfTreeHasMinAverageError)._1 +  averageErrors(indexOfTreeHasMinAverageError)._2) {
                	indexOfFinalTree = i
                	println( "i:" + i +  " current average:" + averageError + " averageErrorOfMinTree:" + averageErrors(indexOfTreeHasMinAverageError))
                }
        }

        println("index of tree having min average error:" + indexOfTreeHasMinAverageError)
        println("min average: " + minAverageError)
        println("index of the final tree:" + indexOfFinalTree)
        if (averageErrors.length > 0)
        println("Error of final tree: (averageError,SE)=" + averageErrors(indexOfFinalTree))
        
        
        
        var leafNodesOfTheBestTree = sequence_alpha_tree(indexOfFinalTree)._1
        println("the final leaf nodes:" + leafNodesOfTheBestTree)
        
        //println("tree after pruning:" + 
        //pruneBranchs(treeModel.tree, leafNodesOfTheBestTree)
        //)
        
        //pruneBranch(treeModel., prunedNodeID)
        println("befor:" + treeModel.tree)
        treeModel.tree = pruneBranchs(treeModel.tree, leafNodesOfTheBestTree)
        println("pruned:" + treeModel.tree)
        
        treeModel
    }
    
    protected def pruneBranchs(root: Node, prunedNodeIDs: Set[BigInt]): Node
    
}

    import fr.eurecom.dsg.treelib.id3._
    import fr.eurecom.dsg.treelib.cart._
    import fr.eurecom.dsg.treelib.id3.PruningForID3._
    import fr.eurecom.dsg.treelib.cart.PruningForCART._
    
object Pruning {

    def Prune(treeModel : TreeModel, complexityParamter : Double, dataset: RDD[String], foldTimes : Int = 5) : TreeModel = {
        if (treeModel.isInstanceOf[ID3TreeModel])
            treeModel.asInstanceOf[ID3TreeModel].Prune(complexityParamter, dataset, foldTimes)
        else if (treeModel.isInstanceOf[CARTTreeModel])
            treeModel.asInstanceOf[CARTTreeModel].Prune(complexityParamter, dataset, foldTimes)
        else treeModel
    }
}
