package fr.eurecom.dsg.treelib.id3


import fr.eurecom.dsg.treelib.core._
import fr.eurecom.dsg.treelib.cart._
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import fr.eurecom.dsg.treelib.utils._


object PruningForID3 {
	implicit class ID3Pruning(treeModel: ID3TreeModel) extends AbstractPruning(treeModel) {
	    private var numberOfInstancesTotal = Int.MaxValue
        
        private def risk(node: Node): Double = {
            // in case of Regression Tree
            if (node.statisticalInformation.YValue.isInstanceOf[Double]) {
                val EX = node.statisticalInformation.YValue.asInstanceOf[Double] / node.statisticalInformation.numberOfInstances
                val EX2 = node.statisticalInformation.sumOfYPower2 / node.statisticalInformation.numberOfInstances
                //math.sqrt(EX2 - EX*EX)*math.sqrt(node.statisticalInformation.numberOfInstances)///node.statisticalInformation.numberOfInstances 	// MSE
                (EX2 - EX * EX) * (node.statisticalInformation.numberOfInstances)
            } else { // case of Classification tree, we calculate the misclasification rate
            	val numInstances = node.statisticalInformation.numberOfInstances
                var temp = node.statisticalInformation.YValue.asInstanceOf[Array[(Any, Int)]]
                if (temp.length > 0) {
                    return (numInstances - temp.head._2)//*1.0 /numberOfInstancesTotal
                }
                //1
            	numInstances
            }
        }

        /**
         *  filter useless pruned nodes
         *  for example, if this set contain 3 and 7;
         *  3 is the parent of 7, we don't need to keep 7 in this set
         */
        override protected def standardizePrunedNodeSet(nodes: Set[BigInt]): Set[BigInt] = {
            var result = nodes
            nodes.toArray.sortWith((a, b) => a < b).foreach(
                id => {
                    id
                })
            result

        }

        /**
         * Prune the full tree to get tree T1 which has the same error rate but smaller size
         */
        override protected def firstPruneTree(root: Node): Set[BigInt] = {
            numberOfInstancesTotal = root.statisticalInformation.numberOfInstances
            var leafNodes = Set[BigInt]()
            def isLeaf(node: Node, id: BigInt) = {
                node.isLeaf || leafNodes.contains(id)
            }

            def firstPruneTreeIter(node: ID3Node): Unit = {

                if (!node.isLeaf) {
                    var everyChildIsLeaf = true
                    for (i <- 0 until node.numberOfChildren) {
                        firstPruneTreeIter(node.children(i).asInstanceOf[ID3Node])
                        everyChildIsLeaf = everyChildIsLeaf & isLeaf(node.children(i), node.children(i).asInstanceOf[ID3Node].id)
                    }

                    var riskOfChildren: Double = 0.0

                    for (i <- 0 until node.numberOfChildren) {
                        riskOfChildren = riskOfChildren + risk(node.children(i))
                    }

                    val R = risk(node)
                    if (R == riskOfChildren)
                        leafNodes = leafNodes + node.asInstanceOf[ID3Node].id

                }
            }

            firstPruneTreeIter(root.asInstanceOf[ID3Node])

            leafNodes
        }

        /**
         * Calculate g(t) of every node
         *
         * @return (id_of_node_will_be_pruned, alpha)
         */
        override def selectNodesToPrune(root: Node, already_pruned_node: Set[BigInt]): (Set[BigInt], Double) = {
            var result = Set[BigInt]()
            var currentMin: Double = Double.MaxValue
            var currentID: BigInt = 0

            def selectNodeToPrunIter(node: ID3Node): (Double, Int) = { // (riskOfBranch, numberChildren)
                val id = node.id
                //println("currentID:" + id + " info:" + node.statisticalInformation + " is belong to leaf node set:" + already_pruned_node.contains(id))
                // if this is an internal node, and the branch of this node hasn't pruned
                if (!node.isLeaf && !already_pruned_node.contains(id)) {

                    var riskOfChildren: Double = 0
                    var numLeafNodeInBranch: Int = 0
                    for (i <- 0 until node.numberOfChildren) {
                        val (risk, numLeafNode) = selectNodeToPrunIter(node.children(i).asInstanceOf[ID3Node])
                        riskOfChildren = riskOfChildren + risk
                        numLeafNodeInBranch = numLeafNodeInBranch + numLeafNode
                    }

                    var gT = (risk(node) - riskOfChildren) / (numLeafNodeInBranch - 1)

                    //println("gt=" + gT + "r(t)=" + risk(node) + " riskChildren:" + riskOfChildren + " numLeafNodes:" + numLeafNodeInBranch)
                    if (gT == currentMin) {
                        result = result + id
                    } else if (gT < currentMin) {
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

            if (!already_pruned_node.contains(1)) {
                //println("prunedNode:" + already_pruned_node + " root node:" + root)
                
                selectNodeToPrunIter(root.asInstanceOf[ID3Node])
                
            }

            //println("result:" + result)
            // union with the previous set of pruned nodes
            result = result ++ already_pruned_node;
            var pruned_node_list = standardizePrunedNodeSet(result)

            //println("final result" + pruned_node_list)
            //println("number of pruned nodes:" + pruned_node_list.size)

            (pruned_node_list, currentMin)
        }

        /**
         * Prune tree with a set of leaf nodes
         * @param	root			root of the tree
         * @param	prunedNodeIds	the nodes will become leaves
         */
        override protected def pruneBranchs(root: Node, prunedNodeIDs: Set[BigInt]): Node = {
            def pruneBranchsIter(node: ID3Node): Node = {
            if (!node.isLeaf) {
                if (prunedNodeIDs.contains(node.id)) {
                    node.toLeafNode
                } else {
                    for (i <- 0 until node.numberOfChildren) {
                        node.setChildren(i, pruneBranchsIter(node.children(i).asInstanceOf[ID3Node]))    
                    }
                    node
                }
            } else {
                node
            }
        }

            pruneBranchsIter(root.asInstanceOf[ID3Node])
        }
	}
}