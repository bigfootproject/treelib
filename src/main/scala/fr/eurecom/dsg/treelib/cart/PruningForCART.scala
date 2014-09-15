package fr.eurecom.dsg.treelib.cart

import fr.eurecom.dsg.treelib.core._
import fr.eurecom.dsg.treelib.cart._
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import fr.eurecom.dsg.treelib.utils.AbstractPruning

object PruningForCART {
    implicit class CARTPruning(treeModel: CARTTreeModel) extends AbstractPruning(treeModel) {

        private var numberOfInstancesTotal = Int.MaxValue
        
        private def risk(node: CARTNode): Double = {
            // in case of Regression Tree
            if (node.statisticalInformation.YValue.isInstanceOf[Double]) {
                val EX = node.statisticalInformation.YValue.asInstanceOf[Double] / node.statisticalInformation.numberOfInstances
                val EX2 = node.statisticalInformation.sumOfYPower2 / node.statisticalInformation.numberOfInstances
                //math.sqrt(EX2 - EX*EX)*math.sqrt(node.statisticalInformation.numberOfInstances)///node.statisticalInformation.numberOfInstances 	// MSE
                (EX2 - EX * EX) * (node.statisticalInformation.numberOfInstances)
            } else { // case of Classification tree, we calculate the misclasification rate
                //println("yvalue:" + node.statisticalInformation.YValue + " id3=" + node.isInstanceOf[ID3Node] + " node:" + node)
                //println("test:" + node.statisticalInformation.YValue.asInstanceOf[Array[(Any, Int)]].mkString(","))
                val numInstances = node.statisticalInformation.numberOfInstances
                var temp = node.statisticalInformation.YValue.asInstanceOf[Array[(Any, Int)]]
                if (temp.length > 0) {
                    return (numInstances - temp.head._2)*1.0 /numberOfInstancesTotal
                }
                1
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
                    var tmp = id
                    while (tmp > 0) {
                        if (result.contains(tmp >> 1)) {
                            result = result - id
                            tmp = 0
                        }
                        tmp = tmp >> 1
                    }
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

            def firstPruneTreeIter(node: CARTNode, currentId: BigInt): Unit = {

                if (!node.isLeaf) {
                    firstPruneTreeIter(node.left, currentId << 1)
                    firstPruneTreeIter(node.right, (currentId << 1) + 1)
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

            firstPruneTreeIter(root.asInstanceOf[CARTNode], 1)

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

            def selectNodeToPrunIter(node: CARTNode, id: BigInt): (Double, Int) = { // (riskOfBranch, numberChildren)
                println("currentID:" + id + " info:" + node.statisticalInformation)
                // if this is an internal node, and the branch of this node hasn't pruned
                if (!node.isLeaf && !already_pruned_node.contains(id)) {

                    var (riskBranchLeft, numLeft) = selectNodeToPrunIter(node.left, (id << 1))
                    var (riskBranchRight, numRight) = selectNodeToPrunIter(node.right, (id << 1) + 1)
                    var gT = (risk(node) - (riskBranchLeft + riskBranchRight)) / (numLeft + numRight - 1)
                    println("Node " + id + " r(t)=" + risk(node) + "  g(t)=" + gT +
                        " (numLeft:%d,gLeft=%f) (numRight=%d, gRight=%f)".format(numLeft, riskBranchLeft, numRight, riskBranchRight))

                    if (gT == currentMin) {
                        result = result + id
                    } else if (gT < currentMin) {
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

            if (!already_pruned_node.contains(1)) {
                println("root node:" + root)
                
                selectNodeToPrunIter(root.asInstanceOf[CARTNode], 1)
                
            }

            //println("result:" + result)
            // union with the previous set of pruned nodes
            result = result ++ already_pruned_node;
            var pruned_node_list = standardizePrunedNodeSet(result)

            println("final result select node to prune of CART" + pruned_node_list)

            (pruned_node_list, currentMin)
        }

        /**
         * Prune tree with a set of leaf nodes
         * @param	root			root of the tree
         * @param	prunedNodeIds	the nodes will become leaves
         */
        override protected def pruneBranchs(root: Node, prunedNodeIDs: Set[BigInt]): Node = {
            def pruneBranchsIter(node: CARTNode, id: BigInt): CARTNode = {
                if (!node.isLeaf) {
                    if (prunedNodeIDs.contains(id)) {
                        node.toLeafNode.asInstanceOf[CARTNode]
                    } else {
                        node.setLeft(pruneBranchsIter(node.left, id << 1))
                        node.setRight(pruneBranchsIter(node.right, (id << 1) + 1))
                        node
                    }
                } else {
                    node
                }
            }

            pruneBranchsIter(root.asInstanceOf[CARTNode], 1)
        }
    }
}
