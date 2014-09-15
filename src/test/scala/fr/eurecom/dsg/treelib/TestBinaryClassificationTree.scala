package fr.eurecom.dsg.treelib

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._

import fr.eurecom.dsg.treelib._
import fr.eurecom.dsg.treelib.core._
import fr.eurecom.dsg.treelib.evaluation.Evaluation
import fr.eurecom.dsg.treelib.cart.RegressionTree
import fr.eurecom.dsg.treelib.cart.ClassificationTree
import fr.eurecom.dsg.treelib.utils._

object TestBinaryClassificationTree {
  def main(args : Array[String]) : Unit = {
    val IS_LOCAL = true

    val inputTrainingFile = (
      if (IS_LOCAL)
        "/Users/michiard/work/teaching/bitbucket/spark-test/data/playgolf.csv"
      else
        "hdfs://spark-master-001:8020/user/ubuntu/input/AIRLINES/training/*")

    val inputTestingFile = (
      if (IS_LOCAL)
        "/Users/michiard/work/teaching/bitbucket/spark-test/data/playgolf.csv"
      else
        "hdfs://spark-master-001:8020/user/ubuntu/input/AIRLINES/testing/*")

    val conf = (
      if (IS_LOCAL)
        new SparkConf()
          .setMaster("local").setAppName("test classification tree")
      else
        new SparkConf()
          .setMaster("spark://spark-master-001:7077")
          .setAppName("rtree example")
          .setSparkHome("/opt/spark")
          .setJars(List("target/scala-2.10/rtree-example_2.10-1.0.jar"))
          .set("spark.executor.memory", "2222m"))

    val context = new SparkContext(conf)

    var stime: Long = 0

    val trainingData = context.textFile(inputTrainingFile, 1)
    val testingData = context.textFile(inputTestingFile, 1)

    val pathOfTreeModel = "/Users/michiard/work/teaching/bitbucket/spark-test/tmp-output/tree.model"
    val pathOfTheeFullTree = "/Users/michiard/work/teaching/bitbucket/spark-test/tmp-output/full-tree.model"

    val tree = new ClassificationTree()
    tree.setDataset(trainingData)
    tree.setMinSplit(1)
    //println("Tree:\n" + tree.buildTree("PlayGolf", Set("Temperature", "Outlook")))
    println("Tree:\n" + tree.buildTree())

    /* TEST WRITING TREE TO MODEL */
    tree.writeModelToFile(pathOfTheeFullTree)

    /* TEST PRUNING */
    println("Final tree:\n%s".format(Pruning.Prune(tree.treeModel, 0.01, trainingData)))

    /* TEST LOADING TREE FROM MODEL FILE */
    val treeFromFile = new RegressionTree()
    try {
      treeFromFile.loadModelFromFile(pathOfTheeFullTree)
      println("OK: Load tree from '%s' successfully".format(pathOfTreeModel))
    } catch {
      case e: Throwable => {
        println("ERROR: Couldn't load tree from '%s'".format(pathOfTreeModel))
        e.printStackTrace()
      }
    }

    println("Evaluation:")
    val predictRDDOfTheFullTree = treeFromFile.predict(testingData)
    val predictRDDOfThePrunedTree = tree.predict(testingData)
    val actualValueRDD = testingData.map(line => line.split(',').last)
    //println("Original tree(full tree):\n%s".format(treeFromFile.treeModel))

    println("Original Tree:\n%s".format(treeFromFile.treeModel))
    println("Evaluation of the full tree:")
    Evaluation("misclassification").evaluate(predictRDDOfTheFullTree, actualValueRDD)

    println("Pruned Tree:\n%s".format(tree.treeModel))
    println("Evaluation of the pruned tree:")
    Evaluation("misclassification").evaluate(predictRDDOfThePrunedTree, actualValueRDD)
  }
}
