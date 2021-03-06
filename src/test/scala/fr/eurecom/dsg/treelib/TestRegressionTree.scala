package fr.eurecom.dsg.treelib.test

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import fr.eurecom.dsg.treelib._
import fr.eurecom.dsg.treelib.cart.RegressionTree
import fr.eurecom.dsg.treelib.evaluation.Evaluation
import fr.eurecom.dsg.treelib.core._
import scala.collection.immutable._
import fr.eurecom.dsg.treelib.utils._

import com.esotericsoftware.kryo.Kryo
import org.apache.spark.serializer.KryoRegistrator

object TestRegressionTree {
    def main(args: Array[String]): Unit = {

        val IS_LOCAL = true

        //System.setProperty("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        //System.setProperty("spark.kryo.registrator", "MyRegistrator")

        val inputTrainingFile = (
            if (IS_LOCAL)
                "/Users/michiard/work/teaching/bitbucket/spark-test/data/training-bodyfat.csv"
            else
                "hdfs://spark-master-001:8020/user/ubuntu/input/AIRLINES/training/*")

        val inputTestingFile = (
            if (IS_LOCAL)
                "/Users/michiard/work/teaching/bitbucket/spark-test/data/testing-bodyfat.csv"
            else
                "hdfs://spark-master-001:8020/user/ubuntu/input/AIRLINES/testing/*")

        val conf = (
            if (IS_LOCAL)
                new SparkConf()
                .setMaster("local").setAppName("rtree example")
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

        val pathOfTreeModel = "/tmp/tree.model"
        val pathOfTheeFullTree = "/tmp/full-tree.model"

        /* TEST BUILDING TREE */

        val tree = new RegressionTree()
        tree.setDataset(trainingData)
        tree.setFeatureNames(Array[String]("","age","DEXfat","waistcirc","hipcirc","elbowbreadth","kneebreadth","anthro3a","anthro3b","anthro3c","anthro4"))

        if (IS_LOCAL) {
            tree.setMinSplit(10)
            //tree.treeBuilder.setMaxDepth(2)

            stime = System.nanoTime()
            println(tree.buildTree("DEXfat", Set("age", "waistcirc", "hipcirc", "elbowbreadth", "kneebreadth")))
            println("\nOK: Build tree in %f second(s)".format((System.nanoTime() - stime) / 1e9))

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
            
            /* TEST PREDICTING AND EVALUATION */
            println("Evaluation:")
            val predictRDDOfTheFullTree = treeFromFile.predict(testingData)
            val predictRDDOfThePrunedTree = tree.predict(testingData)
            val actualValueRDD = testingData.map(line => line.split(',')(2)) // 2 is the index of DEXfat in csv file, based 0
            //println("Original tree(full tree):\n%s".format(treeFromFile.treeModel))

            println("Original Tree:\n%s".format(treeFromFile.treeModel))
            println("Evaluation of the full tree:")
            Evaluation.evaluate(predictRDDOfTheFullTree, actualValueRDD)

            println("Pruned Tree:\n%s".format(tree.treeModel))
            println("Evaluation of the pruned tree:")
            Evaluation.evaluate(predictRDDOfThePrunedTree, actualValueRDD)

            /* TEST RECOVER MODE */
            //val recoverTree = new RegressionTree()
            //recoverTree.treeBuilder = new DataMarkerTreeBuilder(new FeatureSet(Array[String]()))
            //recoverTree.continueFromIncompleteModel(bodyfat_data, "/tmp/model.temp3") // temporary model file
            //println("Model after re-run from the last state:\n" + recoverTree.treeModel)

        } else {
            tree.setMinSplit(100)
            tree.setMaximumComplexity(0.003)
            //tree.treeBuilder.setThreshold(0.3) // coefficient of variation
            //tree.treeBuilder.setMaxDepth(10)

            /* TEST BUILDING */
            stime = System.nanoTime()
            println(tree.buildTree("ArrDelay",
                Set(as.String("Month"), as.String("DayofMonth"), as.String("DayOfWeek"), "CRSDepTime",
                    "UniqueCarrier", "Origin", "Dest", "Distance")))
            println("\nBuild tree in %f second(s)".format((System.nanoTime() - stime) / 1e9))

            /* TEST WRITING TREE TO MODEL */
            tree.writeModelToFile(pathOfTheeFullTree)

            /* TEST PRUNING */
            stime = System.nanoTime()
            println("Final tree:\n%s".format(Pruning.Prune(tree.treeModel, 0.01, trainingData, 5)))
            tree.writeModelToFile(pathOfTreeModel)
            println("\nPrune tree in %f second(s)".format((System.nanoTime() - stime) / 1e9))

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

            /* TEST PREDICTING AND EVALUATION */
            println("Evaluation:")
            val predictRDDOfTheFullTree = treeFromFile.predict(testingData)
            val predictRDDOfThePrunedTree = tree.predict(testingData)
            val actualValueRDD = testingData.map(line => line.split(',')(14)) // 14 is the index of ArrDelay in csv file, based 0
            //println("Original tree(full tree):\n%s".format(treeFromFile.treeModel))

            println("Original Tree:\n%s".format(treeFromFile.treeModel))
            println("Evaluation of the full tree:")
            Evaluation.evaluate(predictRDDOfTheFullTree, actualValueRDD)

            println("Pruned Tree:\n%s".format(tree.treeModel))
            println("Evaluation of the pruned tree:")
            Evaluation.evaluate(predictRDDOfThePrunedTree, actualValueRDD)

            /* TEST RECOVER MODE */
            //val recoverTree = new RegressionTree()
            //recoverTree.treeBuilder = new DataMarkerTreeBuilder(new FeatureSet(Array[String]()))
            //recoverTree.continueFromIncompleteModel(bodyfat_data, "/tmp/model.temp3") // temporary model file
            //println("Model after re-run from the last state:\n" + recoverTree.treeModel)
        }
    }
}
