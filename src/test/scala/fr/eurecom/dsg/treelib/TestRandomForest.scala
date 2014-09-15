package fr.eurecom.dsg.treelib.test

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import fr.eurecom.dsg.treelib._
import fr.eurecom.dsg.treelib.core._
import fr.eurecom.dsg.treelib.evaluation.Evaluation
import fr.eurecom.dsg.treelib.id3.ID3TreeBuilder

object TestRandomForest {
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

        val pathOfTreeModel = "/tmp/randomForest"
        val pathOfTheeFullTree = "/tmp/full-tree.model"
            
        var randomForest = new RandomForestBuilder()
	randomForest.setData(trainingData)
        randomForest.setNumberOfTree(30)
        val forest = randomForest.buildForest[ID3TreeBuilder]()
        
        println(forest)
        
        println("Evaluation:")
        val predictRDDOfTheFullTree = forest.predict(testingData)
        val actualValueRDD = testingData.map(line => line.split(',').last)
        //println("Original tree(full tree):\n%s".format(treeFromFile.treeModel))

        println("Evaluation of the forest:")
        Evaluation("misclassification").evaluate(predictRDDOfTheFullTree, actualValueRDD)

    }
}
