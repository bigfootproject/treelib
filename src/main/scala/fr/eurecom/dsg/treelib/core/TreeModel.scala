package fr.eurecom.dsg.treelib.core

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import java.io._
import java.io.DataOutputStream
import java.io.FileOutputStream

abstract class TreeModel extends Serializable {
    
    /**
     *  The index of target feature
     */ 
	var yIndex : Int = -1;
	
	/**
	 *  The indexes of the features which were used to predict target feature
	 */ 
	var xIndexes : Set[Int] = Set[Int]()
	
	/**
	 *  all features information
	 */ 
	var fullFeatureSet : FeatureSet = new FeatureSet(List[Feature]())

	/**
	 * The features which were used to build the tree, they were re-indexed from featureSet
	 */
	//var usefulFeatureSet : FeatureSet = new FeatureSet(List[Feature]())
	var usefulFeatures : Set[Int] = null
	
	/**
	 *  the root node of the tree
	 */ 
	var tree : Node = null
	
	var minsplit : Int = 10
	var threshold : Double = 0.01
	var maximumComplexity : Double = 0.01
	var maxDepth : Int = 63
	var yFeature: String = ""
	var xFeatures : Set[Any] = Set[Any]()
	
	/***
	 * Is the tree empty ?
	 */
	def isEmpty() = (tree == null)
	
	/**
	 *  Is the tree is build completely
	 */ 
	var isComplete = false
	
	
	/**
	 * The tree builder which created this model
	 */
	var treeBuilder : TreeBuilder = null
	
	
	/******************************************************/
	/*    REGION FUNCTIONS    */
	/******************************************************/
	/**
     * Predict Y base on input features
     * 
     * @param record			an array, which its each element is a value of each input feature
     * @param ignoreBranchSet	a set of branch ID, which won't be used to predict (only use it's root node)
     * @return a predicted value
     * @throw Exception if the tree is empty
     */
    def predict(record: Array[String], ignoreBranchSet: Set[BigInt] = Set[BigInt]()): String
    
    /**
     * Evaluate the accuracy of regression tree
     * @param input	an input record (uses the same delimiter with trained data set)
     * @param delimiter the delimiter of training data
     */
    def evaluate(input: RDD[String], delimiter : Char = ',')
	
	/**
     * Write the current tree model to file
     * 
     * @param path where we want to write to
     */
    def writeToFile(path: String) = {
        val oos = new ObjectOutputStream(new FileOutputStream(path))
        oos.writeObject(this)
        oos.close
    }
}
