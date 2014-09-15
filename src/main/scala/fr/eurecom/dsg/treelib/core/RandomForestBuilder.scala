package fr.eurecom.dsg.treelib.core


import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import scala.collection.immutable.HashMap
import java.io._
import java.io.DataOutputStream
import java.io.FileOutputStream
import java.io.DataInputStream
import java.io.FileInputStream
import fr.eurecom.dsg.treelib.cart._
import scala.concurrent._
import scala.collection.immutable.Queue


/**
 * This class is representative for the model of forest
 */
class RandomForest (var numberOfTrees : Int) extends Serializable {
    
    /**
     * The directory which contains data of this forest
     */
    @transient private var forestPlace = "/tmp/" + System.nanoTime()
    
    /**
	 * Forest of trees
	 */
	private var trees = List[TreeModel]()//new Array[TreeModel](numberOfTrees)
	
	/**
	 * Set index-th tree
	 */
	def setTree(index : Int, treeModel: TreeModel) = {
	    //if (index >=0 && index < numberOfTrees)
	    	//trees.update(index, treeModel)
	        trees = trees :+ (treeModel)
	}
	
	def predictOne(input: String, delimeter: String = ",") : Any = {
	    var prediction_vote = new HashMap[String, Int]()

	    var maxVote = Int.MinValue
        var finalPrediction: Any = null
        var values = input.split(delimeter)

        //for (i <- 0 until numberOfTrees) {
        if (trees == null)
            println("TREE NULL")
            
        trees.foreach(
            tree => {
	        
            val prediction = tree.predict(values)
            var numVote = prediction_vote.getOrElse(prediction, 0) + 1

            prediction_vote = prediction_vote.updated(prediction, numVote)

            if (numVote > maxVote) {
                maxVote = numVote
                finalPrediction = prediction
            }
	    })	
	    
	    finalPrediction
	}
	
	/**
     * Predict value of the target feature base on the values of input features
     * 
     * @param testingData	the RDD of testing data
     * @return a RDD contain predicted values
     */
    def predict(testingData: RDD[String], 
            delimiter : String = ",", 
            ignoreBranchIDs : Set[BigInt] = Set[BigInt]()
            ) : RDD[String] = {
        
        testingData.map(line => RandomForest.this.predictOne(line).toString)
        //testingData.map(line => this.predict(line.split(delimiter))) 
    }
    
    override def toString : String = {
        "number of trees:%d\n%s".format(numberOfTrees, trees.mkString("\n"))
    }
	
	/***********************************************/
    /*    REGION WRITING AND LOADING MODEL    */
    /***********************************************/
    /**
     * Write the current tree model to file
     * 
     * @param path where we want to write to
     */
    def writeToFile(path: String) = {
        val ois = new ObjectOutputStream(new FileOutputStream(path))
        ois.writeObject(trees)
        ois.close()
    }

    /**
     * Load tree model from file
     *
     * @param path the location of file which contains tree model
     */
    def loadModelFromFile(path: String) = {
        //val js = new JavaSerializer(null, null)

        val ois = new ObjectInputStream(new FileInputStream(path)) {
            override def resolveClass(desc: java.io.ObjectStreamClass): Class[_] = {
                try { Class.forName(desc.getName, false, getClass.getClassLoader) }
                catch { case ex: ClassNotFoundException => super.resolveClass(desc) }
            }
        }

        var rt = ois.readObject().asInstanceOf[List[TreeModel]]
        //treeModel = rt
        //this.featureSet = treeModel.featureSet
        //this.usefulFeatureSet = treeModel.usefulFeatureSet
        trees = rt
        this.numberOfTrees = trees.length
        
        ois.close()
    }
}


/**
 * This class is used for building forest model
 */
class RandomForestBuilder {
    /**
     * The number of tree
     */
	private var numberOfTrees : Int = 100
	
	var MAXIMUM_PARALLEL_TREES = 2
	
	def setNumberOfTree(nTrees : Int) = {
	    numberOfTrees = nTrees;
	    forest = new RandomForest(numberOfTrees)
	}
	
	/**
	 * The number of random features, which will be use in each feature split
	 */
	var numberOfRandomFeatures : Int = 0
	
	private var minSplit : Int = 0
	
	/**
	 * Forest of trees
	 */
	var forest : RandomForest = new RandomForest(numberOfTrees)
	
	private var featureNames : Array[String] = null
	
	private var trainingData : RDD[String] = _

	/**
	 * Set training data, which will be used to build the forest
	 * @param	trainingData	the training data (without header)
	 */
	def setData(trainingData: RDD[String]) {
	    this.trainingData = trainingData
	}
	
	/**
	 * Because we didn't support included header in csv file, we use this function to set the features' name
	 * @param	fNames	the names of features
	 */
	def setFeatureName(fNames : Array[String]) = {
	    this.featureNames = fNames
	}
	
	
	def setMinSplit(m : Int) = {
		this.minSplit = m
	}
	
	/**
	 * Build the forest
     * @param yFeature 	name of target feature, the feature which we want to predict.
     * 					Default value is the name of the last feature
     * @param xFeatures set of names of features which will be used to predict the target feature
     * 					Default value is all features names, except target feature
	 */
	def buildForest[T <: TreeBuilder : ClassManifest](yFeature: String = "", 
        xFeatures: Set[Any] = Set[Any]()) : RandomForest = {
	    
	    var waitingTrees = Queue[Int]()
	    var finishedTrees = Queue[Int]()
	    
	    var numberRunningTrees = 0
	    var numberWaitingTrees = numberOfTrees
	    
	    
	    class ThreadTreeBuilder(treeID: Int, caller : RandomForestBuilder) extends Serializable with Runnable {
            @Override
            def run() {
                println("\n\n ====== Build tree " + treeID + " ============= \n\n")
                try{
                var tree: T = (implicitly[ClassManifest[T]]).erasure.newInstance.asInstanceOf[T]
                tree.useCache = true
                val samplingData = trainingData.sample(true, 1.0, System.nanoTime().toInt)
                //val obb = trainingData.subtract(samplingData)
                tree.setDataset(samplingData)
                tree.useRandomSubsetFeature = true
                //tree.setMinSplit(this.minSplit)

                if (caller.featureNames != null)
                    tree.setFeatureNames(caller.featureNames)

                val model = tree.buildTree(yFeature, xFeatures)
                //println(model)
                
                this.synchronized {
                    forest.setTree(treeID, model)
                    finishedTrees = finishedTrees.enqueue(treeID)
                }
                samplingData.unpersist(false)
                }
                catch {
                    case e : Throwable => {
                        println("Error when building tree " + treeID + ":\n" + e.getStackTraceString)
                    }
                    this.synchronized {
                    	finishedTrees = finishedTrees.enqueue(treeID)
                    }
                }
                
                System.gc()
                System.runFinalization()
                
            }
	    } // END CLASS ThreadTreeBuilder

	    
	    
	    for (i <- 0 until numberOfTrees)
	        waitingTrees = waitingTrees.enqueue(i)
	        
	    while (numberRunningTrees != 0 || numberWaitingTrees != 0) {
	    	System.gc()
	    	System.runFinalization()
	    	
	    	while (!finishedTrees.isEmpty){
	    	    finishedTrees.dequeue match {
	    	        case (treeID, xs) =>  {
	    	            println("Finish tree" + treeID)
	    	            numberRunningTrees = numberRunningTrees - 1
	    	            finishedTrees = xs
	    	        }
	    	    }
	    	}
	    	
	    	while (!waitingTrees.isEmpty && numberRunningTrees < MAXIMUM_PARALLEL_TREES){
	    	    println("numberRunningTrees:" + numberRunningTrees)
	    	    waitingTrees.dequeue match {
	    	        case (currentTreeID, xs) => {
	    	            numberRunningTrees = numberRunningTrees + 1
	    	            numberWaitingTrees = numberWaitingTrees - 1
	    	            waitingTrees = xs
	    	            println("launch tree " + currentTreeID)
	    	            launchTreeBuilder(currentTreeID, trainingData)
	    	        }
	    	    }
	    	}
	    	
	    	Thread sleep 1000
	    }
	    
	    
	    def launchTreeBuilder(treeID: Int, trainingData : RDD[String]) = {
	    	var thread = new Thread(new ThreadTreeBuilder(treeID, this))
	    	thread.start()
	    	//thread.join()
	    }
	    /*
	    for (i <- 0 until numberOfTrees) {
	        System.gc()
            System.runFinalization()
	        println("\n\n ====== Build tree " + i + " ============= \n\n")
	        var tree : T = (implicitly[ClassManifest[T]]).erasure.newInstance.asInstanceOf[T]
	        tree.useCache = true
	        val samplingData = trainingData.sample(true, 1.0, System.nanoTime().toInt)
	        val obb = trainingData.subtract(samplingData)
	        tree.setDataset(samplingData)
	        tree.useRandomSubsetFeature = true
	        //tree.setMinSplit(this.minSplit)
	        
	        if (this.featureNames != null)
	        	tree.setFeatureNames(this.featureNames)
	        forest.setTree(i, tree.buildTree(yFeature, xFeatures))
	        samplingData.unpersist(true)
	    }
	    *
	    * 
	    */
	    forest
	}
	
}
