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
import scala.util.Random
import fr.eurecom.dsg.treelib.cart._

/**
 * Abstract class of tree builder
 *
 * @param featureSet		all features in the training data
 * @param usefulFeatureSet	the features which we used to build the tree (included the target feature)
 */
abstract class TreeBuilder extends Serializable {

    protected val DEBUG: Boolean = true

    /**
     * Temporary model file
     */
    val temporaryModelFile = "/tmp/model.temp"

    /********************************************************/
    /*    REGION OF MAIN COMPONENTS    */
    /********************************************************/

    /**
     * Contains raw information about features, which will be used to construct a feature set
     */
    //private var metadata: Array[String] = Array[String]()
    private var headerOfDataset = Array[String]()

    /**
     * Contains information of all features in data
     */
    var fullFeatureSet = new FeatureSet()

    /**
     * The number of features which will be used for building tree (include the target feature)
     */
    //var usefulFeatureSet = new FeatureSet()
	//var usefulFeatures : Set[Int] = null	// means all features will be used for building tree
    
    protected var numberOfUsefulFeatures : Int = fullFeatureSet.numberOfFeature
    
    
    /**
     * Tree model
     */
    var treeModel: TreeModel = new CARTTreeModel()

    /**
     * Default value of the index of the target feature.
     * Here this value is the index of the last feature
     */
    var yIndexDefault = fullFeatureSet.numberOfFeature - 1 // = number_of_feature - 1

    /**
     *  index of target feature,
     *  default value is the index of the last feature in dataset
     */
    protected var yIndex = yIndexDefault

    /**
     *  the indices/indexes of X features, which will be used to predict the target feature
     *  this variable can be infered from featureSet and yIndex
     *  but because it will be used in functions processLine, and buidingTree
     *  so we don't want to calculate it multiple time
     *  The default value is the index of all features, except the last one
     */
    protected var xIndexes = fullFeatureSet.data.map(x => x.index).filter(x => (x != yIndex)).toSet[Int]

    /**
     * A value , which is used to marked a split point is invalid
     */
    protected val ERROR_SPLITPOINT_VALUE = ",,,@,,,"

    /**
     * The data will be used to build tree
     */
    var trainingData: RDD[String] = null;

    /**
     * In each node split, do we choose splitpoint on the  random subset of features ?
     * This argument is often used when building tree with Random Forest
     */
    var useRandomSubsetFeature = false

    /**
     * Cache the dataset
     */
    var useCache = true

    /*****************************************************************/
    /*    REGION OF PARAMETERS   */
    /*****************************************************************/

    /**
     *  minimum records to do a splitting, default value is 10
     */
    var minsplit = 10

    /**
     *  delimiter of fields in data set, default value is ","
     */
    var delimiter = ','

    /**
     *  coefficient of variation, default value is 0.1
     */
    var threshold: Double = 0.1

    /**
     * Max depth of the tree, default value if 30
     */
    protected var maxDepth: Int = 62

    /**
     * The maximum complexity of the tree
     */
    protected var maximumComplexity = 0.001

    def setDelimiter(c: Char) = {
        delimiter = c
    }

    /**
     * Set the minimum records of splitting
     * It's mean if a node have the number of records <= minsplit, it can't be splitted anymore
     *
     * @param xMinSplit	new minimum records for splitting
     */
    def setMinSplit(xMinSplit: Int) = {
        this.minsplit = xMinSplit
        this.treeModel.minsplit = xMinSplit
    }

    /**
     * Set threshold for stopping criterion. This threshold is coefficient of variation
     * A node will stop expand if Dev(Y)/E(Y) < threshold
     * In which:
     * Dev(Y) is standard deviation
     * E(Y) is medium of Y
     *
     * @param xThreshold	new threshold
     */
    def setThreshold(xThreshlod: Double) = {
        threshold = xThreshlod
        treeModel.threshold = xThreshlod
    }

    /**
     * Set the maximum of depth of tree
     */
    def setMaxDepth(value: Int) = {
        this.maxDepth = value
        this.treeModel.maxDepth = value
    }

    def setMaximumComplexity(cp: Double) = {
        this.maximumComplexity = cp
        this.treeModel.maximumComplexity = cp
    }

    /*****************************************************************/
    /*    REGION OF UTILITIES FUNCTIONS   */
    /*****************************************************************/

    /**
     * Convert feature name (or feature info) to theirs index
     * @param	xNames	Set of feature names or set of FeatureInformation
     * @param 	yName	Name of the target feature
     * @output	A tuple with the first component is the set of indices of predictors, the second
     * component is the index of the target feature
     */
    private def getXIndexesAndYIndexByNames(xNames: Set[Any], yName: String): (Set[Int], Int) = {

        var yindex = fullFeatureSet.getIndex(yName)
        if (yName == "" && yindex < 0)
            yindex = this.yIndexDefault

        if (yindex < 0)
            throw new Exception("ERROR: Can not find attribute `" + yName + "` in (" + fullFeatureSet.data.map(f => f.Name).mkString(",") + ")")
        // index of features, which will be used to predict the target feature
        var xindexes =
            if (xNames.isEmpty) // if user didn't specify xFeature, we will process on all feature, exclude Y feature (to check stop criterion)
                fullFeatureSet.data.filter(_.index != yindex).map(x => x.index).toSet[Int]
            else //xNames.map(x => featureSet.getIndex(x)) //+ yindex
            {
                xNames.map(x => {
                    var index = x match {
                        case Feature(name, ftype, _) => {
                            val idx = fullFeatureSet.getIndex(name)
                            fullFeatureSet.update(Feature(name, ftype, idx), idx)
                            idx
                        }
                        case s: String => {
                            fullFeatureSet.getIndex(s)
                        }
                        case _ => { throw new Exception("Invalid feature. Expect as.String(feature_name) or as.Number(feature_name) or \"feature_name\"") }
                    }
                    if (index < 0)
                        throw new Exception("Could not find feature " + x)
                    else
                        index
                })
            }

        (xindexes, yindex)

    }

    /**
     * From the full training set, remove the unused columns
     * @param	trainingData	the training data
     * @param	xIndexes		the set of indices of predictors
     * @param	yIndex			the index of the target feature
     * @param	removeInvalidRecord	remove line which contains invalid feature values or not
     * @output	the new RDD which will be use "directly" in building phase
     */
    /*
    def filterUnusedFeatures(trainingData: RDD[String], xIndexes: Set[Int], yIndex: Int, removeInvalidRecord: Boolean = true): RDD[String] = {
        var i = 0
        var j = 0
        var temp = trainingData.map(line => {
            var array = line.split(this.delimiter)

            i = 0
            j = 0
            var newLine = ""
            try {
                array.foreach(element => {
                    if (yIndex == i || xIndexes.contains(i)) {
                        if (newLine.equals(""))
                            newLine = element
                        else {
                            newLine = "%s,%s".format(newLine, element)
                        }
                        if (removeInvalidRecord) {
                            this.usefulFeatureSet.data(j).Type match {
                                case FeatureType.Categorical => element
                                case FeatureType.Numerical => element.toDouble
                            }
                        }

                        j = j + 1
                    }
                    i = i + 1
                })
                newLine
            } catch {
                case _: Throwable => ""
            }
        })

        temp.filter(line => !line.equals(""))
    }
	*/
    /**
     * Convert index of the useful features into index in the full feature set
     * @param	featureSet			the full feature set
     * @param	usefulFeatureSet	the useful feature set
     * @param	a tuple with the first component is the set of indices of predictors
     * the second component is the index of the target feature
     */
    /*
    private def mapFromUsefulIndexToOriginalIndex(featureSet: FeatureSet, usefulFeatureSet: FeatureSet): (Set[Int], Int) = {
        var xIndexes = treeModel.xIndexes.map(index => treeModel.fullFeatureSet.getIndex(treeModel.usefulFeatureSet.data(index).Name))
        var yIndex = treeModel.fullFeatureSet.getIndex(usefulFeatureSet.data(treeModel.yIndex).Name)
        (xIndexes, yIndex)
    }
	*/
    /**
     * Get node by nodeID
     * @param	id node id
     * @output	node
     */
    protected def getNodeByID(id: BigInt): CARTNode = {
        if (id != 0) {
            val level = (Math.log(id.toDouble) / Math.log(2)).toInt
            var i: Int = level - 1
            var TWO: BigInt = 2
            var parent = treeModel.tree.asInstanceOf[CARTNode]; // start adding from root node
            try {
                while (i >= 0) {

                    if ((id / (TWO << i - 1)) % 2 == 0) {
                        // go to the left
                        parent = parent.left
                    } else {
                        // go go the right
                        parent = parent.right
                    }
                    i -= 1
                } // end while
            } catch {
                case e: Throwable => {
                    e.printStackTrace()
                    if (DEBUG) println("currentID:" + id)
                    if (DEBUG) println("currentTree:\n" + treeModel.tree)
                    throw e
                }
            }

            parent
        } else {
            null
        }
    }

    /*********************************************************************/
    /*    REGION FUNCTIONS OF BUILDING PHASE    */
    /*********************************************************************/

    /**
     * This function is used to build the tree
     *
     * @param yFeature 	name of target feature, the feature which we want to predict.
     * 					Default value is the name of the last feature
     * @param xFeatures set of names of features which will be used to predict the target feature
     * 					Default value is all features names, except target feature
     * @return <code>TreeModel</code> the root of tree
     * @see TreeModel
     */
    def buildTree(yFeature: String = "",
        xFeatures: Set[Any] = Set[Any]()): TreeModel = {

        if (this.trainingData == null) {
            throw new Exception("ERROR: Dataset can not be null.Set dataset first")
        }
        if (yIndexDefault < 0) {
            throw new Exception("ERROR:Dataset is invalid or invalid feature names")
        }

      try

        treeModel.tree = null
        // These information will be used in Pruning phase
        treeModel.xFeatures = xFeatures
        treeModel.yFeature = yFeature

        var (xIndexes, yIndex) = this.getXIndexesAndYIndexByNames(xFeatures, yFeature)
        println("current yIndex=" + yIndex + " xIndex:" + xIndexes.toString)

        // SET UP LIST OF USEFUL FEATURES AND ITS INDEXES //
        var usefulFeatureList = List[Feature]()
        var i = -1
        var usefulIndexes = List[Int]()
        var newYIndex = 0
        fullFeatureSet.data.foreach(feature => {
          if (xIndexes.contains(feature.index) || feature.index == yIndex) {
            i = i + 1
            if (feature.index == yIndex) {
              newYIndex = i
              println("new yindex:" + newYIndex)
            }
            usefulIndexes = usefulIndexes.:+(i)
            usefulFeatureList = usefulFeatureList.:+(Feature(feature.Name, feature.Type, i))
          }
        })

        //this.usefulFeatureSet = new FeatureSet(usefulFeatureList)
        /*
        // FILTER OUT THE UNUSED FEATURES //
        this.trainingData = filterUnusedFeatures(this.trainingData, xIndexes, yIndex)

        // because we remove unused features, so the indices are changed
        var newXIndexes = usefulIndexes.filter(x => x != newYIndex)

        this.usefulFeatureSet = new FeatureSet(usefulFeatureList)

        this.yIndex = newYIndex
        this.xIndexes = newXIndexes.toSet

        treeModel.yIndex = newYIndex
        treeModel.xIndexes = this.xIndexes
        treeModel.usefulFeatureSet = this.usefulFeatureSet
        treeModel.fullFeatureSet = this.fullFeatureSet
        println("build tree with feature set:" + this.usefulFeatureSet + "\n xIndexes:" + this.xIndexes + "\nYIndex:" + this.yIndex)
        treeModel.treeBuilder = this
        // build tree
        if (this.useCache)
            this.startBuildTree(this.trainingData.cache, newXIndexes.toSet, newYIndex)
        else
            this.startBuildTree(this.trainingData, newXIndexes.toSet, newYIndex)
        */

        this.yIndex = yIndex
        this.xIndexes = xIndexes
        treeModel.xIndexes = xIndexes
        treeModel.yIndex = yIndex
        treeModel.fullFeatureSet = this.fullFeatureSet
        treeModel.treeBuilder = this
        this.numberOfUsefulFeatures = this.xIndexes.size + 1

        println("xIndexes:" + xIndexes + " yIndex:" + yIndex)

        println("Building tree with predictors:" + this.xIndexes.map(i => fullFeatureSet.data(i).Name))
        println("Target feature:" + fullFeatureSet.data(yIndex).Name)

        //this.startBuildTree(this.trainingData, xIndexes, yIndex)

        // TODO: the use of cache here is "wrong", should cache RDD only when useful. See also the unpersist method.
        if (this.useCache)
          this.startBuildTree(this.trainingData.cache, xIndexes, yIndex)
        else
          this.startBuildTree(this.trainingData, xIndexes, yIndex)

      catch {
        case e: Throwable => {
          println("Error:" + e.getStackTraceString)
        }
      }
        this.trainingData.unpersist(true)
        this.treeModel
    }

    /**
     * This function is used to build the tree
     *
     * @param yFeature 	name of target feature, the feature which we want to predict.
     * @param xFeatures set of names of features which will be used to predict the target feature
     * @return <code>TreeModel</code> the root of tree
     * @see TreeModel
     */
    protected def startBuildTree(trainingData: RDD[String],
        xIndexes: Set[Int],
        yIndex: Int): Unit

    protected def getPredictedValue(info: StatisticalInformation): Any = {
        info.YValue
    }

    protected def updateModel(info: Array[(BigInt, SplitPoint, StatisticalInformation)], isStopNode: Boolean = false) = {
        info.foreach(stoppedRegion =>
            {

                var (label, splitPoint, statisticalInformation) = stoppedRegion

                if (DEBUG) println("update model with label=%d splitPoint:%s".format(
                    label,
                    splitPoint))

                var newnode = (
                    if (isStopNode) {
                        new CARTLeafNode(splitPoint.point.toString)
                    } else {
                        val chosenFeatureInfoCandidate = fullFeatureSet.data.find(f => f.index == splitPoint.index)
                        chosenFeatureInfoCandidate match {
                            case Some(chosenFeatureInfo) => {
                                new CARTNonLeafNode(chosenFeatureInfo,
                                    splitPoint,
                                    new CARTLeafNode("empty.left"),
                                    new CARTLeafNode("empty.right"));
                            }
                            case None => { new CARTLeafNode(this.ERROR_SPLITPOINT_VALUE) }
                        }
                    }) // end of assign value for new node

                if (newnode.value == this.ERROR_SPLITPOINT_VALUE) {
                    println("Value of job id=" + label + " is invalid")
                } else {

                    if (!isStopNode) { // update predicted value for non-leaf node     
                        newnode.value = getPredictedValue(statisticalInformation)
                    }

                    newnode.statisticalInformation = statisticalInformation

                    if (DEBUG) println("create node with statistical infor:" + statisticalInformation + "\n new node:" + newnode.value)

                    // If tree has zero node, create a root node
                    if (treeModel.isEmpty) {
                        treeModel.tree = newnode;

                    } else //  add new node to current model
                    {
                        var parent = getNodeByID(label >> 1)
                        if (label % 2 == 0) {
                            parent.setLeft(newnode)
                        } else {
                            parent.setRight(newnode)
                        }
                    }
                }
            })
    }

    /************************************************************/
    /*  REGION: SET DATASET AND METADATA */
    /************************************************************/

    /**
     * Set the training dataset to used for building tree
     *
     * @param trainingData	the training dataset
     * @throw Exception if the dataset contains less than 2 lines
     */
    def setDataset(trainingData: RDD[String]) {

        var firstLine = trainingData.take(1)

        // If we can not get 2 first line of dataset, it's invalid dataset
        if (firstLine.length < 1) {
            throw new Exception("ERROR:Invalid dataset")
        } else {
            this.trainingData = trainingData;

            /*
            var header = firstLine(0)
            var temp_header = header.split(delimiter)

            if (hasHeader) {
                this.headerOfDataset = temp_header
            } else {
                var i = -1;
                this.headerOfDataset = temp_header.map(v => { i = i + 1; "Column" + i })
            }
			*/
            // determine types of features automatically
            // Get the second line of dataset and try to parse each of them to double
            // If we can parse, it's a numerical feature, otherwise, it's a categorical feature
            var sampleData = firstLine(0).split(delimiter);
            var i = -1;
            this.headerOfDataset = sampleData.map(v => { i = i + 1; "Column" + i })

            i = 0
            var listOfFeatures = List[Feature]()

            // if we can parse value of a feature into double, this feature may be a numerical feature
            sampleData.foreach(v => {
                Utility.parseDouble(v.trim()) match {
                    case Some(d) => { listOfFeatures = listOfFeatures.:+(Feature(headerOfDataset(i), FeatureType.Numerical, i)) }
                    case None => { listOfFeatures = listOfFeatures.:+(Feature(headerOfDataset(i), FeatureType.Categorical, i)) }
                }
                i = i + 1
            })

            // update the dataset
            fullFeatureSet = new FeatureSet(listOfFeatures)
            updateFeatureSet()

        }

    }

    /**
     * Set feature names
     *
     * @param names a line contains names of features,
     * 				separated by by a delimiter, which you set before (default value is comma ',')
     *     			Example: Temperature,Age,"Type"
     * @throw Exception if the training set is never be set before
     */
    def setFeatureNames(names: Array[String]) = {
        if (this.trainingData == null)
            throw new Exception("Trainingset is null. Set dataset first")
        else {
            if (names.length != fullFeatureSet.data.length) {
                throw new Exception("Incorrect names")
            }

            var i = 0
            names.foreach(name => {
                fullFeatureSet.data(i).Name = name
                fullFeatureSet.update(fullFeatureSet.data(i), i)
                i = i + 1
            })
            updateFeatureSet()
        }
    }

    /**
     * Update the feature set based on the information of metadata
     */
    private def updateFeatureSet() = {

        yIndexDefault = fullFeatureSet.numberOfFeature - 1
        println("Set new yIndexDefault = " + yIndexDefault)
        //treeBuilder = new ThreadTreeBuilder(featureSet);
        //treeBuilder = treeBuilder.createNewInstance(featureSet, usefulFeatureSet)
    }
    /* END REGION DATASET AND METADATA */

    /************************************************/
    /*    REGION FUNCTIONS OF PREDICTION MAKING    */
    /************************************************/
    /**
     * Predict value of the target feature base on the values of input features
     *
     * @param record	an array, which its each element is a value of each input feature (already remove unused features)
     * @return predicted value or '???' if input record is invalid
     */
    private def predictOnPreciseData(record: Array[String], ignoreBranchIDs: Set[BigInt]): String = {
        try {
            treeModel.predict(record, ignoreBranchIDs)
        } catch {
            case e: Exception => { println("Error P: " + e.getStackTraceString); "???" }
        }
    }

    def predictOneInstance(record: Array[String], ignoreBranchIDs: Set[BigInt] = Set[BigInt]()): String = {
        if (record.length == 0)
            "???"
        else {
            /*
            var (xIndexes, yIndex) = mapFromUsefulIndexToOriginalIndex(fullFeatureSet, usefulFeatureSet)
            var newRecord: Array[String] = Array[String]()
            var i = 0
            for (field <- record) {
                if (i == yIndex || xIndexes.contains(i)) {
                    newRecord = newRecord.:+(field)
                }
                i = i + 1
            }

            predictOnPreciseData(newRecord, ignoreBranchIDs)
            * 
            */
            predictOnPreciseData(record, ignoreBranchIDs)
        }
    }

    /**
     * Predict value of the target feature base on the values of input features
     *
     * @param testingData	the RDD of testing data
     * @return a RDD contain predicted values
     */
    def predict(testingData: RDD[String],
        delimiter: String = ",",
        ignoreBranchIDs: Set[BigInt] = Set[BigInt]()): RDD[String] = {
        /*
         * var (xIndexes, yIndex) = mapFromUsefulIndexToOriginalIndex(fullFeatureSet, usefulFeatureSet)
        var newTestingData = filterUnusedFeatures(testingData, xIndexes, yIndex, false)
        newTestingData.map(line => this.predictOnPreciseData(line.split(delimiter), ignoreBranchIDs))
        */
        testingData.map(line => this.predictOnPreciseData(line.split(delimiter), ignoreBranchIDs))
    }

    /***********************************************/
    /*    REGION WRITING AND LOADING MODEL    */
    /***********************************************/
    /**
     * Write the current tree model to file
     *
     * @param path where we want to write to
     */
    def writeModelToFile(path: String) = {
        val ois = new ObjectOutputStream(new FileOutputStream(path))
        ois.writeObject(treeModel)
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

        var rt = ois.readObject().asInstanceOf[TreeModel]
        //treeModel = rt
        //this.featureSet = treeModel.featureSet
        //this.usefulFeatureSet = treeModel.usefulFeatureSet
        setTreeModel(rt)

        ois.close()
    }

    def setTreeModel(tm: TreeModel) = {
        this.treeModel = tm
        this.fullFeatureSet = tm.fullFeatureSet
        //this.usefulFeatureSet = tm.usefulFeatureSet
        //this.usefulFeatures = tm.usefulFeatures
        updateFeatureSet()
    }

    /**
     * Recover, repair and continue build tree from the last state
     */
    def continueFromIncompleteModel(trainingData: RDD[String], path_to_model: String): TreeModel = {
        loadModelFromFile(path_to_model)
        this.treeModel = treeModel
        this.fullFeatureSet = treeModel.fullFeatureSet
        //this.usefulFeatureSet = treeModel.usefulFeatureSet
        //var (xIndexes, yIndex) = mapFromUsefulIndexToOriginalIndex(fullFeatureSet, usefulFeatureSet)
        //var newtrainingData = filterUnusedFeatures(trainingData, xIndexes, yIndex)

        if (treeModel == null) {
            throw new Exception("The tree model is empty because of no building. Please build it first")
        }

        if (treeModel.isComplete) {
            println("This model is already complete")
        } else {
            println("Recover from the last state")
            /* INITIALIZE */
            this.fullFeatureSet = treeModel.fullFeatureSet
            //this.usefulFeatureSet = treeModel.usefulFeatureSet
            this.xIndexes = treeModel.xIndexes
            this.yIndex = treeModel.yIndex

            startBuildTree(trainingData, xIndexes, yIndex)

        }

        treeModel
    }

    def createNewInstance() : TreeBuilder
}
