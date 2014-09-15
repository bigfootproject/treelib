package fr.eurecom.dsg.treelib.id3

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import fr.eurecom.dsg.treelib.core._
import scala.Array.canBuildFrom
import scala.math.BigInt.int2bigInt
import scala.collection.mutable.HashMap
import scala.util.Random

class ID3TreeBuilder
    extends TreeBuilder {

    //override var treeModel : CTreeModel = new CTreeModel()

    def LOG2 = math.log(2)

    class FeatureValue(
        var label: BigInt,
        var featureIndex: Int,
        var xValue: Any,
        var yValue: Any,
        var frequency: Int) extends Serializable {
        override def toString = {
            "(%s,%s,%s,%s,%s)".format(label, featureIndex, xValue, yValue, frequency)
        }

    }

    private var updatingLabelMap: HashMap[BigInt, HashMap[(Int, Any), BigInt]] = HashMap[BigInt, HashMap[(Int, Any), BigInt]]()
    // update (oldLabel-> (splitPointFeatureIndex, splitPointValue) -> newLabel)

    private var mapFullIndexToUsefulIndex: Array[Int] = null

    treeModel = new ID3TreeModel()

    def convertArrayValuesToObjects(values: Array[String]): Array[FeatureValue] = {
        var i: Int = -1
        var nodeID: BigInt = 1
        var result = Array.fill[FeatureValue](this.numberOfUsefulFeatures)(null)
        var count = 0
        // only encapsulate the value of predictor and the target feature (or the value of the useful features)
        values.foreach { value =>
            {
                i = i + 1
                if (this.xIndexes.contains(i) || (i == yIndex)) {
                    if (i != yIndex && fullFeatureSet.data(i).Type == FeatureType.Numerical)
                        result.update(count, new FeatureValue(nodeID, i, value.toDouble, values(this.yIndex), 1))
                    else
                        result.update(count, new FeatureValue(nodeID, i, value, values(this.yIndex), 1))
                    count = count + 1
                }
            }
        }
        result
    }

    private def generateRandomSet(sequenceOfFIndices: Seq[Int]): Array[Int] = {
        val numFeatures = sequenceOfFIndices.length
        val numRandomFeatureSelection = (math.sqrt(numFeatures) + 0.5).toInt

        var selectedFeature = Array.fill(numRandomFeatureSelection)(0)
        var arrayOfIndices = sequenceOfFIndices.toArray

        for (i <- 0 until numRandomFeatureSelection) {
            val j = Random.nextInt(numFeatures - i) + i
            selectedFeature.update(i, arrayOfIndices(j))

            // swap element at index i and j
            arrayOfIndices = arrayOfIndices.updated(j, arrayOfIndices(i))
            arrayOfIndices = arrayOfIndices.updated(i, selectedFeature(i))
        }
        selectedFeature
    }

    /**
     * Start the main part of algorithm
     *
     * @param trainingData	the input data
     * @return TreeModel the root node of tree
     * @see TreeModel
     */
    override def startBuildTree(trainingData: RDD[String],
        xIndexes: Set[Int],
        yIndex: Int): Unit = {
        println("Start building tree")
        fullFeatureSet.update(Feature(fullFeatureSet.data(yIndex).Name, FeatureType.Categorical, yIndex), yIndex)

        mapFullIndexToUsefulIndex = Array.fill[Int](fullFeatureSet.numberOfFeature)(-1)
        var count = 0
        for (i <- 0 until fullFeatureSet.numberOfFeature) {
            val feature = fullFeatureSet.data(i)
            if (this.xIndexes.contains(feature.index) || feature.index == yIndex) {
                mapFullIndexToUsefulIndex.update(i, count)
                count = count + 1
            }
        }
        var rootID = 1

        var expandingNodeIndexes = Set[BigInt]()

        var map_label_to_splitpoint = Map[BigInt, SplitPoint]()

        // parse raw data
        val mydata = trainingData.map(line => line.split(delimiter))

        /* REGION TRANSFORMING */

        // encapsulate each value of each feature in each line into a object
        var transformedData = mydata.map(
            arrayValues => {
                convertArrayValuesToObjects(arrayValues)
            })

        // filter the 'line' which contains the invalid or missing data
        transformedData = transformedData.filter(x => (x.length > 0))
        
        var stop = false

        while (!stop) {
            System.gc
            System.runFinalization()
            // Calculate gain of feature
            var featureValueCount = transformedData.flatMap(x => x.toSeq).
                filter(x => (x.label >= 0 && x.featureIndex >= 0)).map {
                    case featureValue => ((featureValue.label, featureValue.featureIndex, featureValue.xValue, featureValue.yValue), featureValue.frequency)
                }.reduceByKey(_ + _)
            // we get: 
            //     	((1,1,weak,yes),	2)
            //		((1,1,weak,no), 	6)
            //		((1,1,strong,yes),	3)
            //		((1,1,strong,no),	3)
            //		((1,2,hot,yes), 	4)
            //		((1,2,hot,no), 		1)
            //		((1,2,cool,yes), 	6)
            //		((1,2,cool,no), 	3)

            //println("feature value count:" + featureValueCount.collect.mkString("\n"))
            var gains = {
                var featureValueCountGroupByXValue =
                    featureValueCount.map {
                        case ((nodeID, featureIndex, xValue, yValue), frequency) => {
                            ((nodeID, featureIndex, xValue), (yValue, frequency))
                        }

                    }.groupByKey
                // we get:
                //		((1,1,weak), 	[(yes,2), (no,6)])
                //		((1,1,strong), 	[(yes,3), (no,3)])
                //		((1,2,hot),		[(yes,4), (no,1)])
                //		((1,2,cool),	[(yes,6), (no,3)])

                // if (the situation is RandomForest, and) we need to select random subset of features
                if (this.useRandomSubsetFeature) {
                    val temp = featureValueCountGroupByXValue.map(x => (x._1._1, x._1._2)).groupByKey // (nodeid, fIndex)

                    val randomSelectedFeatureAtEachNode = temp.flatMap {
                        case (label, sequenceOfFIndices) => {
                            generateRandomSet(sequenceOfFIndices.toList.distinct).map(x => (label, x))
                        }
                    }.collect.toSet

                    featureValueCountGroupByXValue = featureValueCountGroupByXValue.filter(x => randomSelectedFeatureAtEachNode.contains((x._1._1, x._1._2))) // contain (nodeid,fIndex)
                }

                val categoricalFeatures = featureValueCountGroupByXValue.filter(x => fullFeatureSet.data(x._1._2).Type == FeatureType.Categorical)
                val numericalFeatures = featureValueCountGroupByXValue.filter(x => fullFeatureSet.data(x._1._2).Type == FeatureType.Numerical)

                val gainOfCategoricalFeatures =
                    (
                        categoricalFeatures.map {
                            case ((nodeID, featureIndex, xValue), sequence_yvalue_fre) => {
                                var numRec: Int = (sequence_yvalue_fre.foldLeft(0)((x, y) => x + y._2))
                                var g = 0.0
                                sequence_yvalue_fre.foreach {
                                    case (yValue, frequency) => {
                                        g = g + frequency * math.log(frequency * 1.0 / numRec) / LOG2
                                        //println("(y,fre,g,numrec)=(%s,%d,%f,%d)".format(yValue.toString, frequency, g,numRec))
                                    }
                                }

                                //val topK = sequence_yvalue_fre.sortBy(x => -x._2).take(3) // sort desc by frequency and select top

                                //val statisticalInformation = new StatisticalInformation(topK)

                                //println("nodeid :%s,fIndex:%s,xValue:%s,yValue:%s,gain:%f, numrec:%d".format(
                                //        nodeID,featureIndex,xValue,sequence_yvalue_fre.mkString(","), g, numRec))
                                ((nodeID, featureIndex), (Set[Any](xValue), g)) // we need 'xValue' to determine how many children of the splitting
                            }
                        }

                        // we get:
                        //		((1,1), set(weak), __)	// of value 'weak'
                        //		((1,1), set(strong), __)	// of value 'strong'
                        //		((1,2),	set(hot), __)	// of value 'hot'
                        //		((1,2),	set(cool), __)	// of value 'cool'

                        .reduceByKey { case ((xvalue1, gain1), (xvalue2, gain2)) => { (xvalue1.union(xvalue2), gain1 + gain2) } }
                        )

                // we get:
                //		((1,1), set(weak, strong),__)	// 'gain*' of feature index 1 
                //		((1,2), set(hot,cool),__)	// 'gain*' of feature index 2
                // gain* is a modification of 'gain'; We only transform from theoretical formulation to implementation formulation

                val gainOfNumericalFeatures = (
                    numericalFeatures.map {
                        case ((nodeID, featureIndex, xValue), seqYValue_Frequency) => ((nodeID, featureIndex), (xValue, seqYValue_Frequency))
                    }
                    .groupByKey
                    // we get:
                    // (3,1, [(12.5, [(yes, 10), (no,3)]) , (14, [(yes,10), (no, 40)]), (8, [(yes,5), (no, 3)]) ])
                    // (3,1, [(9.5, [(yes, 30), (no,13)]) , (8.4, [(yes,1), (no, 5)]), (20, [(yes,4), (no, 3)]) ])
                    .map {
                        case ((nodeID, featureIndex), seqXValue_YValue_Frequency) => {
                            if (seqXValue_YValue_Frequency.size > 1) {
                                val (bestSplitPoint, maxGain) = findBestSplitPointOfNumericalFeature(seqXValue_YValue_Frequency)
                                ((nodeID, featureIndex), (Set[Any](bestSplitPoint), maxGain))
                            } else { // if the number of XValue is 1 or 0, this is a leaf node
                                ((nodeID, -1), (null, Double.MinValue))
                            }
                        }
                    })
                //println("gain of numerical features:" + gainOfNumericalFeatures.collect.mkString(","))
                val result = gainOfCategoricalFeatures.union(gainOfNumericalFeatures)
                //numericalFeatures.unpersist(false)
                //categoricalFeatures.unpersist(false)
                //gainOfCategoricalFeatures.unpersist(false)
                //featureValueCountGroupByXValue.unpersist(false)
                result

            }

            //println("gains:\n" + gains.collect.mkString("\n"))

            // In each node, Select top 3 yvalue by its frequency
            val statisticalInformationEachNode = featureValueCount.filter { case ((_, fIndex, _, _), _) => fIndex == yIndex }.map {
                case ((nodeID, featureIndex, xValue, yValue), frequency) => {
                    (nodeID, (yValue, frequency))
                }

            }.groupByKey.map {
                case (nodeID, seqOfYValue_freq) => {
                    var numInstances = seqOfYValue_freq.foldLeft(0)((x, y) => x + y._2)
                    (nodeID, new StatisticalInformation(seqOfYValue_freq.toList.sortBy(x => -x._2).take(3).toArray, 0, numInstances))
                }
            }.collectAsMap

            //println ("statisticalInformation each node:" + statisticalInformationEachNode.mkString(","))
            val bestSplittedFeatureOfEachNode = gains.map {
                case ((nodeID, fIndex), xValues_gain) => (nodeID, (fIndex, xValues_gain))
            }.groupByKey
                .map {
                    case (nodeID, seqIndex_XValueGains) => {
                        //println("%s %s".format(nodeID,seqIndex_XValueGains.mkString(",")))
                        val temp = seqIndex_XValueGains.filter(x => (x._1 != yIndex && x._1 >= 0))
                        if (temp.isEmpty) { // if the predictor set is empty -> create a leaf node
                            (nodeID, new SplitPoint(-1, null, 0), statisticalInformationEachNode.getOrElse(nodeID, null))
                        } else {
                            val (splittingFeature, xValue_gain) = seqIndex_XValueGains.filter(x => x._1 != yIndex).maxBy(x => x._2._2) // max by gain
                            if (fullFeatureSet.data(splittingFeature).Type == FeatureType.Categorical)
                                (nodeID, new SplitPoint(splittingFeature, xValue_gain._1.toArray.sortBy(x => x.asInstanceOf[String]), xValue_gain._2), statisticalInformationEachNode.getOrElse(nodeID, null))
                            else
                                (nodeID, new SplitPoint(splittingFeature, xValue_gain._1.toArray.head, xValue_gain._2), statisticalInformationEachNode.getOrElse(nodeID, null))

                        }
                    }
                }.collect

            val nonStopNode = bestSplittedFeatureOfEachNode.filter {
                case (nodeID, splitpoint, ydistribution) => {
                    //splitpoint.index != -1
                    ydistribution != null && ydistribution.YValue.asInstanceOf[Array[_]].length > 1
                }
            }

            // we get:
            //		(1, splitPointOfNode1, y_value_distribution)
            // 		(2, splitPointOfNode2, y_value_distribution)

            updateModel(bestSplittedFeatureOfEachNode)
            transformedData = updateLabels(transformedData, updatingLabelMap.clone)
            //println("current Tree:" + treeModel)

            stop = (nonStopNode.length == 0)

            //gains.unpersist(true)
            //featureValueCount.unpersist(true)
            //System.gc()
            //System.runFinalization()
        }
        
        mydata.unpersist(true)
        updatingLabelMap = null
        treeModel.isComplete = true

    }

    private def findBestSplitPointOfNumericalFeature(seqXValue_YValue_Frequency: Iterable[(Any, Iterable[(Any, Int)])]): (Double, Double) = {
        var newSeqXValue_YValue_Frequency = seqXValue_YValue_Frequency.toList.sortBy(x => x._1.asInstanceOf[Double]) // sort by xValue
        var mapYValueToFrequency = (seqXValue_YValue_Frequency.flatMap(x => x._2).groupBy(_._1)
            .map { case (group, traversable) => traversable.reduce { (a, b) => (a._1, a._2 + b._2) } })
        val mapYValueToIndex = mapYValueToFrequency.keys.zipWithIndex.map(x => (x._1 -> x._2)).toMap
        val numberOfYValue = mapYValueToIndex.size
        var totalFrequencyOfEachYValue = Array.fill(numberOfYValue)(0)
        var frequencyOfYValueInLeftNode = Array.fill(numberOfYValue)(0)
        var sumOfFrequency: Int = 0
        var sumFrequencyLeft: Int = 0
        var sumFrequencyRight: Int = 0

        mapYValueToFrequency.foreach {
            x =>
                {
                    totalFrequencyOfEachYValue.update(mapYValueToIndex.getOrElse(x._1, -1), x._2)
                    sumOfFrequency = sumOfFrequency + x._2
                }
        }

        var lastXValue: Double = 0
        var splitPoint: Double = 0
        var maxGain = Double.MinValue

        sumFrequencyRight = sumOfFrequency

        // we don't need to consider the case: "there is only 1 XValue -> create leaf node"
        // because in that case, we set the splitpoint is 0.0 and the gain is Double.minValue
        // -> the selected feature (which has max gain) will be another feature.
        // if the "another feature" is the target feature -> create leaf node when calculating "bestSplittedFeatureOfEachNode"
        // in function startBuildTree
        for (i <- 0 until newSeqXValue_YValue_Frequency.length - 1) {
            val (xValue, seqYValue_Frequency) = newSeqXValue_YValue_Frequency(i)
            val nextXValue = newSeqXValue_YValue_Frequency(i + 1)._1

            val splitPointCandidate = (xValue.asInstanceOf[Double] + nextXValue.asInstanceOf[Double]) / 2

            seqYValue_Frequency.foreach(
                x => {
                    var targetIndex = mapYValueToIndex.getOrElse(x._1, -1)
                    frequencyOfYValueInLeftNode.update(targetIndex, frequencyOfYValueInLeftNode(targetIndex) + x._2)
                    totalFrequencyOfEachYValue.update(targetIndex, totalFrequencyOfEachYValue(targetIndex) - x._2)
                    sumFrequencyLeft = sumFrequencyLeft + x._2
                    sumFrequencyRight = sumFrequencyRight - x._2
                })
            //println("frequencyLeft:" + frequencyOfYValueInLeftNode.mkString(",") + " sumLeft:" + sumFrequencyLeft)
            //println("frequencyRight:" + totalFrequencyOfEachYValue.mkString(",") + " sumRight:" + sumFrequencyRight)

            var g: Double = 0
            for (j <- 0 until numberOfYValue) {
                if (frequencyOfYValueInLeftNode(j) != 0 && sumFrequencyLeft != 0)
                    g = g + frequencyOfYValueInLeftNode(j) * math.log(frequencyOfYValueInLeftNode(j) * 1.0 / sumFrequencyLeft)

                if (totalFrequencyOfEachYValue(j) != 0 && sumFrequencyRight != 0)
                    g = g + totalFrequencyOfEachYValue(j) * math.log(totalFrequencyOfEachYValue(j) * 1.0 / sumFrequencyRight)
            }

            //println("consider splitpoint:" + splitPointCandidate + " gain:" + g + " maxGain:" + maxGain + " slitpoint:" + splitPoint)
            if (g > maxGain) {
                maxGain = g
                splitPoint = splitPointCandidate
            }
        }

        (splitPoint, maxGain)
        //.reduce((x,y) => (x._1, x._2 + y._2))
    }

    override protected def getPredictedValue(info: StatisticalInformation): Any = {
        info.YValue.asInstanceOf[Array[(_, _)]].head._1
    }

    override protected def updateModel(info: Array[(BigInt, SplitPoint, StatisticalInformation)], isStopNode: Boolean = false) = {

        updatingLabelMap = HashMap[BigInt, HashMap[(Int, Any), BigInt]]()

        info.sortBy(x => x._1).foreach(stoppedRegion =>
            {

                var (label, splitPoint, statisticalInformation) = stoppedRegion

                //if (DEBUG) println("update model with label=%d splitPoint:%s".format(
                //    label,
                //    splitPoint))

                var newnode = treeModel.asInstanceOf[ID3TreeModel].createNode(splitPoint) //new ID3Node(splitPoint)
                newnode.statisticalInformation = statisticalInformation
                newnode.value = getPredictedValue(statisticalInformation)
                val numberOfObservation = statisticalInformation.YValue.asInstanceOf[Array[(_, Int)]].foldLeft(0)((x, y) => x + y._2)
                //println("DEBUG: statisticalInfo:" + statisticalInformation.YValue.asInstanceOf[Array[_]].mkString(",") )

                //if (splitPoint.index == -1)	// stopnode/leaf node because the target feature has only 1 value
                if (statisticalInformation.YValue.asInstanceOf[Array[_]].length <= 1
                    || splitPoint.index == -1
                    || (numberOfObservation <= this.minsplit)) {
                    //println("this is a stop node")
                    newnode.isLeaf = true
                } else {
                    newnode.isLeaf = false
                    val chosenFeatureInfoCandidate = fullFeatureSet.data.find(f => f.index == splitPoint.index)

                    chosenFeatureInfoCandidate match {
                        case Some(chosenFeatureInfo) => {
                            newnode.feature = chosenFeatureInfo
                        }
                        case None => {
                            newnode.value = this.ERROR_SPLITPOINT_VALUE
                        }
                    }
                }

                if (newnode.value == this.ERROR_SPLITPOINT_VALUE) {
                    println("Value of job id=" + label + " is invalid")
                } else {

                    //if (DEBUG) println("create node with statistical infor:" + statisticalInformation + "\n new node:" + newnode.value)

                    if (label == 1) // root node
                    {
                        newnode.id = 1
                        treeModel.tree = newnode
                    } else {
                        var (tempNode, positionInParent) = treeModel.asInstanceOf[ID3TreeModel].findNodeByID(label)
                        if (tempNode != null) {
                            newnode.parent = tempNode.parent
                            newnode.id = tempNode.id
                            newnode.parent.setChildren(positionInParent, newnode)
                            //println("node " + newnode.id + " isleaf:" + newnode.isLeaf)
                        } else {
                            println("Can not find node id=" + label)
                        }
                    }

                    var mapSplitPointToNewLabel = HashMap[(Int, Any), BigInt]()
                    if (!newnode.isLeaf) { // prepare to expand the intermediate nodes

                        if (splitPoint.point.isInstanceOf[Array[_]]) // splitpoint of categorical feature
                            for (i <- 0 until splitPoint.point.asInstanceOf[Array[_]].length) {
                                var newTempChild = treeModel.asInstanceOf[ID3TreeModel].createNode() //new ID3Node()

                                newnode.addChild(newTempChild)
                                if (splitPoint.index != -1) // not a leaf node
                                    mapSplitPointToNewLabel = mapSplitPointToNewLabel.+=((splitPoint.index, splitPoint.point.asInstanceOf[Array[_]](i)) -> newTempChild.id)

                            }
                        else if (splitPoint.point.isInstanceOf[Double]) {

                            var leftChild = treeModel.asInstanceOf[ID3TreeModel].createNode()
                            newnode.addChild(leftChild)
                            var rightChild = treeModel.asInstanceOf[ID3TreeModel].createNode()
                            newnode.addChild(rightChild)

                            if (splitPoint.index != -1) // not a leaf node
                                mapSplitPointToNewLabel = mapSplitPointToNewLabel.+=((splitPoint.index, splitPoint.point) -> leftChild.id)
                        }

                        updatingLabelMap = updatingLabelMap.+=(label -> mapSplitPointToNewLabel)

                        //println("updateLabelMap:" + updatingLabelMap.mkString(","))
                    }

                }
            })
    }

    private def updateLabels(data: RDD[Array[FeatureValue]], updatingLabelMap: HashMap[BigInt, HashMap[(Int, Any), BigInt]]) = {
        val updatingLabelMap_broadcast = data.context.broadcast(updatingLabelMap)
        //println("data before:")
        //data.foreach (x => println(x.mkString(",")))
        val newdata = data.map(array => {
            //println("before:" + array.mkString(","))
            val currentLabel = array(0).label
            if (currentLabel >= 0) {

                val newLabels = updatingLabelMap_broadcast.value.getOrElse(currentLabel, null)
                //println("current label:" + currentLabel + " newLabel:" + newLabels)

                // if the current record is belong to a leaf node
                if (newLabels == null || newLabels.isEmpty) {
                    //println("could not find the updating of label " + currentLabel)
                    array.foreach(element => { element.label = -9 })
                } else {
                    val splittedFeatureIndex = newLabels.head._1._1
                    var currentXValueAtSplittedFeature = array(mapFullIndexToUsefulIndex(splittedFeatureIndex)).xValue

                    val newLabel = this.fullFeatureSet.data(splittedFeatureIndex).Type match {
                        case FeatureType.Categorical => {
                            val newLabel = newLabels.getOrElse((splittedFeatureIndex, currentXValueAtSplittedFeature), BigInt(-9))
                            if (newLabel == -9) {
                                println("splitFeatureIndex:" + splittedFeatureIndex + " currentXValueAtSplittedFeature:" + currentXValueAtSplittedFeature)
                            }

                            newLabel
                        }

                        case FeatureType.Numerical => {
                            val splitPointValue = newLabels.head._1._2.asInstanceOf[Double]
                            var newLabel = newLabels.head._2 // new label for the left node
                            if (currentXValueAtSplittedFeature.asInstanceOf[Double] > splitPointValue) {
                                newLabel = newLabel + 1 // if this record is belong to the right node, the new label = labelLeft + 1
                            }
                            //array.foreach(element => element.label = newLabel)
                            newLabel
                        }
                    }

                    for (i <- 0 until array.length) {
                        if (i == mapFullIndexToUsefulIndex(splittedFeatureIndex) && fullFeatureSet.data(splittedFeatureIndex).Type == FeatureType.Categorical)
                            array(i).featureIndex = -1
                        array(i).label = newLabel
                    }

                }
                //println(array.mkString(","))
            }
            array

        })
        data.unpersist(false)

        newdata
    }

    /*
    private def updateModel(nodeID : Int, featureID: Int, splitpoint : SplitPoint , isLeafNode : Boolean = false) {
        //if (this.treeModel)
        var newNode : ID3Node = 
            if (isLeafNode) {
                var rs = new ID3Node()
                rs.isLeaf = true
                rs
            }
            else {
                var rs = new ID3Node()
                rs.isLeaf = false
                rs
            }
        
        
        if (this.treeModel.isEmpty){
            //this.treeModel.tree = newNode
        }
        
        val chosenFeature = this.usefulFeatureSet.data(featureID)
        
        
    }
    * 
    */

    override def createNewInstance: TreeBuilder = {
        new ID3TreeBuilder()
    }
}