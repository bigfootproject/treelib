package fr.eurecom.dsg.treelib.cart

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import scala.concurrent._
import java.io._
import fr.eurecom.dsg.treelib.core._
import scala.Array.canBuildFrom
import scala.math.BigInt.int2bigInt

/**
 * This class is used to build a regression tree
 */
class RegressionTree() extends TreeBuilder {

    /**
     * This class is representative for each value of each feature in the data set
     * @param index Index of the feature in the whole data set, based zero
     * @param xValue Value of the current feature
     * @param yValue Value of the Y feature associated (target, predicted feature)
     * @param frequency Frequency of this value
     */
    class FeatureValueLabelAggregate(var index: Int, var xValue: Any, var yValue: Double, var yValuePower2: Double, var frequency: Int, var label: BigInt = 1)
        extends Serializable {

        /**
         * Sum two FeatureValueAggregates (sum two yValues and two frequencies)
         */
        def +(that: FeatureValueLabelAggregate) = {
            new FeatureValueLabelAggregate(this.index, this.xValue,
                this.yValue + that.yValue,
                this.yValuePower2 + that.yValuePower2,
                this.frequency + that.frequency,
                this.label)
        }

        override def toString() =
            "Feature(index:%d | xValue:%f | yValue:%f | frequency:%d | label:%d)".format(
                index, xValue, yValue, frequency, label)
        //"Feature(index:" + index + " | xValue:" + xValue +
        //" | yValue" + yValue + " | frequency:" + frequency + " | label:" + label + ")";
    }
    
    
    
    
    type AggregateInfo = (Any, Double, Double, Int) // (xValue, yValue, yValuePower2, frequency)

    var regions = List[(BigInt, List[Condition])]()
    
    /*****************************************************************/
    /* REGION FOR DATA-PREPARATION (encapsulating into objects...)   */
    /*****************************************************************/
    
    /**
     * Process a line of data set
     * For each value of each feature, encapsulate it into a FeatureAgregateInfo(fetureIndex, xValue, yValue, frequency)
     *
     * @param line			array of value of each feature in a "record"
     * @param numbeFeatures	the TOTAL number of feature in data set (include features which may be not processed)
     * @param fTypes		type of each feature in each line (in ordered)
     * @return an array of FeatureAggregateInfo, each element is a value of each feature on this line
     */
    private def convertArrayValuesToObjects(arrayValues: Array[String]): Array[FeatureValueLabelAggregate] = {
        
        var yValue = arrayValues(yIndex).toDouble
        
        var i = -1
        //Utility.parseDouble(arrayValues(yIndex)) match {
        //    case Some(yValue) => { // check type of Y : if isn't continuous type, return nothing
        arrayValues.map {
            element =>
                {
                    i = (i + 1) % fullFeatureSet.numberOfFeature
                    if (!this.xIndexes.contains(i)) {
                        var f = encapsulateValueIntoObject(-i - 1, "0", 0, FeatureType.Numerical)
                        f.frequency = -1
                        f
                    } else
                        fullFeatureSet.data(i).Type match {
                            case FeatureType.Categorical => encapsulateValueIntoObject(i, element, yValue, FeatureType.Categorical)
                            case FeatureType.Numerical => encapsulateValueIntoObject(i, element, yValue, FeatureType.Numerical)
                        }
                }
        }
    }

    def encapsulateValueIntoObject(index: Int, value: String, yValue: Double, featureType: FeatureType.Value): FeatureValueLabelAggregate = {
        featureType match {
            case FeatureType.Categorical => new FeatureValueLabelAggregate(index, value, yValue, yValue * yValue, 1)
            case FeatureType.Numerical => new FeatureValueLabelAggregate(index, value.toDouble, yValue, yValue * yValue, 1)
        }
    }
    
    /*********************************************************************/
    /*    REGION FUNCTIONS OF BUILDING PHASE    */
    /*********************************************************************/
    
        /**
     * Check a sub data set has meet stop criterion or not
     *
     * @param data	RDD[(label, indexOfFeature, xValue)(yValue, yValues, frequency)]
     * @return <code>true</code>/<code>false</code> and the average of value of target feature
     */
    def checkStopCriterion(data: RDD[((BigInt, Int, Any), (Double, Double, Int))]): Array[(BigInt, Boolean, StatisticalInformation)] = {
        
        // select only 1 feature of each region
        val firstFeature = (data.filter{ 
            	case ((label, index, xValue), (yValue, yValuePower2, frequency)) => index == this.xIndexes.head}
        		.map{ 
        		    case ((label, index, xValue), (yValue, yValuePower2, frequency)) 
        		    	=> (label, (yValue, yValuePower2, frequency))}
                ) // (label, feature)
        //val firstFeature = data.filter(_._1._2 == this.xIndexes.head).map(x => (x._1._1, x._2)) // (label, feature)

        //yFeature.collect.foreach(println)

        val aggregateFeatures = firstFeature.reduceByKey((x,y) => (x._1 + y._1, x._2 + y._2, x._3 + y._3)) // sum by label

        val standardDeviations = aggregateFeatures.collect.map{
            case (label, (yValue, yValuePower2, frequency)) => {
            //val feature = f._2
            val meanY = yValue / frequency
            val meanOfYPower2 = yValuePower2 / frequency
            (label, math.sqrt(meanOfYPower2 - meanY * meanY), frequency, yValue, yValuePower2)
            // (label, standardDeviation, numberOfRecords, yValue, yValuePower2)
        }}

        // Array[(Label, isStop, meanY, standardDeviation)]	// we use standard deviation is a error metric
        var result = standardDeviations.map(
            label_sd_fre_yValue_yValuePower2 => {
                var (label, standardDeviation, numInstances,sumYValue, sumYValuePower2) = label_sd_fre_yValue_yValuePower2
                var statisticalInformation = new StatisticalInformation(sumYValue, sumYValuePower2, numInstances)
                var meanY = sumYValue/numInstances
                
                (
                    label,	// label
                    (
                        (numInstances <= this.minsplit) // or the number of records is less than minimum
                        || (((standardDeviation <= this.threshold) && (meanY == 0))
                            || (standardDeviation / meanY <= this.threshold)) // or standard devariance of values of Y feature is small enough
                     ),
                     statisticalInformation)
            })

            // compare the risk of each node with it's parent. If the different is too small, stop expanding this node
        if (!this.treeModel.isEmpty && !this.treeModel.tree.isLeaf) {

            result.map(x => {

                val (label, isStop, statisticalInfor) = x

                var parent = getNodeByID(label >> 1)
                if (parent != null) {
                    var EY2OfParent: Double = parent.statisticalInformation.sumOfYPower2.asInstanceOf[Double] / parent.statisticalInformation.numberOfInstances
                    var EYOfParent: Double = parent.statisticalInformation.YValue.asInstanceOf[Double] / parent.statisticalInformation.numberOfInstances
                    var MSEOfParent = (EY2OfParent - EYOfParent * EYOfParent) * parent.statisticalInformation.numberOfInstances.toInt

                    val EY2: Double = statisticalInfor.sumOfYPower2.asInstanceOf[Double] / statisticalInfor.numberOfInstances.toInt
                    val EY: Double = statisticalInfor.YValue.asInstanceOf[Double] / statisticalInfor.numberOfInstances.toInt
                    val MSE = (EY2 - EY * EY) * statisticalInfor.numberOfInstances.toInt
                    if (DEBUG) println("label " + label + " current MSE:" + MSE + " parent MSE: " + MSEOfParent + " statisParent:" + parent.statisticalInformation)
                    
                    if ((math.abs(MSE - MSEOfParent) / MSEOfParent) <= this.maximumComplexity) {
                        (label, true, statisticalInfor)
                    } else {
                        x
                    }
                } else x

            })
        } else result

    }

    
    
	override def startBuildTree(trainingData: RDD[String],
        xIndexes: Set[Int],
        yIndex: Int): Unit = {
	    var rootID = 1
        
        var expandingNodeIndexes = Set[BigInt]()
        
        var map_label_to_splitpoint = Map[BigInt, SplitPoint]()

        def finish() = {
            expandingNodeIndexes.isEmpty
            //map_label_to_splitpoint.isEmpty
        }

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

        /* END OF REGION TRANSFORMING */

        // set label for the first job
        // already set by default constructor of class FeatureValueLabelAggregate , so we don't need to put data to regions
        // if this function is called by ContinueFromIncompleteModel, mark the data by the last labels
        transformedData = markDataByLabel(transformedData, regions)

        // NOTE: label == x, means, this is data used for building node id=x

        //var map_label_to_splitpoint = Map[BigInt, SplitPoint]()
        var isError = false;
        var errorStack : String = ""

        var iter = 0;


        do {
            iter = iter + 1
            
            try {
                //if (iter == 5)
                //    throw new Exception("Break for debugging")

                println("\n\n\nITERATION---------------------%d------------- expands from %d node\n\n".format(iter, expandingNodeIndexes.count(p => true)))

                
                // save current model before growing tree
                this.treeModel.writeToFile(this.temporaryModelFile)
                
                var data = transformedData.flatMap(x => x.toSeq).filter(x => (x.index >= 0))
                
                var featureValueAggregate = data.map(x => ((x.label, x.index, x.xValue), (x.yValue, x.yValuePower2, x.frequency))).
                	reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2, x._3 + y._3))	// sum of (yValue, yValuePower2, frequency)
                
                var checkedStopExpanding = checkStopCriterion(featureValueAggregate)
                // we get Array[(label, isStop, statisticalInformation)]
                if (DEBUG) println("Checked stop expanding:\n%s".format(checkedStopExpanding.mkString("\n")))
                
                
                // if the tree height enough, mark all node is stop node
                if (iter > this.maxDepth)
                    checkedStopExpanding = checkedStopExpanding.map{case (label, isStop, statisticalInfo) => (label, true, statisticalInfo)}
                    
                    
                // select stopped group
                val stopExpandingGroups = checkedStopExpanding.filter{case (label, isStop, statisticalInfo) => isStop}.
                    map{ 
                    case (label, isStop, statisticalInfo)
                    	=> (
                    	        label, 
                    	        new SplitPoint(-1, statisticalInfo.YValue.asInstanceOf[Double]/statisticalInfo.numberOfInstances, 0), 
                    	        statisticalInfo
                    	   )
                    }	
                // Array[(label, splitpoint, statisticalInformation)]


                // update model with in-expandable group
                updateModel(stopExpandingGroups, true)

                // select indexes/labels of expanding groups
                val continueExpandingGroups = checkedStopExpanding.filter{case (label, isStop, statisticalInfo) => !isStop}
                val expandingLabels = continueExpandingGroups.map{case (label, isStop, statisticalInfo) => label}.toSet
                
                var mapLabel_To_CheckStopResult_Of_ExpandingNodes = Map[BigInt, StatisticalInformation]()
                continueExpandingGroups.foreach{ case (label, isStop, statisticalInfo) => {
                    mapLabel_To_CheckStopResult_Of_ExpandingNodes = 
                        mapLabel_To_CheckStopResult_Of_ExpandingNodes.+(label -> statisticalInfo)
                }}
                
                // ((label, index, xValue), featureAggregate)
                featureValueAggregate = featureValueAggregate.filter(f => expandingLabels.contains(f._1._1))
                
                val sortedFeatureValueAggregates = (
                    featureValueAggregate.map{ 
                        case ((label, index, xValue),(yValue, yValuePower2, frequency)) 
                        	=> ((label, index), (xValue, yValue, yValuePower2, frequency))} // ((label,index), feature)
                    .groupByKey()
                    .map { 
                        case ((label, index), arrayOf_xValue_yValue_yPower2_fre) =>
                        ((label, index), arrayOf_xValue_yValue_yPower2_fre.toList.sortBy{
                            case (xValue, yValue, yValuePower2, frequency) => xValue match {
                                case d: Double => d // sort by xValue if this is numerical feature
                                case s: String => yValue / frequency // sort by the average of Y if this is categorical value
                            }
                            })
                     }
                    )

                val splittingPointFeatureOfEachRegion =
                    (sortedFeatureValueAggregates.map{
                        case ((label, index), arrayOf_xValue_yValue_yPower2_fre) => {
                        //val index = x._1._2
                        //val region = x._1._1
                        this.fullFeatureSet.data(index).Type match {
                            case FeatureType.Numerical => {
                                (label, findBestSplitPointForNumericalFeature(label, index, arrayOf_xValue_yValue_yPower2_fre))
                            }

                            case FeatureType.Categorical => {
                                (label, findBestSplitPointForCategoricalFeature(label, index, arrayOf_xValue_yValue_yPower2_fre))
                            }
                        }
                    }} // find best split point of all features
                        .groupBy(_._1) // group by region
                        .collect
                        .map(f => f._2.maxBy(region_sp => region_sp._2.weight))
                    )

                
                // process split points
                val validSplitPoint = splittingPointFeatureOfEachRegion.filter(_._2.index != -9)	// (label, splitpoint)

                // select split point of region with has only one feature --> it is a leaf node
                val stoppedSplitPoints = validSplitPoint.filter(_._2.index == -1).
                	map(x => {
                	    val checkStopResult = mapLabel_To_CheckStopResult_Of_ExpandingNodes.getOrElse(x._1, new StatisticalInformation())
                	    (x._1, x._2, checkStopResult)
                	})

                val nonstoppedSplitPoints = validSplitPoint.filter(_._2.index != -1).
                	map(x => {
                	    val checkStopResult = mapLabel_To_CheckStopResult_Of_ExpandingNodes.getOrElse(x._1, new StatisticalInformation())
                	    (x._1, x._2, checkStopResult)
                	})

                updateModel(stoppedSplitPoints, true)
                updateModel(nonstoppedSplitPoints, false)

                
                expandingNodeIndexes = Set[BigInt]()
                if (iter >= 2)
                	map_label_to_splitpoint = map_label_to_splitpoint.filter(p => p._1 > (1 << iter))
                
                nonstoppedSplitPoints.foreach(point =>
                    // add expanding Indexes into set
                    {
                        expandingNodeIndexes = expandingNodeIndexes + (point._1)
                        map_label_to_splitpoint = map_label_to_splitpoint + (point._1 -> point._2) // label -> splitpoint
                    })

                //println("expandingNodeIndexes:" + expandingNodeIndexes)
                //println("map_label_to_splitpoint:%s\n\n".format(map_label_to_splitpoint))
                
                // mark new label for expanding data
                transformedData = updateLabels(transformedData, map_label_to_splitpoint)
                
            } catch {
                case e: Exception => {
                    isError = true;
                    errorStack = e.getStackTraceString
                    expandingNodeIndexes = Set[BigInt]()
                }
            }
        } while (!finish())

        treeModel.isComplete = !isError;

        /* FINALIZE THE ALGORITHM */
        if (!isError) {
            this.treeModel.isComplete = true
            println("\n------------------DONE WITHOUT ERROR------------------\n")
        } else {
            this.treeModel.isComplete = false
            println("\n--------FINISH with some failed jobs at iteration " + iter + " ----------\n")
            println("Error Message: \n%s\n".format(errorStack))
            println("Temporaty Tree model is stored at " + this.temporaryModelFile + "\n")
        }

	}
    
    private def updateLabels(data : RDD[Array[FeatureValueLabelAggregate]],
            map_label_to_splitpoint: Map[BigInt, SplitPoint]) 
    = {
        data.map(array => {

                    val currentLabel = array(0).label
                    		
                    val splitPoint = map_label_to_splitpoint.getOrElse(currentLabel, new SplitPoint(-9, 0, 0))
   
                    if (splitPoint.index < 0) { // this is stop node
                        //println("split point index:" + splitPoint.index)
                        array.foreach(element => { element.index = -9 })
                    } else { // this is expanding node => change label of its data
                        splitPoint.point match {
                            // split on numerical feature
                            case d: Double =>
                                {
                                    if (array(splitPoint.index).xValue.asInstanceOf[Double] < splitPoint.point.asInstanceOf[Double]) {
                                        array.foreach(element => element.label = (element.label << 1))
                                    } else {
                                        array.foreach(element => element.label = (element.label << 1 ) +  1)
                                    }
                                }

                            // split on categorical feature    
                            case s: Set[String] =>
                                {
                                    if (splitPoint.point.asInstanceOf[Set[String]].contains(array(splitPoint.index).xValue.asInstanceOf[String])) {
                                        array.foreach(element => element.label = (element.label << 1))
                                    } else {
                                        array.foreach(element => element.label = (element.label << 1 ) + 1)
                                    }
                                }
                        }
                    }
                    array
                })
    }
    
    private def findBestSplitPointForNumericalFeature(label: BigInt, index: Int, allValues: Seq[AggregateInfo]): SplitPoint = {
    	
        if (allValues.length == 1) { // have only 1 xvalue (maybe more than one Y values) -> can not split anymore
            val (xValue, yValue, yValuePower2, frequency) = allValues(0)
            new SplitPoint(-1, getPredictedValue(new StatisticalInformation(yValue, yValuePower2, frequency)), 0.0) // sign of stop node
        } else {
            var acc: Int = 0 // number of records on the left of the current element
            var currentSumY: Double = 0

            var (_, sumY, _, numRecs) = allValues.reduce((f1, f2) => (f1._1, f1._2 + f2._2, f1._3 + f2._3, f1._4 + f2._4))
            // sum (yValue, yPower2, frequency); we don't care about the xValue

            var posibleSplitPoint: Double = 0
            //var lastFeatureValue = new FeatureValueLabelAggregate(-1, 0, 0, 0, 0, label)
            var (lastIndex, lastXValue: Any, lastYValue, lastFrequency) = (-1, 0.0, 0.0, 0)
            var bestSplitPoint = new SplitPoint(index, posibleSplitPoint, 0)
            var maxWeight = Double.MinValue
            var currentWeight: Double = 0

            allValues.foreach {
                case (xValue, yValue, yValuePower2, frequency) => {

                    if (lastIndex == -1) {
                        lastIndex = 0; lastXValue = xValue; lastYValue = yValue; lastFrequency = frequency
                    } else {
                        posibleSplitPoint = (xValue.asInstanceOf[Double] + lastXValue.asInstanceOf[Double]) / 2;
                        currentSumY = currentSumY + lastYValue
                        acc = acc + lastFrequency
                        currentWeight = currentSumY * currentSumY / acc + (sumY - currentSumY) * (sumY - currentSumY) / (numRecs - acc)
                        lastXValue = xValue; lastYValue = yValue; lastFrequency = frequency
                        if (currentWeight > maxWeight) {
                            bestSplitPoint.point = posibleSplitPoint
                            bestSplitPoint.weight = currentWeight
                            maxWeight = currentWeight
                        }
                    }
                }
            }
            bestSplitPoint
        }
    }

    private def findBestSplitPointForCategoricalFeature(label: BigInt, index: Int, allValues: Seq[AggregateInfo]): SplitPoint = {
        
        if (allValues.length == 1) { // have only 1 xvalue (maybe more than one Y values) -> can not split anymore
            val (xValue, yValue, yValuePower2, frequency) = allValues(0)
            new SplitPoint(-1, getPredictedValue(new StatisticalInformation(yValue, yValuePower2, frequency)), 0.0) // sign of stop node
        } else {

        	var acc: Int = 0 // number of records on the left of the current element
            var currentSumY: Double = 0

            var (_, sumY, _, numRecs) = allValues.reduce((f1, f2) => (f1._1, f1._2 + f2._2, f1._3 + f2._3, f1._4 + f2._4))
            // sum (yValue, yPower2, frequency); we don't care about the xValue
            var splitPointIndex: Int = 0
            
            var (lastIndex, lastXValue: Any, lastYValue, lastFrequency) = (-1, 0.0, 0.0, 0)
            
            var bestSplitPoint = new SplitPoint(index, splitPointIndex, 0)
            var maxWeight = Double.MinValue
            var currentWeight: Double = 0

            allValues.foreach{
            	case (xValue, yValue, yValuePower2, frequency) => {
            	    
                if (lastIndex == -1) {
                        lastIndex = 0; lastXValue = xValue; lastYValue = yValue; lastFrequency = frequency
                } else {
                    currentSumY = currentSumY + lastYValue
                    splitPointIndex = splitPointIndex + 1
                    acc = acc + lastFrequency
                    currentWeight = currentSumY * currentSumY / acc + (sumY - currentSumY) * (sumY - currentSumY) / (numRecs - acc)
                    lastXValue = xValue; lastYValue = yValue; lastFrequency = frequency
                    if (currentWeight > maxWeight) {
                        bestSplitPoint.point = splitPointIndex
                        bestSplitPoint.weight = currentWeight
                        maxWeight = currentWeight
                    }
                }
            }}

            var splitPointValue = allValues.map{case (xValue, yValue, yValuePower2, frequency) => xValue}.take(splitPointIndex).toSet
            bestSplitPoint.point = splitPointValue
            bestSplitPoint
        }
    }

    private def markDataByLabel(data: RDD[Array[FeatureValueLabelAggregate]], regions: List[(BigInt, List[Condition])]): RDD[Array[FeatureValueLabelAggregate]] = {
        var newdata =
            if (regions.length > 0) {
                data.map(line => {
                    var labeled = false

                    // if a line can match one of the Conditions of a region, label it by the ID of this region
                    regions.foreach(region => {
                        if (region._2.forall(c => c.check(line(c.splitPoint.index).xValue))) {
                            line.foreach(element => element.label = region._1)
                            labeled = true
                        }
                    })

                    // if this line wasn't marked, it means this line isn't used for building tree
                    if (!labeled) line.foreach(element => element.index = -9)
                    line
                })
            } else data

        newdata
    }

    /**
     * Init the last labels from the leaf nodes
     */
    private def initTheLastLabelsFromLeafNodes() = {

        var jobIDList = List[(BigInt, List[Condition])]()

        def generateJobIter(currentNode: CARTNode, id: BigInt, conditions: List[Condition]): Unit = {

            if (currentNode.isLeaf &&
                (currentNode.value == "empty.left" || currentNode.value == "empty.right")) {
                jobIDList = jobIDList :+ (id, conditions)
            }

            if (!currentNode.isLeaf) { // it has 2 children
                var newConditionsLeft = conditions :+
                    new Condition(new SplitPoint(currentNode.feature.index, currentNode.splitpoint, 0), true)
                generateJobIter(currentNode.left, id * 2, newConditionsLeft)

                var newConditionsRight = conditions :+
                    new Condition(new SplitPoint(currentNode.feature.index, currentNode.splitpoint, 0), false)
                generateJobIter(currentNode.right, id * 2 + 1, newConditionsRight)
            }
        }

        generateJobIter(treeModel.tree.asInstanceOf[CARTNode], 1, List[Condition]())

        jobIDList.sortBy(-_._1) // sort jobs by ID descending

        var highestLabel = Math.log(jobIDList(0)._1.toDouble) / Math.log(2)
        jobIDList.filter(x => Math.log(x._1.toDouble) / Math.log(2) == highestLabel)

        regions = jobIDList

    }
    
    override def createNewInstance : TreeBuilder = {
    	new RegressionTree()	   
    }
}