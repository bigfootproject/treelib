package fr.eurecom.dsg.treelib.cart


import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import scala.collection.mutable.HashMap
import java.io._
import scala.util.Random
import fr.eurecom.dsg.treelib.core._

import scala.Array.canBuildFrom
import scala.math.BigInt.int2bigInt


class ClassificationTree extends TreeBuilder {

  type AggregateInfo = (Any, String, Int) // (xValue, yValue, frequency)
  /**
   * The collection of current existing nodes and the corresponding condition to go to each node
   */
   var regions = List[(BigInt, List[Condition])]()
  /**
   * The number of feature will be choose in Random Feature Selection
   */
   private var numberOfRandomFeature: Int = 0

  /** *******************************************************************/
  override def startBuildTree(trainingData: RDD[String],
    xIndexes: Set[Int],
    yIndex: Int): Unit =
  {

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
    var errorStack: String = ""

    var iter = 0;

    //println("featureset" + usefulFeatureSet.data.mkString("\n"))

    do {
      iter = iter + 1

      try {
        //if (iter == 5)
        //    throw new Exception("Break for debugging")

        println("\n\n\nITERATION---------------------%d------------- expands from %d node\n\n".format(iter, expandingNodeIndexes.count(p => true)))

        // TODO: I would like this to be optional.
        // save current model before growing tree
        this.treeModel.writeToFile(this.temporaryModelFile)

        // TODO: Check all these var, they should be val
        var data = transformedData.flatMap(x => x.toSeq).filter(x => (x.index >= 0 && x.label > 0))
        //transformedData.foreach(x => println(x.mkString(" || ")))

        var featureValueAggregate = data.map(x => {
          ((x.label, x.index, x.xValue, x.yValue), (x.frequency))
          }).reduceByKey(_ + _)

        val YValueFrequenciesOfEachXValue = featureValueAggregate.map {
          case ((label, index, xValue, yValue), (frequency)) => ((label, index, xValue), (yValue, frequency))
          }.groupByKey

          var distributionOfEachFeature = YValueFrequenciesOfEachXValue.map {
            case ((label, index, xValue), seqYValueFrequency) => ((label, index), (xValue, seqYValueFrequency))
            }.groupByKey
        /*
        var frequencyOfEachPair = featureValueAggregate.map {
            case ((label, index, xValue, yValue), frequency) => {
                ((label, index),(xValue, yValue, frequency))
            }
        }.groupByKey
        */

        // if (the situation is RandomForest, and) we need to select random subset of features
        if (this.useRandomSubsetFeature) {
          var temp = distributionOfEachFeature.map(x => x._1).groupByKey

          val randomSelectedFeatureAtEachNode = temp.flatMap {
            case (label, sequenceOfFIndices) => {
              generateRandomSet(sequenceOfFIndices).map(x => (label, x))
            }
            }.collect.toSet

            distributionOfEachFeature = distributionOfEachFeature.filter(x => randomSelectedFeatureAtEachNode.contains(x._1))
          }

        // TODO: cache splitPoint_And_YValueDistribution_OfEachNode
        var splitPoint_And_YValueDistribution_OfEachNode = distributionOfEachFeature.map {
          case ((label, index), seq_xValue_yValue_fre) => {
            fullFeatureSet.data(index).Type match {
              case FeatureType.Numerical => {
                (label, (index, findBestSplitPointNumericalFeature(label, index, seq_xValue_yValue_fre)))
              }
              case FeatureType.Categorical => {
                (label, (index, findBestSplitPointCategoricalFeature(label, index, seq_xValue_yValue_fre)))
              }
            }

          }
          }.groupByKey.map {
            case (label, seq_fIndex_splitPoint) => {
              (label, seq_fIndex_splitPoint.minBy(x => x._2._1.weight)._2)
            }
          }

        /*
        var splitPoint_And_YValueDistribution_OfEachNode = frequencyOfEachPair.map {
            case ((label, index),seq_xValue_yValue_fre) => {
                (label, (index, findBestSplitPoint(label, index, seq_xValue_yValue_fre)))
            }
        }.groupByKey.map {
            case (label, seq_fIndex_splitPoint) => {
                (label, seq_fIndex_splitPoint.minBy(x => x._2._1.weight)._2)
            }
        }
        *
        */
        // we get: (label, splitPoint)

        //println("Debug splitpoint of each node:")
        //splitPoint_And_YValueDistribution_OfEachNode.collect.foreach(println)

        val stopNodes = splitPoint_And_YValueDistribution_OfEachNode.filter {
          case (label, (splitPoint, yValueDistribution)) => (splitPoint.index == -1)
          }.collect.map {
            case (label, (splitPoint, yValueDistribution)) => (label, splitPoint, yValueDistribution)
            }.toArray



            val non_stopNodes = splitPoint_And_YValueDistribution_OfEachNode.filter {
              case (label, (splitPoint, yValueDistribution)) => (splitPoint.index != -1)
              }.collect.map {
                case (label, (splitPoint, yValueDistribution)) => (label, splitPoint, yValueDistribution)
                }.toArray

                expandingNodeIndexes = non_stopNodes.map(x => x._1).toSet

        // generate leaf-nodes
        updateModel(stopNodes, true)
        // generate intermediate nodes
        updateModel(non_stopNodes, false)



        var splitPointOfEachNode = HashMap[BigInt, SplitPoint]()
        splitPoint_And_YValueDistribution_OfEachNode.map(x => (x._1, x._2._1)).collect.foreach {
          case (key, value) => {
            splitPointOfEachNode.update(key, value)
          }
        }

        // TODO: unpersist splitPoint_And_YValueDistribution_OfEachNode

        //println("MAP:" + splitPointOfEachNode.mkString(","))

        transformedData = updateLabels(transformedData, splitPointOfEachNode.clone)

        //println("current data:")
        //transformedData.foreach (x => {
        //    println(x.mkString(","))
        //})
        //temp.collect.foreach(println)

        //println("current Tree:\n" + this.treeModel)

        } catch {
          case e: Exception => {
            isError = true
            errorStack = e.getStackTraceString
            expandingNodeIndexes = Set[BigInt]()
          }
        }
        } while (!finish)

        treeModel.isComplete = !isError

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


  /** ***************************************************************/
  /* REGION FOR DATA-PREPARATION (encapsulating into objects...)   */
  /** ***************************************************************/

  /**
   * Process a line of data set
   * For each value of each feature, encapsulate it into a FeatureAgregateInfo(fetureIndex, xValue, yValue, frequency)
   *
   * @param line			array of value of each feature in a "record"
   * @param numbeFeatures	the TOTAL number of feature in data set (include features which may be not processed)
   * @param fTypes		type of each feature in each line (in ordered)
   * @return an array of FeatureAggregateInfo, each element is a value of each feature on this line
   */
   private def convertArrayValuesToObjects(arrayValues: Array[String]): Array[FeatureValueAggregate] = {
    var yValue = arrayValues(yIndex) //.toDouble
    var i = -1
    //Utility.parseDouble(arrayValues(yIndex)) match {
    //    case Some(yValue) => { // check type of Y : if isn't continuous type, return nothing
    arrayValues.map {
      element => {
        i = (i + 1) % fullFeatureSet.numberOfFeature
        if (!this.xIndexes.contains(i)) {
          //println("---------------------- " + i)
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

  /**
   * Encapsulate a feature value into object
   * @param	index		the feature index
   * @param	value		the feature value
   * @param	yValue		the associate value of the target feature
   * @param	featureType	type of the feature (Numerical or Categorical)
   * @output	an aggregated value of feature
   */
   def encapsulateValueIntoObject(index: Int, value: String, yValue: Any, featureType: FeatureType.Value): FeatureValueAggregate = {
    featureType match {
      case FeatureType.Categorical => new FeatureValueAggregate(index, value, yValue, 1)
      //new FeatureValueLabelAggregate(index, value, yValue, yValue * yValue, 1)
      case FeatureType.Numerical => new FeatureValueAggregate(index, value.toDouble, yValue, 1)
      //new FeatureValueLabelAggregate(index, value.toDouble, yValue, yValue * yValue, 1)
    }
  }


  /** *******************************************************************/
  /*    REGION FUNCTIONS OF BUILDING PHASE    */

  private def findBestSplitPointNumericalFeature(label: BigInt, index: Int, seqXValue_YValue_Frequency: Iterable[(Any, Iterable[(Any, Int)])])
  : (SplitPoint, StatisticalInformation) = {
    var newSeqXValue_YValue_Frequency = seqXValue_YValue_Frequency.toList.sortBy(x => x._1.asInstanceOf[Double]) // sort by xValue
    var mapYValueToFrequency = (seqXValue_YValue_Frequency.flatMap(x => x._2).groupBy(_._1)
      .map { case (group, traversable) => traversable.reduce { (a, b) => (a._1, a._2 + b._2)}}
      )
    val mapYValueToIndex = mapYValueToFrequency.keys.zipWithIndex.map(x => (x._1 -> x._2)).toMap
    val numberOfYValue = mapYValueToIndex.size
    var totalFrequencyOfEachYValue = Array.fill(numberOfYValue)(0)
    var frequencyOfYValueInLeftNode = Array.fill(numberOfYValue)(0)
    var sumOfFrequency: Int = 0
    var sumFrequencyLeft: Int = 0
    var sumFrequencyRight: Int = 0

    mapYValueToFrequency.foreach {
      x => {
        totalFrequencyOfEachYValue.update(mapYValueToIndex.getOrElse(x._1, -1), x._2)
        sumOfFrequency = sumOfFrequency + x._2
      }
    }

    val statisticalInfo = mapYValueToFrequency.toArray.sortBy(x => -x._2).take(3)

    // if the target feature has only 1 value => don't need to split anymore
    if (numberOfYValue == 1
      || (seqXValue_YValue_Frequency.size == 1) // or if the number of value of the predictor is 1
      || (sumOfFrequency <= this.minsplit)) {
      return (new SplitPoint(-1, statisticalInfo.head._1, 0.0), new StatisticalInformation(statisticalInfo, 0, sumOfFrequency))
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
        }
        )
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

    (new SplitPoint(index, splitPoint, maxGain), new StatisticalInformation(statisticalInfo, 0, sumOfFrequency))

    //.reduce((x,y) => (x._1, x._2 + y._2))
  }

  private def findBestSplitPointCategoricalFeature(label: BigInt, index: Int, seqXValue_YValue_Frequency: Iterable[(Any, Iterable[(Any, Int)])])
  : (SplitPoint, StatisticalInformation) = {
    var newSeqXValue_YValue_Frequency = seqXValue_YValue_Frequency //.sortBy(x => x._1.asInstanceOf[Double])// sort by xValue
    var mapYValueToFrequency = (seqXValue_YValue_Frequency.flatMap(x => x._2).groupBy(_._1)
      .map { case (group, traversable) => traversable.reduce { (a, b) => (a._1, a._2 + b._2)}}
      )
    val mapYValueToIndex = mapYValueToFrequency.keys.zipWithIndex.map(x => (x._1 -> x._2)).toMap
    val numberOfYValue = mapYValueToIndex.size
    var totalFrequencyOfEachYValue = Array.fill(numberOfYValue)(0)
    //var frequencyOfYValueInLeftNode = Array.fill(numberOfYValue)(0)
    //var frequencyOfYValueInRightNode = Array.fill(numberOfYValue)(0)
    var sumOfFrequency: Int = 0
    var sumFrequencyLeft: Int = 0
    var sumFrequencyRight: Int = 0

    mapYValueToFrequency.foreach {
      x => {
        totalFrequencyOfEachYValue.update(mapYValueToIndex.getOrElse(x._1, -1), x._2)
        sumOfFrequency = sumOfFrequency + x._2
      }
    }

    val xValueAndArrayOfYFrequency = newSeqXValue_YValue_Frequency.map {
      case (xValue, seqYValueFrequency) => {
        var temp = Array.fill(numberOfYValue)(0)
        seqYValueFrequency.foreach(x => {
          temp.update(mapYValueToIndex.getOrElse(x._1, -1), x._2)
          })
        (xValue, temp)
      }
    }

    val statisticalInfo = mapYValueToFrequency.toArray.sortBy(x => -x._2).take(3)

    // if the target feature has only 1 value => don't need to split anymore
    if (numberOfYValue == 1
      || (seqXValue_YValue_Frequency.size == 1) // or if the number of value of the predictor is 1
      || (sumOfFrequency <= this.minsplit)) {
      return (new SplitPoint(-1, statisticalInfo.head._1, 0.0), new StatisticalInformation(statisticalInfo, 0, sumOfFrequency))
    }

    var lastXValue: Double = 0
    var minGain = Double.MaxValue
    var splitPoint = Set[Any]()

    sumFrequencyRight = sumOfFrequency



    def generatePossibleSplitpoint(values: List[(Any, Array[Int])]) = {
      def generateIter(currentIndex: Int, currentSet: Set[Any],
       currentFrequenciesLeft: Array[Int]): Unit = {

        for (i <- currentIndex until values.length) {
          val newSet = currentSet.+(values(i)._1) // add XValue into splitpoint
          var frequenciesInLeft = values(i)._2
          var frequenciesInRight = totalFrequencyOfEachYValue.clone
          var fre: Int = 0
          for (j <- 0 until numberOfYValue) {
            fre = fre + frequenciesInLeft(j)
            frequenciesInLeft.update(j, frequenciesInLeft(j) + currentFrequenciesLeft(j))
            frequenciesInRight.update(j, frequenciesInRight(j) - frequenciesInLeft(j))
          }
          //println(newSet)

          var giniLeft: Double = 0
          var giniRight: Double = 0
          sumFrequencyLeft = sumFrequencyLeft + fre
          sumFrequencyRight = sumOfFrequency - fre

          for (j <- 0 until frequenciesInLeft.length) {
            //println("FrequencyLeft:" + frequenciesInLeft.mkString(","))
            //println("FrequencyRight:" + frequenciesInRight.mkString(","))
            //println("sumLeft=" + sumLeft + " sumRight:" + sumRight)
            if (sumFrequencyLeft > 0)
            giniLeft = giniLeft + (frequenciesInLeft(j) / sumFrequencyLeft) * (frequenciesInLeft(j) / sumFrequencyLeft)
            if (sumFrequencyRight > 0) {
              //println("freinRight" +frequenciesInRight(j) + " sumRight" + sumRight )
              giniRight = giniRight + (frequenciesInRight(j) / sumFrequencyRight) * (frequenciesInRight(j) / sumFrequencyRight)
            }
          }
          giniLeft = 1 - giniLeft
          giniRight = 1 - giniRight

          val gain = (sumFrequencyLeft * giniLeft + sumFrequencyRight * giniRight) / sumOfFrequency
          //println("DEBUG:giniTotal" + giniTotal + " giniLeft:" + giniLeft + " giniRight:" + giniRight)
          if (minGain > gain) {
            minGain = gain
            splitPoint = newSet
          }

          generateIter(i + 1, newSet, frequenciesInLeft)
        }
      }

      generateIter(0, Set[Any](), Array.fill[Int](numberOfYValue)(0))
    }

    //println("Find split point of a feature have " + xValueAndArrayOfYFrequency.length + " xvalues")
    generatePossibleSplitpoint(xValueAndArrayOfYFrequency.toList)

    (new SplitPoint(index, splitPoint, minGain), new StatisticalInformation(statisticalInfo, 0, sumOfFrequency))

  }

  /**
   * select best splitpoint and the corresponding gini value
   * @output	(SplitPoint, Top3_YValueDistribution)
   */
  /*
  private def findBestSplitPoint(label :  BigInt, index : Int, seq : Seq[(Any, Any, Int)]) : (SplitPoint, StatisticalInformation) = {

      usefulFeatureSet.data(index).Type match {

          /**************************/
          /* CASE NUMERICAL FEATURE */
          /**************************/
          case FeatureType.Numerical => {

              // TODO: sort the "seq" by values of X
              val sortedSeq = seq.map(x => (x._1.asInstanceOf[Double], x._2, x._3)).sortBy{
                  case (xValue1, yValue1, frequency1) => {
                      xValue1.asInstanceOf[Double]
                  }
              }

              var lastYValue = Double.MaxValue
              var sumLeft : Double = 0	// the total frequency in the left node
              var sumRight : Double = 0	// the total frequency in the right node
              var frequencyTotal : Int = 0	// the total frequency in node which we want to split

              // Calculate the frequency of each yValue
              val frequencyOfYValue = (sortedSeq.map(x => (x._2, x._3)).groupBy(x => x._1) // group by yValue
                  .map {
                      case (yValue, arrayValues) => {
                          var sum: Int = 0
                          arrayValues.foreach(x => {
                              sum = sum + x._2

                          })
                          frequencyTotal = frequencyTotal + sum
                          (yValue, sum)
                      }
                  })

              //println("frequency of Yvalue:" + frequencyOfYValue.mkString(",") + " frequencyTotal:" + frequencyTotal)
              //val yValuesDistinct = yValues.distinct
              val yValuesDistinct = frequencyOfYValue.map (x => x._1).toArray

              // TODO: calculate M = the number of distinct value of Y
              val numberOfYValue = yValuesDistinct.length

              val yValuesWithIndex = yValuesDistinct.zipWithIndex.map ( x => (x._1 -> x._2)).toMap	// value -> index

              // TODO: Init array of length M with all zero values
              var frequencyInLeftNode = Array.fill[Int](numberOfYValue)(0)
              var frequencyInParent = Array.fill[Int](numberOfYValue)(0)
              frequencyOfYValue.foreach {
                  x => {
                      frequencyInParent.update(yValuesWithIndex.getOrElse(x._1, -1), x._2)
                  }
              }


              val statisticalInfo = frequencyOfYValue.toArray.sortBy(x => -x._2).take(3)
              // if the target feature has only 1 value => don't need to split anymore
              if (numberOfYValue == 1
                      || (sortedSeq.head == sortedSeq.last)	// or if the number of value of the predictor is 1
                      || (frequencyTotal <= this.minsplit)
                      ){
                  return (new SplitPoint(-1, statisticalInfo.head._1, 0.0), new StatisticalInformation(statisticalInfo, 0, frequencyTotal))
              }

              var lastXValue : Double = sortedSeq(0)._1.asInstanceOf[Double]
              var splitPoint = Double.MinPositiveValue
              var minGain = Double.MaxValue

              // TODO: Go through all possible value of split point, update the counting array and calculate the gini index

              sortedSeq.foreach {
                  case (xValue, yValue, frequency) => {
                      //println("Consider:" + (xValue, yValue, frequency))
                      val yValueIndex =  yValuesWithIndex.getOrElse(yValue, -1)


                      if (xValue != lastXValue || sortedSeq.last == (xValue, yValue, frequency)) {

                        var giniLeft :Double = 0
                        var giniRight : Double = 0

                        //println("freInLeft:" + frequencyInLeftNode.mkString(","))
                          for (j <- 0 until frequencyInLeftNode.length) {
                              //giniLeft = giniLeft + (freArray(j)/sumLeft)*(freArray(j)/sumLeft)
                              val frequencyLeft = frequencyInLeftNode(j)
                              val frequencyRight = frequencyInParent(j) - frequencyLeft
                              //giniRight = giniRight + (frequencyRight/sumRight)*(frequencyRight/sumRight)
                              //println("sumLeft:" + sumLeft + " sumRight:" + sumRight + " freLeft:" + frequencyInLeftNode(j) + " freqRight:" + frequencyRight)

                              if (sumLeft > 0)
                                  giniLeft = giniLeft + (frequencyLeft / sumLeft) * (frequencyLeft / sumLeft)
                              if (sumRight > 0) {
                                  //println("freinRight" +frequenciesInRight(j) + " sumRight" + sumRight )
                                  giniRight = giniRight + (frequencyRight / sumRight) * (frequencyRight / sumRight)
                              }

                              //println("split:" + (xValue + lastXValue) / 2 + " freLeft:" + frequencyLeft + " freRight:" + frequencyRight + " sumLeft:" + sumLeft + " sumRight:" + sumRight)
                          }
                          giniLeft = 1 - giniLeft
                          giniRight = 1 - giniRight

                          val gain = (sumLeft * giniLeft + sumRight * giniRight) / (sumLeft + sumRight)
                          //println("DEBUG:giniTotal" + gain + " giniLeft:" + giniLeft + " giniRight:" + giniRight)
                          if (minGain > gain) {
                              minGain = gain
                              splitPoint = (xValue + lastXValue) / 2
                          }

                      }

                      frequencyInLeftNode.update(yValueIndex, frequencyInLeftNode(yValueIndex) + frequency)
                      sumLeft = sumLeft + frequency
                    sumRight = frequencyTotal - sumLeft

                      lastXValue = xValue
                  }
              }


              (new SplitPoint(index, splitPoint, minGain), new StatisticalInformation(statisticalInfo, 0, frequencyTotal))

          }


          /****************************/
          /* CASE CATEGORICAL FEATURE */
          /****************************/

          case FeatureType.Categorical => {
              var splitPoint = Set[Any]()
              var minGain = Double.MaxValue

              //val sortedSeq = seq

              var mapYValueToFrequency = new HashMap[Any, Int]()
              var lastYValue = Double.MaxValue

              var frequencyTotal : Int = 0


              // Calculate the frequency of each yValue
              val frequencyOfYValue = (seq.map(x => (x._2, x._3)).groupBy(x => x._1) // group by yValue
                  .map {
                      case (yValue, arrayValues) => {
                          var sum: Int = 0
                          arrayValues.foreach(x => { sum = sum + x._2 })
                          frequencyTotal = frequencyTotal + sum
                          (yValue, sum)
                      }
                  })

             //println("DEBUG: frequencyOfYValue=" + frequencyOfYValue.mkString(" , "))

              //val yValuesDistinct = yValues.distinct
              val yValuesDistinct = frequencyOfYValue.map (x => x._1).toArray

              val yValuesWithIndex = yValuesDistinct.zipWithIndex.map ( x => (x._1 -> x._2)).toMap	// value -> index

              //println("map yvalue to index:" + yValuesWithIndex)
              val frequenciesInParentNode = Array.fill(yValuesDistinct.length)(0)
              // generate schema:
              // xValue, Array[frequency_class1, frequency_class2, ...., frequency_classN]

              val xValue_ArrayYFrequency = (seq.groupBy(x => x._1) // group by xValue
                  .map {
                      case (xValue, seqXValue_YValue_Freq) => {
                          val frequenciesInLeftNode = Array.fill(yValuesDistinct.length)(0)
                          seqXValue_YValue_Freq.foreach(x => {
                              val yValue = x._2
                              val frequency = x._3
                              val index = yValuesWithIndex.getOrElse(yValue, -1)
                              println("yvalue:" + yValue + " index=" + index + " frequency=" + frequency)
                              frequenciesInLeftNode.update(index, frequenciesInLeftNode(index) + frequency)
                              frequenciesInParentNode.update(index, frequenciesInParentNode(index) + frequency)
                          })
                          (xValue, frequenciesInLeftNode)
                      }
                  }).toArray


              println("DEBUG: xValue_ArrayYFrequency=")

              xValue_ArrayYFrequency.foreach(x => {
                println("" + x._1 + " left:" + x._2.mkString(",") )
              })

              val statisticalInfo = frequencyOfYValue.toArray.sortBy(x => -x._2).take(3)

              println("StatisticalInfo:" + statisticalInfo.mkString(","))
              // if the target feature has only 1 value => don't need to split anymore
              if ((yValuesDistinct.length == 1)
                  || (xValue_ArrayYFrequency.length == 1)	// or if the number of value of the predictor is 1
                  || (frequencyTotal <= this.minsplit)){
                  println("Have only 1 yValue:" + yValuesDistinct.mkString(","))
                  return (new SplitPoint(-1, statisticalInfo.head._1, 0.0), new StatisticalInformation( statisticalInfo, 0, frequencyTotal))
              }

              //println("frequencies in parent node:" + frequenciesInParentNode.mkString(","))
              // TODO: calculate M = the number of distinct value of Y
              val numberOfYValue = yValuesDistinct.length

              // TODO: Init array of length M with all zero values
              //var freArray = Array.fill[Int](numberOfYValue)(0)

              val lastXValue : Any = seq(0)._1//.asInstanceOf[Double]
              var sumLeft : Double = 0	// Use type double for division, instead of "Int"
              var sumRight : Double = 0

              def generatePossibleSplitpoint(values: Array[(Any, Array[Int])]) = {
                  def generateIter(currentIndex: Int, currentSet: Set[Any],
                      currentFrequenciesLeft: Array[Int]): Unit = {
                      for (i <- currentIndex until values.length) {
                          val newSet = currentSet.+(values(i)._1) // add XValue into splitpoint
                          var frequenciesInLeft = values(i)._2
                          var frequenciesInRight = frequenciesInParentNode.clone
                          var fre: Int = 0
                          for (j <- 0 until numberOfYValue) {
                              fre = fre + frequenciesInLeft(j)
                              frequenciesInLeft.update(j, frequenciesInLeft(j) + currentFrequenciesLeft(j))
                              frequenciesInRight.update(j, frequenciesInRight(j) - frequenciesInLeft(j))
                          }
                          //println(newSet)

                          var giniLeft: Double = 0
                          var giniRight: Double = 0
                          sumLeft = sumLeft + fre
                          sumRight = frequencyTotal - fre

                          for (j <- 0 until frequenciesInLeft.length) {
                              //println("FrequencyLeft:" + frequenciesInLeft.mkString(","))
                              //println("FrequencyRight:" + frequenciesInRight.mkString(","))
                              //println("sumLeft=" + sumLeft + " sumRight:" + sumRight)
                              if (sumLeft > 0)
                                  giniLeft = giniLeft + (frequenciesInLeft(j) / sumLeft) * (frequenciesInLeft(j) / sumLeft)
                              if (sumRight > 0) {
                                  //println("freinRight" +frequenciesInRight(j) + " sumRight" + sumRight )
                                  giniRight = giniRight + (frequenciesInRight(j) / sumRight) * (frequenciesInRight(j) / sumRight)
                              }
                          }
                          giniLeft = 1 - giniLeft
                          giniRight = 1 - giniRight

                          val gain = (sumLeft * giniLeft + sumRight * giniRight) / (sumLeft + sumRight)
                          //println("DEBUG:giniTotal" + giniTotal + " giniLeft:" + giniLeft + " giniRight:" + giniRight)
                          if (minGain > gain) {
                              minGain = gain
                              splitPoint = newSet
                          }

                          generateIter(i + 1, newSet, frequenciesInLeft)
                      }
                  }

                  generateIter(0, Set[Any](), Array.fill[Int](numberOfYValue)(0))
              }

              generatePossibleSplitpoint(xValue_ArrayYFrequency)

              (new SplitPoint(index, splitPoint, minGain), new StatisticalInformation( statisticalInfo, 0, frequencyTotal))
              //(splitPoint, minGain)

          }

      }


  }
  */
  private def updateLabels(data: RDD[Array[FeatureValueAggregate]],
   map_label_to_splitpoint: HashMap[BigInt, SplitPoint])
  = {
    data.map(array => {

      val currentLabel = array(0).label

      val splitPoint = map_label_to_splitpoint.getOrElse(currentLabel, new SplitPoint(-9, 0, 0))

      if (splitPoint.index < 0) {
        // this is stop node
        //println("split point index:" + splitPoint.index)
        array.foreach(element => {
          element.label = -9
          })
        } else {
        // this is expanding node => change label of its data
        splitPoint.point match {
          // split on numerical feature
          case d: Double => {
            if (array(splitPoint.index).xValue.asInstanceOf[Double] < splitPoint.point.asInstanceOf[Double]) {
              array.foreach(element => element.label = (element.label << 1))
              } else {
                array.foreach(element => element.label = (element.label << 1) + 1)
              }
            }

          // split on categorical feature
          case s: Set[_] => {
            if (splitPoint.point.asInstanceOf[Set[String]].contains(array(splitPoint.index).xValue.asInstanceOf[String])) {
              array.foreach(element => element.label = (element.label << 1))
              } else {
                array.foreach(element => element.label = (element.label << 1) + 1)
              }
            }
          }
        }
        array
        })
}

  /**
   * Recover, repair and continue build tree from the last state
   *
   * @throw Exception if the tree is never built before
   */
  /*
  override def continueFromIncompleteModel(trainingData: RDD[String]) = {
      if (treeModel == null) {
          throw new Exception("The tree model is empty because of no building. Please build it first")
      }

      if (treeModel.isComplete) {
          println("This model is already complete")
      } else {
          println("Recover from the last state")
          /* INITIALIZE */
          this.featureSet = treeModel.featureSet
          this.usefulFeatureSet = treeModel.usefulFeatureSet
          this.xIndexes = treeModel.xIndexes
          this.yIndex = treeModel.yIndex

          startBuildTree(trainingData)

      }
  }
  *
  */


  private def generateRandomSet(sequenceOfFIndices: Iterable[Int]): Array[Int] = {
    var arrayOfIndices = sequenceOfFIndices.toArray
    val numFeatures = arrayOfIndices.length
    val numRandomFeatureSelection = (math.sqrt(numFeatures) + 0.5).toInt

    var selectedFeature = Array.fill(numRandomFeatureSelection)(0)


    for (i <- 0 until numRandomFeatureSelection) {
      val j = Random.nextInt(numFeatures - i) + i
      selectedFeature.update(i, arrayOfIndices(j))

      // swap element at index i and j
      arrayOfIndices = arrayOfIndices.updated(j, arrayOfIndices(i))
      arrayOfIndices = arrayOfIndices.updated(i, selectedFeature(i))
    }

    selectedFeature
  }

  private def markDataByLabel(data: RDD[Array[FeatureValueAggregate]], regions: List[(BigInt, List[Condition])]): RDD[Array[FeatureValueAggregate]] = {
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

    override def createNewInstance: TreeBuilder = {
      new ClassificationTree()
    }

    override protected def getPredictedValue(info: StatisticalInformation): Any = {
      val yValueDistributed = info.YValue.asInstanceOf[Array[(Any, Int)]]
      yValueDistributed.head._1
    }

  /**
   * Init the last labels from the leaf nodes
   */
   private def initTheLastLabelsFromLeafNodes() = {

    var jobIDList = List[(BigInt, List[Condition])]()

    def generateJobIter(currentNode: CARTNode, id: BigInt, conditions: List[Condition]): Unit = {

      if (currentNode.isLeaf &&
        (currentNode.value == "empty.left" || currentNode.value == "empty.right")) {
        jobIDList = jobIDList :+(id, conditions)
      }

      if (!currentNode.isLeaf) {
        // it has 2 children
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

  class FeatureValueAggregate(
   var index: Int = -1,
   var xValue: Any,
   var yValue: Any,
   var frequency: Int = 0,
   var label: BigInt = 1
   ) extends Serializable {
    override def toString = {
      "(index: %d xValue:%s yValue:%s frequency:%d label:%d)".format(index, xValue, yValue, frequency, label)
    }
  }

  /*
  override def createNewInstance(featureSet: FeatureSet, usefulFeatureSet: FeatureSet) : TreeBuilder = {
    var tb : TreeBuilder = new DataMarkerTreeBuilder(featureSet, usefulFeatureSet)
    tb.setMinSplit(this.minsplit)
    tb.setMaxDepth( this.maxDepth)
    tb.setDelimiter(this.delimiter)
    tb.setMaximumComplexity(this.maximumComplexity)
    tb
  }
  */
}