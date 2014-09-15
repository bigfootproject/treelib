package fr.eurecom.dsg.treelib.core

/**
 * This class is representative for each value of each feature in the data set
 * @param index Index of the feature in the whole data set, based zero
 * @param xValue Value of the current feature
 * @param yValue Value of the Y feature associated (target, predicted feature)
 * @param frequency Frequency of this value
 */
class FeatureValueAggregate(var index: Int, var xValue: Any, 
        var yValue: Double, var yValuePower2 : Double, 
        var frequency: Int) extends Serializable {

    /**
     * Increase the frequency of this feature
     * @param acc the accumulator
     */
    def updateByAggregate(yAcc : Double, yp2Acc : Double, frequencyAcc: Int) = {
        this.yValue = this.yValue + yAcc
        this.yValuePower2 = this.yValuePower2 + yp2Acc
        this.frequency = this.frequency + frequencyAcc
    }

    /**
     * Sum two FeatureValueAggregates (sum two yValues and two frequencies)
     */
    def +(that: FeatureValueAggregate) = {
        new FeatureValueAggregate(this.index, this.xValue,
            this.yValue + that.yValue,
            this.yValuePower2 + that.yValuePower2,
            this.frequency + that.frequency)
    }

    override def toString() = "Feature(index:" + index + " | xValue:" + xValue +
        " | yValue" + yValue + " | frequency:" + frequency + ")";
}
