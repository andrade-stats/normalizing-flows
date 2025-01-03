import numpy


def getOutlierRecall(trueOutlierIndicesZeroOne, estimatedOutlierIndicesZeroOne, calculationForRPrecision = True):
    assert(trueOutlierIndicesZeroOne.shape[0] == estimatedOutlierIndicesZeroOne.shape[0])
    
    nrTrueOutliers = numpy.sum(trueOutlierIndicesZeroOne)
    
    if calculationForRPrecision:
        assert(numpy.sum(estimatedOutlierIndicesZeroOne) == nrTrueOutliers)
    
    if nrTrueOutliers == 0:
        return 1.0
    else:
        nrDiscoveredOutliers = numpy.sum(trueOutlierIndicesZeroOne[estimatedOutlierIndicesZeroOne == 1])
        assert(nrDiscoveredOutliers <= nrTrueOutliers)
        return nrDiscoveredOutliers / nrTrueOutliers

def getNrFalseDetections(trueOutlierIndicesZeroOne, estimatedOutlierIndicesZeroOne):
    nrWrongDiscoveries = numpy.sum(trueOutlierIndicesZeroOne[estimatedOutlierIndicesZeroOne == 1] == 0)
    return nrWrongDiscoveries



def showOutlierDetectionPerformance_power_fdr(trueOutlierIndicesZeroOne, estimatedOutlierIndicesZeroOne, dataType = ""):
    
    ROUND_DIGITS = 2
    outlierRecall = getOutlierRecall(trueOutlierIndicesZeroOne, estimatedOutlierIndicesZeroOne, calculationForRPrecision = False)
    nrFalseDetections = getNrFalseDetections(trueOutlierIndicesZeroOne, estimatedOutlierIndicesZeroOne)
    
    nrDiscoveries = numpy.sum(estimatedOutlierIndicesZeroOne)
    assert(nrFalseDetections <= nrDiscoveries)
    
    # print("true number of outliers = ", numpy.sum(trueOutlierIndicesZeroOne))
    # print("estimated number of outliers = ", nrDiscoveries)
    
    if nrDiscoveries == 0:
        FDR = 0.0
    else:
        FDR = nrFalseDetections / nrDiscoveries
    
    # print(dataType + ": outlierRecall(power) = " + str(round(outlierRecall, ROUND_DIGITS)) + ", nrFalseDetections = " + str(nrFalseDetections) + ", FDR = " + str(round(FDR, ROUND_DIGITS)))
    
    return outlierRecall, nrFalseDetections, FDR




def showAvgAndStd_str(results_allFolds, ROUND_DIGITS = 5):
    m = numpy.nanmean(results_allFolds)
    std = numpy.nanstd(results_allFolds)
    if m == numpy.nan:
        return "-"
    else:
        # return f"{round(m, ROUND_DIGITS)} ({round(std, ROUND_DIGITS)})"
        return str(round(m, ROUND_DIGITS)) + "(" + str(round(std, ROUND_DIGITS)) + ")"


def showAvgAndStd(results_allFolds, ROUND_DIGITS = 2):
    m = numpy.nanmean(results_allFolds)
    std = numpy.nanstd(results_allFolds)
    return (round(m, ROUND_DIGITS), round(std, ROUND_DIGITS))


def getHighlightedResults(allResult_pairs, bestIsHigh):

    avgResults = numpy.asarray(allResult_pairs)[:, 0]

    if bestIsHigh is None:
        bestResult = numpy.nan
    else:
        if bestIsHigh:
            bestResult = numpy.nanmax(avgResults)
        else:
            bestResult = numpy.nanmin(avgResults)

    allResultStrs = []

    for avgRes, stdValue in allResult_pairs:
        if numpy.isnan(avgRes):
            allResultStrs.append("-")
            continue

        resStr = ""
        if avgRes == bestResult:
            resStr += "\\textbf{" + str(avgRes) + "}"
        else:
            resStr += str(avgRes)
        resStr += f" ({stdValue})"
        allResultStrs.append(resStr)

    return " & ".join(allResultStrs)


def showOneLineSummary(allOutlierRecall, allFDR, allNrFalseDetections, auc, outlierRecall_topNrTrueOutliers):
    numpy.set_printoptions(precision=2)
    print(f"POWER = {numpy.mean(allOutlierRecall, axis = 1)}, FDR = {numpy.mean(allFDR, axis = 1)}, nr false dectections = {numpy.mean(allNrFalseDetections, axis = 1)}, AUC = {numpy.mean(auc)}, topNrTrueOutlierRecall = {numpy.mean(outlierRecall_topNrTrueOutliers)}")
    return