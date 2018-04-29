def informationGain(xSplit, ySplit):
	# going to assume that xSplit is left Branch
	# going to assume ySplit is right branch
	hSList = []
	temp = -1
	posNegativeRatio = 0.0
	negPositiveRatio = 0.0
	if (np.size(xSplit) > 0)
		posNegativeRatio = xSplit.numPositives/(float(xSplit.numPositives + xSplit.numNegatives))
		negPositiveRatio = xSplit.numNegatives/(float(xSplit.numPositives + xSplit.numNegatives))
	if posNegativeRatio == 0 or negPositiveRatio == 0:
		temp = 0
	else:
		temp = -posNegativeRatio*np.log2(posNegativeRatio)-negPositiveRatio*np.log2(negPositiveRatio)
	hSList.append(temp)
	temp = -1
	if (np.size(ySplit) > 0)
		posNegativeRatio = ySplit.numPositives/(float(ySplit.numPositives + ySplit.numNegatives))
		negPositiveRatio = ySplit.numPositives/(float(ySplit.numPositives + ySplit.numNegatives))
	if posNegativeRatio == 0 or negPositiveRatio == 0:
		temp = 0
	else:
		temp = -posNegativeRatio*np.log2(posNegativeRatio)-negPositiveRatio*np.log2(negPositiveRatio)
	hSList.append(temp)
	return hS-(xSplit.numPositives+ xSplit.numNegatives)/(float(np.size(X, 0)))*hSList[0]-(ySplit.numPositives+ ySplit.numNegatives)/(float(np.size(X, 0)))*hSList[1]