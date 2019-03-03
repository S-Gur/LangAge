import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.exceptions import UndefinedMetricWarning
from typing import NamedTuple
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# Structure for selected features:
class SlctFtr(NamedTuple):
    iFtrID: int
    strFtrName: str
    fFtrImpt: float

# Structure for results of a single LOPO:
class LOPORslt(NamedTuple):
    cSlctFtrs: list
    fAcc: float


def runRF_LOPO(i_arrDS, i_arrDSHdr, i_iNofTrees=50, i_iNofSlctFtrs=10, i_iNofFtrSlctRFs=100):
    """
    A function to run Random Forest (RF) with  i_iNofTrees trees, and i_iNofSlctFtrs features using
    RF-based (feature importance) feature selection (i_iNofFtrSlctRFs times bootstrap)
    Input:
        i_arrDS (2D numpy array of float): the dataset with row per instance and column per feature +
            column for particpant ID + column for label (order needs to correpond to order in i_arrDSHdr)
        i_arrDSHdr (1D numpy array of string): the names of the columns in i_arrDS (assuming 'ID' for
            participant IDs, 'Class' for the classification classes, and all the rest are features to
            be used (and selected from) in the classification
        i_iNofTrees (integer): The number of trees to be used in the RF (default: 50)
        i_iNofSlctFtrs (integer): The number of features to be selected. Feature selection is performed
            only if value is > 0 (default: 10)
        i_iNofFtrSlctRFs (integer): The number of RF to run for feature selection per LOPO iteration
            (defult: 100)
    Output:
        o_dLOPORslts (dictionary of LOPORslt class; size = #Participants): Key is ID of LOPO's test-set
            partcipant and values are the respective LOPO's list of SlctFtr class [the feature's ID, name,
            and importance (#of RF iterations in which feature was in top i_iNofSlctFtrs features for
            feature selection, or final RF model's feature importance in case of no feature selection)]
            and accuracy of the LOPO iteration
        o_arrPrds (2D array of float; size = #Instances X #Classes): The models' prediction per instance
            (probability of belonging to each class); order of instance corresponds to order in i_arrDS
    """

    arrData = np.copy(i_arrDS)
    arrFtrNames = np.copy(i_arrDSHdr)

    iNofInsts = arrData.shape[0]
    iNofFtrs = np.size(arrFtrNames) - 2 # Assuming anything other than 'ID' and 'Class' are features

    iIDsInd = np.where(arrFtrNames=='ID')[0]
    arrIDs = np.ndarray.flatten(arrData[:, iIDsInd])
    iLbslInd = np.where(arrFtrNames=='Class')[0]
    arrLbls = np.ndarray.flatten(arrData[:, iLbslInd])
    arrUnqIDs = np.unique(arrIDs)
    iNofCls = np.size(np.unique(arrLbls))
    iNofParticipants = np.size(arrUnqIDs)
    arrData = np.delete(arrData, [iIDsInd, iLbslInd], 1)
    arrFtrNames = np.delete(arrFtrNames, [iIDsInd, iLbslInd])

    o_dLOPORslts = dict()
    o_arrPrds = np.empty([iNofInsts, iNofCls])

    # LOPO per participant:
    for iItrInd in range(iNofParticipants):
        # Split the dataset into test (all records from single participants) and training (all records from
        # all other participants) set:
        iTstPrts = arrUnqIDs[iItrInd]
        tst_inds = np.ndarray.flatten(np.array([np.where(arrIDs == iTstPrts)[0]]))
        trn_inds = np.delete(np.arange(iNofInsts), tst_inds)
        data_test = np.copy(arrData[tst_inds, :])
        data_training = np.copy(arrData[trn_inds, :])

        # Get training and test sets' respecive labels:
        lbl_test = arrLbls[tst_inds]
        lbl_training = arrLbls[trn_inds]

        if (0 < i_iNofSlctFtrs):
            # Feature selection:
            arrTmps = np.empty([i_iNofFtrSlctRFs, i_iNofSlctFtrs])
            arrTmpImprts = np.empty([i_iNofFtrSlctRFs, i_iNofSlctFtrs])
            for iRnd in range(i_iNofFtrSlctRFs):
                cls = RFC(n_estimators=i_iNofTrees, random_state=iRnd)
                cls.fit(data_training, lbl_training)
                tmp = sorted(zip(map(lambda x: round(x, 4), cls.feature_importances_), np.arange(0, iNofFtrs)),
                             reverse=True)
                arrTmps[iRnd, :] = [x[1] for x in tmp[:i_iNofSlctFtrs]]
                arrTmpImprts[iRnd, :] = [x[0] for x in tmp[:i_iNofSlctFtrs]]
            tmpCount = [sum(sum(arrTmps==x)) for x in range(iNofFtrs)]
            tmp = sorted(zip(map(lambda x: round(x, 4), tmpCount), np.arange(0, iNofFtrs)), reverse=True)
            tmp = tmp[:i_iNofSlctFtrs]
            arrFtrInds = [x[1] for x in tmp]
            lstSlctFtrs = [SlctFtr(arrFtrInds[i], arrFtrNames[arrFtrInds[i]], tmpCount[tmp[i][1]])
                           for i in range(len(tmp))]

            # Updating training and test sets to include only selected features:
            data_training = data_training[:, arrFtrInds]
            data_test = data_test[:, arrFtrInds]
            # Re-training the RF model on the updated training set:
            cls.fit(data_training, lbl_training)
        else:
            # No feature selection
            cls = RFC(n_estimators=i_iNofTrees, random_state=i_iNofFtrSlctRFs)
            cls.fit(data_training, lbl_training)

            lstSlctFtrs = [SlctFtr(i, arrFtrNames[i], cls.feature_importances_[i]) for i in range(iNofFtrs)]

        o_dLOPORslts[int(iTstPrts)] = LOPORslt(lstSlctFtrs, cls.score(data_test, lbl_test))
        o_arrPrds[tst_inds] = cls.predict_proba(data_test)

    return o_dLOPORslts, o_arrPrds

