from dataset.vl_cmu_cd import get_VL_CMU_CD, get_VL_CMU_CD_Raw
from dataset.pcd import get_pcd_cv, get_pcd_cv_wo_rot
from dataset.pcd import get_GSV, get_TSUNAMI
from dataset.changesim import get_ChangeSim_Binary, get_ChangeSim_Multi, get_ChangeSim_Semantic

dataset_dict = {
    "VL_CMU_CD": get_VL_CMU_CD,
    'VL_CMU_CD_Raw': get_VL_CMU_CD_Raw,
    'PCD_CV': get_pcd_cv,
    'PCD_CV_woRot': get_pcd_cv_wo_rot,
    'GSV': get_GSV,
    'TSUNAMI': get_TSUNAMI,
    'ChangeSim_Binary': get_ChangeSim_Binary,
    'ChangeSim_Multi': get_ChangeSim_Multi,
    'ChangeSim_Semantic': get_ChangeSim_Semantic,
}
