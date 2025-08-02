import pickle
import os

filepath = "../run/TS_CXR_Text/ihm-48-cxr-notes-ecg/TS_CXR_Text/TS_48/Atten/Text_48/bioLongformer/1024/hme/softmax_laplace/2_4/top_1_2/batch/irregular_TS_64/irregular_Text_64/use_pt_text_embeddings/2e-05_2_0_0.0004_8_8_128_1_1_512/"
with open(os.path.join(filepath, 'result.pkl'), 'rb') as f:
    result = pickle.load(f)

print(result)