import pickle

with open("gt_catalog.pkl", "rb") as f:
    gt_catalog = pickle.load(f)

type(gt_catalog)
print(len(gt_catalog))
print(gt_catalog)